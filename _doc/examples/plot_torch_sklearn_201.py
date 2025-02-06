"""
.. _l-plot-torch-sklearn-201:

=======================================================
201: Use torch to export a scikit-learn model into ONNX
=======================================================>>>>>>

When :epkg:`sklearn-onnx` is missing a converter, :epkg:`torch` can be used
to write it. We use :class:`sklearn.impute.KNNImputer` as an example.
The first step is to rewrite the scikit-learn model with torch functions.
We introduce two modules implementing the following
from :epkg:`scikit-learn`.

* :func:`sklearn.metrics.nan_euclidean`
* :class:`sklearn.impute.KNNImputer`

torch implementation of nan_euclidean
=====================================

See :func:`sklearn.metrics.nan_euclidean`.

Module
++++++
"""

import onnx
import sklearn
import torch
import onnxruntime
from experimental_experiment.helpers import max_diff
from experimental_experiment.skl.helpers import flatnonzero, _get_weights
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.torch_interpreter.onnx_export_errors import (
    bypass_export_some_errors,
)


class NanEuclidean(torch.nn.Module):
    """Implements :func:`sklearn.metrics.nan_euclidean`."""

    def forward(self, X, Y, squared=False, copy=True):
        X = X.clone()
        Y = Y.to(X.dtype).clone()

        missing_X = torch.isnan(X)
        missing_Y = torch.isnan(Y)

        # set missing values to zero
        X[missing_X] = 0
        Y[missing_Y] = 0

        # Adjust distances for missing values
        XX = X * X
        YY = Y * Y

        distances = -2 * X @ Y.T + XX.sum(1, keepdim=True) + YY.sum(1, keepdim=True).T

        distances -= XX @ missing_Y.to(X.dtype).T
        distances -= missing_X.to(X.dtype) @ YY.T

        distances = torch.clip(distances, 0, None)

        present_X = 1 - missing_X.to(X.dtype)
        present_Y = ~missing_Y
        present_count = present_X @ present_Y.to(X.dtype).T
        distances[present_count == 0] = torch.nan
        # avoid divide by zero
        present_count = torch.maximum(
            torch.tensor([1], dtype=present_count.dtype), present_count
        )
        distances /= present_count
        distances *= X.shape[1]

        if not squared:
            distances = distances.sqrt()

        return distances


# %%
# Validation
# ++++++++++

model = NanEuclidean()
X = torch.randn((5, 2))
Y = torch.randn((5, 2))
for i in range(5):
    X[i, i % 2] = torch.nan
for i in range(4):
    Y[i + 1, i % 2] = torch.nan

d1 = sklearn.metrics.nan_euclidean_distances(X.numpy(), Y.numpy())
d2 = model(X, Y)
print(f"discrepancies: {max_diff(d1, d2)}")


# %%
# torch implementation of KNNImputer
# ==================================
#
# See :class:`sklearn.impute.KNNImputer`.


def _get_mask(X, value_to_mask):
    return (
        torch.isnan(X)
        if sklearn.utils._missing.is_scalar_nan(value_to_mask)
        else (value_to_mask == X)
    )


class TorchKNNImputer(torch.nn.Module):
    def __init__(self, knn_imputer):
        super().__init__()
        assert (
            knn_imputer.metric == "nan_euclidean"
        ), f"Not implemented for metric={knn_imputer.metric!r}"
        self.dist = NanEuclidean()
        self.missing_values = knn_imputer.missing_values
        self.n_neighbors = knn_imputer.n_neighbors
        self.weights = knn_imputer.weights
        self.metric = knn_imputer.metric
        self.keep_empty_features = knn_imputer.keep_empty_features
        self.add_indicator = knn_imputer.add_indicator
        # results of fitting
        self.indicator_ = knn_imputer.indicator_
        self._fit_X = torch.from_numpy(knn_imputer._fit_X)
        self._mask_fit_X = torch.from_numpy(knn_imputer._mask_fit_X)
        # We need this to be a constant.
        self._valid_mask = tuple(map(bool, knn_imputer._valid_mask))

    def _transform_indicator(self, X):
        if self.add_indicator:
            if not hasattr(self, "indicator_"):
                raise ValueError(
                    "Make sure to call _fit_indicator before _transform_indicator"
                )
            raise NotImplementedError(type(self.indicator_))
            # return self.indicator_.transform(X)
        return None

    def _concatenate_indicator(self, X_imputed, X_indicator):
        if not self.add_indicator:
            return X_imputed
        if X_indicator is None:
            raise ValueError(
                "Data from the missing indicator are not provided. Call "
                "_fit_indicator and _transform_indicator in the imputer "
                "implementation."
            )
        return torch.cat([X_imputed, X_indicator], dim=0)

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        donors_idx = torch.topk(
            dist_pot_donors, n_neighbors, dim=1, largest=False, sorted=True
        )
        donors_idx = donors_idx.indices[:, :n_neighbors]

        # Get weight matrix from distance matrix
        donors_dist = dist_pot_donors[torch.arange(donors_idx.shape[0])[:, None], donors_idx]

        weight_matrix = _get_weights(donors_dist, self.weights)

        # fill nans with zeros
        if weight_matrix is not None:
            weight_matrix[torch.isnan(weight_matrix)] = 0.0
        else:
            weight_matrix = torch.ones_like(donors_dist)
            weight_matrix[torch.isnan(donors_dist)] = 0.0

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = 1 - mask_fit_X_col.take(donors_idx).to(int)

        new_weights = donors_mask.to(donors.dtype)
        new_weights *= weight_matrix.to(donors.dtype)

        weights_sum = new_weights.sum(axis=1, keepdim=True)
        div = torch.where(weights_sum == 0, 1, weights_sum)
        res = (donors * new_weights).sum(axis=1, keepdim=True) / div
        return res.squeeze(dim=1)

    def transform(self, X):
        X = X.clone()
        mask = _get_mask(X, self.missing_values)
        mask_fit_X = self._mask_fit_X

        X_indicator = self._transform_indicator(mask)

        row_missing_idx = flatnonzero(mask[:, self._valid_mask].any(axis=1))
        non_missing_fix_X = torch.logical_not(mask_fit_X)

        # Maps from indices from X to indices in dist matrix
        dist_idx_map = torch.zeros(X.shape[0], dtype=int)
        dist_idx_map[row_missing_idx] = torch.arange(row_missing_idx.shape[0])

        # process in fixed-memory chunks
        pairwise_distances = self.dist(
            X[row_missing_idx, :],
            self._fit_X,
            # missing_values=self.missing_values, # not supported for the time being
        )

        def process_one_col(dist_chunk, col):
            row_missing_chunk = row_missing_idx
            col_mask = mask[row_missing_chunk, col]

            # Should we keep this?
            # if not torch.any(col_mask):
            #     # column has no missing values
            #     return

            potential_donors_idx = torch.nonzero(non_missing_fix_X[:, col], as_tuple=True)[0]

            # receivers_idx are indices in X
            receivers_idx = row_missing_chunk[flatnonzero(col_mask)]

            # distances for samples that needed imputation for column
            dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]

            # receivers with all nan distances impute with mean
            all_nan_dist_mask = torch.isnan(dist_subset).all(axis=1)
            all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

            # when all_nan_receivers_idx is empty (training set is small)
            mask_ = (~mask_fit_X[:, col]).to(self._fit_X.dtype)
            mask_sum = mask_.to(X.dtype).sum()

            col_sum = (self._fit_X[mask_ == 1, col]).sum().to(X.dtype)
            div = torch.where(mask_sum > 0, mask_sum, 1)
            X[all_nan_receivers_idx, col] = col_sum / div

            # receivers with at least one defined distance
            receivers_idx = receivers_idx[~all_nan_dist_mask]
            dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]

            # when all_nan_receivers_idx is not empty (training set is big)
            n_neighbors = min(self.n_neighbors, len(potential_donors_idx))
            value = self._calc_impute(
                dist_subset,
                n_neighbors,
                self._fit_X[potential_donors_idx, col],
                mask_fit_X[potential_donors_idx, col],
            )
            X[receivers_idx, col] = value.to(X.dtype)

        for col in range(X.shape[1]):
            if not self._valid_mask[col]:
                # column was all missing during training
                return

            process_one_col(pairwise_distances, col)

        if self.keep_empty_features:
            Xc = X
            Xc[:, ~self._valid_mask] = 0
        else:
            Xc = X[:, self._valid_mask]

        return self._concatenate_indicator(Xc, X_indicator)

    def forward(self, X):
        return self.transform(X)


# %%
# Validation
# ++++++++++
#
# We need to do that with different sizes of training set.


def validate(size):
    X = torch.randn((size, 2))
    Y = torch.randn((5, 2))
    for i in range(5):
        X[i, i % 2] = torch.nan
    for i in range(4):
        Y[i + 1, i % 2] = torch.nan

    knn_imputer = sklearn.impute.KNNImputer(n_neighbors=3)
    knn_imputer.fit(X)

    model = TorchKNNImputer(knn_imputer)

    p1 = knn_imputer.transform(Y)
    p2 = model.transform(Y)
    print(f"knn discrepancies for size={size}: {max_diff(p1, p2)}")

    p1 = knn_imputer.transform(Y[1:2])
    p2 = model.transform(Y[1:2])
    print(f"knn discrepancies for size={size}: {max_diff(p1, p2)}")


validate(5)
validate(50)

# %%
# Export to ONNX
# ==============
# import torch.export._draft_export

# ep, report = torch.export._draft_export.draft_export(model, (Y,), strict=False)
# print("------")
# print(ep)
# print("------")
# print(report)
# print("------")
# stop
with bypass_export_some_errors():
    torch.export.export(
        model, (Y,), dynamic_shapes=({0: torch.export.Dim.DYNAMIC},), strict=False
    )
onx = to_onnx(
    model,
    (Y,),
    dynamic_shapes=({0: "batch"},),
    verbose=1,
    export_options=ExportOptions(strict=False),
)
onnx.save(onx, "plot_torch_sklearn_201.onnx")

# %%
# Validation
# ==========

sess = onnxruntime.InferenceSession(
    onx.SerializeToString(), providers=["CPUExecutionProvider"]
)
got = sess.run(None, {"X": Y.numpy()})
print(f"onnx discrepancies: {max_diff(p1, got[0])}")
