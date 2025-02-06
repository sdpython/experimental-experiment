"""
.. _l-plot-torch-sklearn-201:

=======================================================
201: Use torch to export a scikit-learn model into ONNX
=======================================================>>>>>>

When :epkg:`sklearn-onnx` is missing a converter, :epkg:`torch` can be used
to write it. We use :class:`sklearn.impute.KNNImputer` as an example.
The first step is to rewrite the scikit-learn model with torch functions.
The code is then refactored and split into submodules to be able
to bypass some pieces :func:`torch.export.export` cannot process.

torch implementation of nan_euclidean
=====================================

Let's start with a simple case, a pairwise distance.
See :func:`sklearn.metrics.nan_euclidean`.

Module
++++++
"""

import contextlib
import io
import logging
import onnx
import sklearn
import torch
import onnxruntime
from experimental_experiment.helpers import max_diff
from experimental_experiment.skl.helpers import flatnonzero, _get_weights
from experimental_experiment.torch_interpreter import (
    to_onnx,
    ExportOptions,
    make_undefined_dimension,
)
from experimental_experiment.torch_interpreter.onnx_export_errors import (
    bypass_export_some_errors,
)
from experimental_experiment.torch_interpreter.piece_by_piece import (
    trace_execution_piece_by_piece,
    CustomOpStrategy,
)


class NanEuclidean(torch.nn.Module):
    """Implements :func:`sklearn.metrics.nan_euclidean`."""

    def __init__(self, squared=False, copy=True):
        super().__init__()
        self.squared = squared
        self.copy = copy

    def forward(self, X, Y):
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

        if not self.squared:
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
# The code is split into several :class:`torch.nn.Module`
# and refactored to avoid control flow.


def _get_mask(X, value_to_mask):
    return (
        torch.isnan(X)
        if sklearn.utils._missing.is_scalar_nan(value_to_mask)
        else (value_to_mask == X)
    )


class SubTopKIndices(torch.nn.Module):
    def forward(self, x, k):
        return torch.topk(x, k, dim=1, largest=False, sorted=True).indices


class SubWeightMatrix(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, donors_dist):
        weight_matrix = _get_weights(donors_dist, self.weights)

        # fill nans with zeros
        if weight_matrix is not None:
            weight_matrix = weight_matrix.clone()
            weight_matrix[torch.isnan(weight_matrix)] = 0.0
        else:
            weight_matrix = torch.ones_like(donors_dist)
            weight_matrix[torch.isnan(donors_dist)] = 0.0
        return weight_matrix


class SubTake(torch.nn.Module):
    def forward(self, fit_X_col, donors_idx):
        return fit_X_col.take(donors_idx)


class SubDonorsIdx(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._topk = SubTopKIndices()

    def forward(self, dist_pot_donors, n_neighbors):
        donors_idx = self._topk(dist_pot_donors, n_neighbors)
        # donors_idx = self._select(donors_idx, n_neighbors)

        # Get weight matrix from distance matrix
        donors_dist = dist_pot_donors[torch.arange(donors_idx.shape[0])[:, None], donors_idx]
        return donors_idx, donors_dist


class MakeDiv(torch.nn.Module):
    def forward(self, weights_sum):
        return torch.where(weights_sum == 0, 1, weights_sum)


class MakeMask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._take2 = SubTake()

    def forward(self, donors_idx, mask_fit_X_col):
        return torch.tensor([1], dtype=donors_idx.dtype) - self._take2(
            mask_fit_X_col, donors_idx
        ).to(donors_idx.dtype)


class MakeRes(torch.nn.Module):
    def forward(self, donors, new_weights, div):
        return (donors * new_weights).sum(axis=1, keepdim=True) / div


class MakeNewWeights(torch.nn.Module):
    def forward(self, donors_mask, donors, weight_matrix):
        new_weights = donors_mask.to(donors.dtype)
        return new_weights * weight_matrix.to(donors.dtype)


class CalcImpute(torch.nn.Module):
    """Implements :meth:`sklearn.impute.KNNImputer._calc_impute`."""

    def __init__(self, weights):
        super().__init__()
        self._take1 = SubTake()
        self._weights = SubWeightMatrix(weights)
        self._donors_idx = SubDonorsIdx()
        self._make_div = MakeDiv()
        self._make_mask = MakeMask()
        self._make_res = MakeRes()
        self._make_new_neights = MakeNewWeights()

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        donors_idx, donors_dist = self._donors_idx(dist_pot_donors, n_neighbors)
        weight_matrix = self._weights(donors_dist)
        # Retrieve donor values and calculate kNN average
        donors = self._take1(fit_X_col, donors_idx)
        donors_mask = self._make_mask(donors_idx, mask_fit_X_col)

        new_weights = self._make_new_neights(donors_mask, donors, weight_matrix)

        weights_sum = new_weights.sum(axis=1, keepdim=True)
        div = self._make_div(weights_sum)
        res = self._make_res(donors, new_weights, div)
        return res.squeeze(dim=1).to(dist_pot_donors.dtype)

    def forward(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        return self._calc_impute(dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col)


class ColProcessor(torch.nn.Module):
    """Processes one column (= one feature)."""

    def __init__(self, col, n_neighbors, weights):
        super().__init__()
        self._calc_impute = CalcImpute(weights)
        self.col = col
        self.n_neighbors = n_neighbors

    def process_one_col(
        self,
        X,
        dist_chunk,
        non_missing_fix_X,
        mask_fit_X,
        dist_idx_map,
        mask,
        row_missing_idx,
        _fit_X,
    ):
        col = self.col
        X = X.clone()
        row_missing_chunk = row_missing_idx
        col_mask = mask[row_missing_chunk, col]

        potential_donors_idx = torch.nonzero(non_missing_fix_X[:, col], as_tuple=True)[0]

        # receivers_idx are indices in X
        receivers_idx = row_missing_chunk[flatnonzero(col_mask)]

        # distances for samples that needed imputation for column
        dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]

        # receivers with all nan distances impute with mean
        all_nan_dist_mask = torch.isnan(dist_subset).all(axis=1)
        all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

        # when all_nan_receivers_idx is not empty (training set is small)
        mask_ = (~mask_fit_X[:, col]).to(_fit_X.dtype)
        mask_sum = mask_.to(X.dtype).sum()

        col_sum = (_fit_X[mask_ == 1, col]).sum().to(X.dtype)
        div = torch.where(mask_sum > 0, mask_sum, 1)
        X[all_nan_receivers_idx, col] = col_sum / div

        # receivers with at least one defined distance
        receivers_idx = receivers_idx[~all_nan_dist_mask]
        dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]

        # when all_nan_receivers_idx is not empty (training set is big)
        tn = torch.tensor(self.n_neighbors)
        n_neighbors = torch.where(
            tn < potential_donors_idx.shape[0], tn, potential_donors_idx.shape[0]
        )
        # to make sure n_neighbors > 0
        n_neighbors = torch.where(n_neighbors <= 0, torch.tensor(1), n_neighbors)
        value = self._calc_impute(
            dist_subset,
            n_neighbors,
            _fit_X[potential_donors_idx, col],
            mask_fit_X[potential_donors_idx, col],
        )
        X[receivers_idx, col] = value.to(X.dtype)
        return X

    def forward(
        self,
        X,
        dist_chunk,
        non_missing_fix_X,
        mask_fit_X,
        dist_idx_map,
        mask,
        row_missing_idx,
        _fit_X,
    ):
        return self.process_one_col(
            X,
            dist_chunk,
            non_missing_fix_X,
            mask_fit_X,
            dist_idx_map,
            mask,
            row_missing_idx,
            _fit_X,
        )


class MakeDictIdxMap(torch.nn.Module):
    def forward(self, X, row_missing_idx):
        dist_idx_map = torch.zeros(X.shape[0], dtype=int)
        dist_idx_map[row_missing_idx] = torch.arange(row_missing_idx.shape[0])
        return dist_idx_map


class TorchKNNImputer(torch.nn.Module):
    def __init__(self, knn_imputer):
        super().__init__()
        assert (
            knn_imputer.metric == "nan_euclidean"
        ), f"Not implemented for metric={knn_imputer.metric!r}"
        self.dist = NanEuclidean()
        cols = []
        for col in range(knn_imputer._fit_X.shape[1]):
            cols.append(ColProcessor(col, knn_imputer.n_neighbors, knn_imputer.weights))
        self.columns = torch.nn.ModuleList(cols)
        # refactoring
        self._make_dict_idx_map = MakeDictIdxMap()
        # knn imputer
        self.missing_values = knn_imputer.missing_values
        self.n_neighbors = knn_imputer.n_neighbors
        self.weights = knn_imputer.weights
        self.metric = knn_imputer.metric
        self.keep_empty_features = knn_imputer.keep_empty_features
        self.add_indicator = knn_imputer.add_indicator
        # results of fitting
        self.indicator_ = knn_imputer.indicator_
        # The training results.
        # self._fit_X = torch.from_numpy(knn_imputer._fit_X)
        # self._mask_fit_X = torch.from_numpy(knn_imputer._mask_fit_X)
        # self._valid_mask = torch.from_numpy(knn_imputer._valid_mask)

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

    def transform(self, mask_fit_X, _valid_mask, _fit_X, X):
        X = X.clone()
        mask = _get_mask(X, self.missing_values)

        X_indicator = self._transform_indicator(mask)

        row_missing_idx = flatnonzero(mask[:, _valid_mask].any(axis=1))
        non_missing_fix_X = torch.logical_not(mask_fit_X)

        # Maps from indices from X to indices in dist matrix
        dist_idx_map = self._make_dict_idx_map(X, row_missing_idx)

        # process in fixed-memory chunks
        pairwise_distances = self.dist(X[row_missing_idx, :], _fit_X)

        # The export unfold the loop as it depends on the number of features.
        # Fixed in this case.
        for col_processor in self.columns:
            X = col_processor(
                X,
                pairwise_distances,
                non_missing_fix_X,
                mask_fit_X,
                dist_idx_map,
                mask,
                row_missing_idx,
                _fit_X,
            )

        if self.keep_empty_features:
            Xc = X.clone()
            Xc[:, ~_valid_mask] = 0
        else:
            Xc = X[:, _valid_mask]

        return self._concatenate_indicator(Xc, X_indicator)

    def forward(self, _mask_fit_X, _valid_mask, _fit_X, X):
        return self.transform(_mask_fit_X, _valid_mask, _fit_X, X)


# %%
# Validation
# ++++++++++
#
# We need to do that with different sizes of training set.


def validate(size, sizey):
    X = torch.randn((size, 2))
    Y = torch.randn((sizey, 2))
    for i in range(5):
        X[i, i % 2] = torch.nan
    for i in range(4):
        Y[i + 1, i % 2] = torch.nan

    knn_imputer = sklearn.impute.KNNImputer(n_neighbors=3)
    knn_imputer.fit(X)

    model = TorchKNNImputer(knn_imputer)

    p1 = knn_imputer.transform(Y)
    p2 = model.transform(
        torch.from_numpy(knn_imputer._mask_fit_X),
        torch.from_numpy(knn_imputer._valid_mask),
        torch.from_numpy(knn_imputer._fit_X),
        Y,
    )
    d = max_diff(p1, p2)
    assert d["abs"] < 1e-5, f"Discrepancies for size={size}, d={d}"
    print(f"knn discrepancies for size={size}: {d}")

    p1 = knn_imputer.transform(Y[1:2])
    p2 = model.transform(
        torch.from_numpy(knn_imputer._mask_fit_X),
        torch.from_numpy(knn_imputer._valid_mask),
        torch.from_numpy(knn_imputer._fit_X),
        Y[1:2],
    )
    d = max_diff(p1, p2)
    assert d["abs"] < 1e-5, f"Discrepancies for size={size}, d={d}"
    print(f"knn discrepancies for size={size}: {d}")
    return knn_imputer, Y


knn5, Y10 = validate(5, 10)
knn50, Y40 = validate(50, 40)

# %%
# Export to ONNX
# ==============
#
# The module cannot be exported as is because one operator :func:`torch.topk`
# expects a fixed number of neighbour but the model makes it variable.
# This is case not supported by :func:`torch.export.export`.
# We need to isolate that part before exporting the model.
# It is done by replacing it with a custom op.
# This is automatically done by function :func:`trace_execution_piece_by_piece`.
#
# First step, we create two sets of inputs. A function will use this
# to infer the dynamic shapes.

inputs = [
    (
        (
            torch.from_numpy(knn50._mask_fit_X),
            torch.from_numpy(knn50._valid_mask),
            torch.from_numpy(knn50._fit_X),
            Y40,
        ),
        {},
    ),
    (
        (
            torch.from_numpy(knn5._mask_fit_X),
            torch.from_numpy(knn5._valid_mask),
            torch.from_numpy(knn5._fit_X),
            Y10,
        ),
        {},
    ),
]

# %%
# Then we trace the execution to capture every input and output of every submodule.
# The model implementation was refactored to introduce many tiny one and get
# a fine-grained evaluation of the exportability.

trace = trace_execution_piece_by_piece(TorchKNNImputer(knn5), inputs, verbose=0)
pretty = trace.get_export_report()
print(pretty)

# %%
# The method ``try_export`` cannot infer all links between input shapes and output shapes
# for every submodule. The following function fills this gap.

shape_functions = {
    "NanEuclidean": {
        0: lambda *args, **kwargs: torch.empty(
            (args[0].shape[0], args[1].shape[0]), dtype=args[0].dtype
        )
    },
    "CalcImpute": {
        0: lambda *args, **kwargs: torch.empty((args[0].shape[0],), dtype=args[0].dtype)
    },
    "SubTopKIndices": {
        0: lambda *args, **kwargs: torch.empty(
            (
                args[0].shape[0],
                make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
            ),
            dtype=args[0].dtype,
        )
    },
    "SubDonorsIdx": {
        0: lambda *args, **kwargs: torch.empty(
            (
                args[0].shape[0],
                make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
            ),
            dtype=args[0].dtype,
        ),
        1: lambda *args, **kwargs: torch.empty(
            (
                args[0].shape[0],
                make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
            ),
            dtype=torch.float32,
        ),
    },
    "MakeDictIdxMap": {
        0: lambda *args, **kwargs: torch.empty((args[0].shape[0],), dtype=args[1].dtype),
    },
}

# %%
# Then we we try to export piece by piece.
# We capture the standard output to avoid being overwhelmed
# and we use function :func:`bypass_export_some_errors` to skip some
# errors with shape checking made by :mod:`torch`.

logging.disable(logging.CRITICAL)

with contextlib.redirect_stderr(io.StringIO()), bypass_export_some_errors():
    ep = trace.try_export(
        exporter="fx",
        use_dynamic_shapes=True,
        exporter_kwargs=dict(strict=False),
        replace_by_custom_op=CustomOpStrategy.LOCAL,
        verbose=0,
        shape_functions=shape_functions,
    )

assert ep.status in (
    ep.status.OK,
    ep.status.OK_CHILDC,
), f"FAIL: {ep}\n-- report --\n{trace.get_export_report()}"
print(trace.get_export_report())

# %%
# ``OK`` means the module is exportable. ``OK_CHILDC`` means the module
# can be exported after its submodules are replaced by custom ops.
# It works except for the topk function. ``FAIL`` means
# the submodule cannot be exported at all but that
# module is sipple enough and its ONNX conversion can be provided.

# %%
# Final step
# ++++++++++
#
# Let's export everything. Every submodule is exported as a local function.
# That's make

# onx = trace.to_onnx()
onx = trace.to_onnx_local(verbose=0)

onnx.save(onx, "plot_torch_sklearn_201.onnx")
print(onx)


# %%
# Validation
# ==========


def validate_onnx(size):
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

    onx = to_onnx(
        model,
        (Y,),
        dynamic_shapes=({0: "batch"},),
        verbose=0,
        export_options=ExportOptions(strict=False),
    )

    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    got = sess.run(None, {"X": Y.numpy()})
    print(f"onnx discrepancies: {max_diff(p1, got[0])}")


validate_onnx(5)
validate_onnx(50)
