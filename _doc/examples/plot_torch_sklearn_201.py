"""
.. _l-plot-torch-sklearn-201:

=======================================================
201: Use torch to export a scikit-learn model into ONNX
=======================================================

When :epkg:`sklearn-onnx` is missing a converter, :epkg:`torch` can be used
to write it. We use :class:`sklearn.impute.KNNImputer` as an example.
The first step is to rewrite the scikit-learn model with torch functions.
The code is then refactored and split into submodules to be able
to bypass some pieces :func:`torch.export.export` cannot process.

torch implementation of nan_euclidean_distances
===============================================

Let's start with a simple case, a pairwise distance.
See :func:`sklearn.metrics.pairwise.nan_euclidean_distances`.

Module
++++++
"""

import contextlib
import io
import logging
import math
import numbers
import warnings
import numpy as np
import onnx
import sklearn
import torch
import onnxruntime
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.helpers.onnx_helper import pretty_onnx
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.skl.helpers import flatnonzero, _get_weights
from experimental_experiment.torch_interpreter import make_undefined_dimension
from onnx_diagnostic.torch_export_patches import torch_export_patches
from experimental_experiment.torch_interpreter.piece_by_piece import (
    trace_execution_piece_by_piece,
    CustomOpStrategy,
)
from experimental_experiment.xbuilder.reverse_graph_builder import to_graph_builder_code


class NanEuclidean(torch.nn.Module):
    """Implements :func:`sklearn.metrics.pairwise.nan_euclidean_distances`."""

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
        present_count = torch.maximum(torch.tensor([1], dtype=present_count.dtype), present_count)
        distances /= present_count
        distances *= X.shape[1]

        if not self.squared:
            distances = distances.sqrt()

        return distances


# %%
# Validation
# ++++++++++


def get_xy(sizex=5, sizey=3, col=3, n_nans=None):
    X = torch.randn((sizex, col))
    Y = torch.randn((sizey, col))
    i_nans = 0
    for i in range(sizex):
        X[i, i % X.shape[1]] = torch.nan
        i_nans += 1
        if n_nans and n_nans >= i_nans:
            break
        X[i, (i + 1) % X.shape[1]] = torch.nan
        i_nans += 1
        if n_nans and n_nans >= i_nans:
            break
    i_nans = 0
    for i in range(sizey):
        Y[(i + 1) % sizey, i % Y.shape[1]] = torch.nan
        i_nans += 1
        if n_nans and n_nans >= i_nans:
            break
        Y[(i + 1) % sizey, (i + 1) % Y.shape[1]] = torch.nan
        i_nans += 1
        if n_nans and n_nans >= i_nans:
            break
    return X, Y


X, Y = get_xy()
model = NanEuclidean()


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
#
# Module and sub modules
# ++++++++++++++++++++++


def _get_mask(X, value_to_mask):
    return (
        torch.isnan(X)
        if (  # sklearn.utils._missing.is_scalar_nan(value_to_mask)
            not isinstance(value_to_mask, numbers.Integral)
            and isinstance(value_to_mask, numbers.Real)
            and math.isnan(value_to_mask)
        )
        else (value_to_mask == X)
    )


class SubWeightMatrix(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, donors_dist):
        weight_matrix = _get_weights(donors_dist, self.weights)
        if weight_matrix is not None:
            weight_matrix = weight_matrix.clone()
            weight_matrix[torch.isnan(weight_matrix)] = 0.0
        else:
            weight_matrix = torch.ones_like(donors_dist)
            weight_matrix[torch.isnan(donors_dist)] = 0.0
        return weight_matrix


class SubDonorsIdx(torch.nn.Module):
    def forward(self, dist_pot_donors, n_neighbors):
        xn = torch.nan_to_num(dist_pot_donors, nan=1.0e10)
        tk = torch.topk(xn, n_neighbors, dim=1, largest=False, sorted=True)
        return tk.indices, tk.values


class MakeNewWeights(torch.nn.Module):
    def forward(self, donors_mask, donors, weight_matrix):
        return donors_mask.to(donors.dtype) * weight_matrix.to(donors.dtype)


class CalcImpute(torch.nn.Module):
    """Implements :meth:`sklearn.impute.KNNImputer._calc_impute`."""

    def __init__(self, weights):
        super().__init__()
        self._weights = SubWeightMatrix(weights)
        self._donors_idx = SubDonorsIdx()
        self._make_new_neights = MakeNewWeights()

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        donors_idx, donors_dist = self._donors_idx(dist_pot_donors, n_neighbors)
        weight_matrix = self._weights(donors_dist)
        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = torch.tensor([1], dtype=donors_idx.dtype) - (
            mask_fit_X_col.take(donors_idx)
        ).to(donors_idx.dtype)

        new_weights = self._make_new_neights(donors_mask, donors, weight_matrix)

        weights_sum = new_weights.sum(axis=1, keepdim=True)
        div = torch.where(
            weights_sum == 0, torch.tensor([1], dtype=weights_sum.dtype), weights_sum
        )
        res = (donors * new_weights).sum(axis=1, keepdim=True) / div
        return res.squeeze(dim=1).to(dist_pot_donors.dtype)

    def forward(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        return self._calc_impute(dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col)


class ColProcessorAllNan(torch.nn.Module):
    def __init__(self, col: int):
        super().__init__()
        self.col = col

    def forward(
        self,
        X,
        dist_subset,
        mask_fit_X,
        _fit_X,
        receivers_idx,
        all_nan_receivers_idx,
        all_nan_dist_mask,
        dist_chunk,
        dist_idx_map,
        potential_donors_idx,
    ):
        col = self.col
        X = X.clone()
        mask_ = (~mask_fit_X[:, col]).to(_fit_X.dtype)
        mask_sum = mask_.to(X.dtype).sum()

        col_sum = (_fit_X[mask_ == 1, col]).sum().to(X.dtype)
        div = torch.where(mask_sum > 0, mask_sum, torch.tensor([1], dtype=mask_sum.dtype))
        X[all_nan_receivers_idx, col] = col_sum / div

        # receivers with at least one defined distance
        receivers_idx = receivers_idx[~all_nan_dist_mask]
        dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]
        return X, dist_subset, receivers_idx


class ColProcessorIdentity(torch.nn.Module):
    def forward(
        self,
        X,
        dist_subset,
        mask_fit_X,
        _fit_X,
        receivers_idx,
        all_nan_receivers_idx,
        all_nan_dist_mask,
        dist_chunk,
        dist_idx_map,
        potential_donors_idx,
    ):
        # .clone() not efficient but torch.cond does not like simple return
        return (
            X.contiguous(),
            dist_subset.contiguous(),
            receivers_idx.contiguous(),
        )


class ColProcessorCond(torch.nn.Module):
    def __init__(self, col: int):
        super().__init__()
        self.col = col
        self._all_nan = ColProcessorAllNan(col)
        self._identity = ColProcessorIdentity()

    def forward(
        self,
        X,
        dist_subset,
        mask_fit_X,
        _fit_X,
        receivers_idx,
        all_nan_receivers_idx,
        all_nan_dist_mask,
        dist_chunk,
        dist_idx_map,
        potential_donors_idx,
    ):
        X, dist_subset, receivers_idx = torch.cond(
            all_nan_receivers_idx.numel() > 0,
            self._all_nan,
            self._identity,
            [
                X,
                dist_subset,
                mask_fit_X,
                _fit_X,
                receivers_idx,
                all_nan_receivers_idx,
                all_nan_dist_mask,
                dist_chunk,
                dist_idx_map,
                potential_donors_idx,
            ],
        )
        return X.contiguous(), dist_subset.contiguous(), receivers_idx.contiguous()


class ColProcessor(torch.nn.Module):
    """Processes one column (= one feature)."""

    def __init__(self, col, n_neighbors, weights):
        super().__init__()
        self._calc_impute = CalcImpute(weights)
        self._col_cond = ColProcessorCond(col)
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
        # ... if all_nan_receivers_idx.size > 0:
        #    # onnxruntime does not like this part when it is empty
        #    mask_ = (~mask_fit_X[:, col]).to(_fit_X.dtype)
        #    mask_sum = mask_.to(X.dtype).sum()
        #
        #    col_sum = (_fit_X[mask_ == 1, col]).sum().to(X.dtype)
        #    div = torch.where(mask_sum > 0, mask_sum, torch.tensor([1], dtype=mask_sum.dtype))
        #    X[all_nan_receivers_idx, col] = col_sum / div
        #
        #     # receivers with at least one defined distance
        #     receivers_idx = receivers_idx[~all_nan_dist_mask]
        #     dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]
        # else
        #     ... identity
        X, dist_subset, receivers_idx = self._col_cond(
            X,
            dist_subset,
            mask_fit_X,
            _fit_X,
            receivers_idx,
            all_nan_receivers_idx,
            all_nan_dist_mask,
            dist_chunk,
            dist_idx_map,
            potential_donors_idx,
        )

        # when all_nan_receivers_idx is not empty (training set is big)
        tn = torch.tensor(self.n_neighbors)
        n_neighbors = torch.where(
            tn < potential_donors_idx.shape[0], tn, potential_donors_idx.shape[0]
        )
        # to make sure n_neighbors > 0
        n_neighbors = torch.where(
            n_neighbors <= 0, torch.tensor([1], dtype=n_neighbors.dtype), n_neighbors
        )
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
        # self._fit_X = torch.from_numpy(knn_imputer._fit_X.astype(np.float32))
        # self._mask_fit_X = torch.from_numpy(knn_imputer._mask_fit_X)
        # self._valid_mask = torch.from_numpy(knn_imputer._valid_mask)

    def _transform_indicator(self, X):
        if self.add_indicator:
            if not hasattr(self, "indicator_"):
                raise ValueError("Make sure to call _fit_indicator before _transform_indicator")
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


def validate(size, sizey, col=3, n_nans=None):
    X, Y = get_xy(size, sizey, col, n_nans)
    knn_imputer = sklearn.impute.KNNImputer(n_neighbors=3)
    knn_imputer.fit(X)

    model = TorchKNNImputer(knn_imputer)

    p1 = knn_imputer.transform(Y)
    p2 = model.transform(
        torch.from_numpy(knn_imputer._mask_fit_X),
        torch.from_numpy(knn_imputer._valid_mask),
        torch.from_numpy(knn_imputer._fit_X.astype(np.float32)),
        Y,
    )
    d = max_diff(p1, p2)
    assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
    print(f"knn discrepancies for size={size}: {d}")

    p1 = knn_imputer.transform(Y[1:2])
    p2 = model.transform(
        torch.from_numpy(knn_imputer._mask_fit_X),
        torch.from_numpy(knn_imputer._valid_mask),
        torch.from_numpy(knn_imputer._fit_X.astype(np.float32)),
        Y[1:2],
    )
    d = max_diff(p1, p2)
    assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
    print(f"knn discrepancies for size={size}: {d}")
    return knn_imputer, Y


knn5, Y10 = validate(5, 10)
knn50, Y40 = validate(50, 40)
knn1, Y1 = validate(10, 10, n_nans=1)
knn11, Y11 = validate(11, 11, n_nans=1)

# %%
# Export to ONNX
# ==============
#
# The module cannot be exported as is because one operator :func:`torch.topk`
# expects a fixed number of neighbour but the model makes it variable.
# This is case not supported by :func:`torch.export.export`.
# We need to isolate that part before exporting the model.
# It is done by replacing it with a custom op.
# This is automatically done by function
# :func:`trace_execution_piece_by_piece
# <experimental_experiment.torch_interpreter.piece_by_piece.trace_execution_piece_by_piece>`.
#
# First step, we create two sets of inputs. A function will use this
# to infer the dynamic shapes.
#
# First step: tracing intermediate outputs
# ++++++++++++++++++++++++++++++++++++++++

used = [(knn50, Y40), (knn5, Y10), (knn1, Y1), (knn11, Y11)]
inputs = [
    (
        (
            torch.from_numpy(knn._mask_fit_X),
            torch.from_numpy(knn._valid_mask),
            torch.from_numpy(knn._fit_X.astype(np.float32)),
            y,
        ),
        {},
    )
    for knn, y in used
]

# %%
# Then we trace the execution to capture every input and output of every submodule.
# The model implementation was refactored to introduce many tiny one and get
# a fine-grained evaluation of the exportability.
# We track messages such as ``-needs-more-inputs`` or ``-no-input``.
# If any, we must provide the tracer more input to make sure
# every submodule receives enough data to guess dynamic shapes and export.
# When the model has control flow, we need more data to make sure every
# piece is used.

trace = trace_execution_piece_by_piece(TorchKNNImputer(knn5), inputs, verbose=0)
pretty = trace.get_export_report()
print(pretty)

# %%
# We need more so let's add more.


def rotate(inputs, col=3):
    if isinstance(inputs, torch.Tensor):
        if len(inputs.shape) == 2 and inputs.shape[1] == 3:
            return torch.cat([inputs[:, 1:], inputs[:, :1]], axis=1)
        if len(inputs.shape) == 1 and inputs.shape[0] == 3:
            return torch.cat([inputs[1:], inputs[:1]], axis=0)
        return inputs
    if isinstance(inputs, tuple):
        return tuple(rotate(i, col=col) for i in inputs)
    if isinstance(inputs, list):
        return [rotate(i, col=col) for i in inputs]
    if isinstance(inputs, dict):
        return {k: rotate(v, col=col) for k, v in inputs.items()}
    raise TypeError(f"Unexpected type {type(inputs)}")


inputs = [*inputs, *rotate(inputs), *rotate(rotate(inputs))]

# %%
# Let's try again.

print("---------")
trace = trace_execution_piece_by_piece(TorchKNNImputer(knn5), inputs, verbose=0)
pretty = trace.get_export_report()
print(pretty)

# %%
# The dynamic shapes for the whole model:
print("dynamic shapes:")
print(trace.guess_dynamic_shapes())

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
    "SubDonorsIdx": {
        0: lambda *args, **kwargs: torch.empty(
            (
                make_undefined_dimension(111111),  # args[0].shape[0]),
                make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
            ),
            dtype=args[0].dtype,
        ),
        1: lambda *args, **kwargs: torch.empty(
            (
                make_undefined_dimension(111111),  # args[0].shape[0]),
                make_undefined_dimension(min(args[0].shape[1], knn5.n_neighbors)),
            ),
            dtype=torch.float32,
        ),
    },
    "MakeDictIdxMap": {
        0: lambda *args, **kwargs: torch.empty((args[0].shape[0],), dtype=args[1].dtype),
    },
    "ColProcessorCond": {
        0: lambda *args, **kwargs: torch.empty(args[0], dtype=args[0].dtype),
        1: lambda *args, **kwargs: torch.empty(
            make_undefined_dimension(0), args[1].shape[1], dtype=args[0].dtype
        ),
        2: lambda *args, **kwargs: torch.empty(
            (make_undefined_dimension(0),), dtype=args[0].dtype
        ),
    },
}

# %%
# Then we we try to export piece by piece.
# We capture the standard output to avoid being overwhelmed
# and we use function
# :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
# to skip some errors with shape checking made by :mod:`torch`.

logging.disable(logging.CRITICAL)

with contextlib.redirect_stderr(io.StringIO()), torch_export_patches():
    ep = trace.try_export(
        exporter="fx",
        use_dynamic_shapes=True,
        exporter_kwargs=dict(strict=False),
        replace_by_custom_op=CustomOpStrategy.LOCAL,
        verbose=0,
        shape_functions=shape_functions,
        quiet=1,
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
# module is simple enough and its ONNX conversion can be provided.

# %%
# Final step
# ++++++++++
#
# We first start by running the decompositions on every exported program.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for t in trace:
        if t.exporter_status.exported is None:
            print(f"[run_decompositions] {t.dot_name} - skipped")
            continue
        print(f"[run_decompositions] {t.dot_name}")
        t.exporter_status.exported = t.exporter_status.exported.run_decompositions({})

# %%
# Let's run the conversion. We also check the conversion into ONNX
# is accurate. It is doable because every intermediate results
# were previously traced.
try:
    onx = trace.to_onnx_local(
        verbose=1,
        check_conversion_cls=dict(cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5),
        inline=False,
    )
except Exception as e:
    print(f"The example is broken: {e}")
    onx = None

# %%
# Let's save it.
if onx:
    onnx.save(onx, "plot_torch_sklearn_201.onnx")

# %%
# We can also print it.
if onx:
    print(pretty_onnx(onx))


# %%
# Validation again
# ++++++++++++++++


def validate_onnx(size, sizey, onx, verbose: int = 1, use_ort: bool = False, col: int = 3):
    X, Y = get_xy(size, sizey, col=col)

    knn_imputer = sklearn.impute.KNNImputer(n_neighbors=3)
    knn_imputer.fit(X)

    model = TorchKNNImputer(knn_imputer)

    expected = p1 = knn_imputer.transform(Y)

    model_inputs = (
        torch.from_numpy(knn_imputer._mask_fit_X),
        torch.from_numpy(knn_imputer._valid_mask),
        torch.from_numpy(knn_imputer._fit_X.astype(np.float32)),
        Y,
    )
    p2 = model.transform(*model_inputs)
    d = max_diff(p1, p2)
    assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
    if verbose:
        print(f"Torch Discrepancies for size={size} and sizey={sizey}, d={d}")

    input_names = [i.name for i in onx.graph.input]
    feeds = feeds0 = dict(zip(input_names, [t.numpy() for t in model_inputs]))

    if verbose:
        print("python: loading the model...")
    sess = ExtendedReferenceEvaluator(onx, verbose=0)
    if verbose:
        print("python: running the model...")
    got = sess.run(None, feeds)
    d = max_diff(p1, got[0])
    assert d["abs"] < 1e-5, f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}"
    if verbose:
        print(f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}")

    if use_ort:
        if verbose:
            print("onnxruntime: loading the model...")
        opts = onnxruntime.SessionOptions()
        opts.optimized_model_filepath = "plot_torch_sklearn_201.ort.onnx"
        opts.log_severity_level = 0
        opts.log_verbosity_level = 0
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        if verbose:
            print("onnxruntime: running the model...")
        got = sess.run(None, feeds)
        d = max_diff(p1, got[0])
        assert d["abs"] < 1e-5, f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}"
        if verbose:
            print(f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}")

    model_inputs = (
        torch.from_numpy(knn_imputer._mask_fit_X),
        torch.from_numpy(knn_imputer._valid_mask),
        torch.from_numpy(knn_imputer._fit_X.astype(np.float32)),
        Y[1:2],
    )
    p1 = knn_imputer.transform(Y[1:2])
    p2 = model.transform(*model_inputs)
    d = max_diff(p1, p2)
    assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
    feeds = dict(zip(input_names, [t.numpy() for t in model_inputs]))
    if verbose:
        print("ReferenceEvaluator: running the model...")
    got = sess.run(None, feeds)
    d = max_diff(p1, got[0])
    assert d["abs"] < 1e-5, f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}"
    if verbose:
        print("done")
    return feeds0, expected


# %%
# This does not work yet.
if onx:
    feeds, expected = validate_onnx(5, 10, onx)
    validate_onnx(50, 40, onx)

# %%
# ModelProto to python Code
# =========================
#
# We finally call function :func:`to_graph_builder_code
# <experimental_experiment.xbuilder.reverse_graph_builder.to_graph_builder_code>`
# to convert the onnx model into pseudo code if that helps moving that code
# to a converter library (:epkg:`sklearn-onnx`).

if onx:
    code = to_graph_builder_code(onx)
    addition = f"""

    feeds = {feeds!r}
    expected = {expected!r}
    ref = ExtendedReferenceEvaluator(model)
    got = ref.run(None, feeds)
    print("disrepancies:", max_diff(expected, got[0]))
    """.replace("nan", "np.nan").replace("array", "np.array").replace("float32", "np.float32")
    code = f"""
    from experimental_experiment.reference import ExtendedReferenceEvaluator
    from experimental_experiment.helpers import max_diff
    {code}
    {addition}
    """
    print(code)


# %%
# Let's finally check it produces the same results.
if onx:
    with open("_plot_torch_sklearn_201_knnpy.py", "w") as f:
        f.write(code)

# %%
# Let's run it...
# It can be run this way.
#
# ``subprocess.run([sys.executable, "_plot_torch_sklearn_201_knnpy.py"])``
