import contextlib
import io
import math
import numbers
import unittest
from typing import Any, Dict, List, Optional
from onnx_diagnostic.helpers import max_diff
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
)
from experimental_experiment.xbuilder import GraphBuilder
from experimental_experiment.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from experimental_experiment.torch_interpreter.piece_by_piece import (
    trace_execution_piece_by_piece,
    CustomOpStrategy,
)


def traceable_local_f(x, y):
    return x.abs() + y.abs() + 1e-5


def traceable_local_f_recursive(x, y):
    return traceable_local_f(x, y)


class TestPieceByPieceBig(ExtTestCase):
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_e2e_knn_imputer(self):
        import onnx
        import sklearn
        import torch
        from experimental_experiment.skl.helpers import flatnonzero, _get_weights
        from experimental_experiment.torch_interpreter import (
            make_undefined_dimension,
            Dispatcher,
        )

        class NanEuclidean(torch.nn.Module):
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

        class SubTopKIndices(torch.nn.Module):
            def forward(self, x, k):
                # torch does not like nans
                xn = torch.nan_to_num(x, nan=1.0e10)
                return torch.topk(xn, k, dim=1, largest=False, sorted=True).indices

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
            def __init__(self):
                super().__init__()
                self._topk = SubTopKIndices()

            def forward(self, dist_pot_donors, n_neighbors):
                donors_idx = self._topk(dist_pot_donors, n_neighbors)
                donors_dist = dist_pot_donors[
                    torch.arange(donors_idx.shape[0])[:, None], donors_idx
                ]
                return donors_idx, donors_dist

        class MakeNewWeights(torch.nn.Module):
            def forward(self, donors_mask, donors, weight_matrix):
                return donors_mask.to(donors.dtype) * weight_matrix.to(donors.dtype)

        class CalcImpute(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self._weights = SubWeightMatrix(weights)
                self._donors_idx = SubDonorsIdx()
                self._make_new_neights = MakeNewWeights()

            def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
                donors_idx, donors_dist = self._donors_idx(dist_pot_donors, n_neighbors)
                weight_matrix = self._weights(donors_dist)
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
                return self._calc_impute(
                    dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col
                )

        class ColProcessor(torch.nn.Module):
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

                potential_donors_idx = torch.nonzero(non_missing_fix_X[:, col], as_tuple=True)[
                    0
                ]
                receivers_idx = row_missing_chunk[flatnonzero(col_mask)]
                dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]
                all_nan_dist_mask = torch.isnan(dist_subset).all(axis=1)
                all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]
                mask_ = (~mask_fit_X[:, col]).to(_fit_X.dtype)
                mask_sum = mask_.to(X.dtype).sum()
                col_sum = (_fit_X[mask_ == 1, col]).sum().to(X.dtype)
                div = torch.where(
                    mask_sum > 0, mask_sum, torch.tensor([1], dtype=mask_sum.dtype)
                )
                X[all_nan_receivers_idx, col] = col_sum / div
                receivers_idx = receivers_idx[~all_nan_dist_mask]
                dist_subset = dist_chunk[dist_idx_map[receivers_idx]][:, potential_donors_idx]
                tn = torch.tensor(self.n_neighbors)
                n_neighbors = torch.where(
                    tn < potential_donors_idx.shape[0], tn, potential_donors_idx.shape[0]
                )
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
                    cols.append(
                        ColProcessor(col, knn_imputer.n_neighbors, knn_imputer.weights)
                    )
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

            def _transform_indicator(self, X):
                if self.add_indicator:
                    if not hasattr(self, "indicator_"):
                        raise ValueError(
                            "Make sure to call _fit_indicator before _transform_indicator"
                        )
                    raise NotImplementedError(type(self.indicator_))
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
                dist_idx_map = self._make_dict_idx_map(X, row_missing_idx)
                pairwise_distances = self.dist(X[row_missing_idx, :], _fit_X)
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
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"

            p1 = knn_imputer.transform(Y[1:2])
            p2 = model.transform(
                torch.from_numpy(knn_imputer._mask_fit_X),
                torch.from_numpy(knn_imputer._valid_mask),
                torch.from_numpy(knn_imputer._fit_X),
                Y[1:2],
            )
            d = max_diff(p1, p2)
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
            return knn_imputer, Y

        knn5, Y10 = validate(5, 10)
        knn50, Y40 = validate(50, 40)

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

        trace = trace_execution_piece_by_piece(TorchKNNImputer(knn5), inputs, verbose=0)

        shape_functions = {
            "NanEuclidean": {
                0: lambda *args, **kwargs: torch.empty(
                    (args[0].shape[0], args[1].shape[0]), dtype=args[0].dtype
                )
            },
            "CalcImpute": {
                0: lambda *args, **kwargs: torch.empty(
                    (args[0].shape[0],), dtype=args[0].dtype
                )
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
                0: lambda *args, **kwargs: torch.empty(
                    (args[0].shape[0],), dtype=args[1].dtype
                ),
            },
        }

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

        for t in trace:
            if t.exporter_status.exported is None:
                print(f"[run_decompositions] {t.dot_name} - skipped")
                continue
            print(f"[run_decompositions] {t.dot_name}")
            t.exporter_status.exported = t.exporter_status.exported.run_decompositions({})

        T = str

        def onnx_topk_indices(
            g: GraphBuilder,
            sts: Optional[Dict[str, Any]],
            outputs: List[str],
            x: T,
            k: T,
            name: str = "topk",
        ):
            assert len(outputs) == 1, f"Only one output is expected but outputs={outputs}"
            unique_name = g.unique_name("unused_topk_values")
            g.op.TopK(
                x, k, name=name, outputs=[unique_name, *outputs], largest=False, sorted=True
            )
            return outputs[0]

        dispatcher = Dispatcher(
            {
                (
                    "diag_lib::C_TorchKNNImputer_columns_0___calc_impute__donors_idx__topk"
                ): onnx_topk_indices,
                (
                    "diag_lib::C_TorchKNNImputer_columns_1___calc_impute__donors_idx__topk"
                ): onnx_topk_indices,
            }
        )

        onx = trace.to_onnx_local(
            verbose=1,
            dispatcher=dispatcher,
            check_conversion_cls=dict(cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5),
        )

        onnx.save(onx, self.get_dump_file("test_e2e_knn_imputer.onnx"))

        def validate_onnx(size, sizey, onx, verbose: int = 1):
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

            model_inputs = (
                torch.from_numpy(knn_imputer._mask_fit_X),
                torch.from_numpy(knn_imputer._valid_mask),
                torch.from_numpy(knn_imputer._fit_X),
                Y,
            )
            p2 = model.transform(*model_inputs)
            d = max_diff(p1, p2)
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"

            input_names = [i.name for i in onx.graph.input]
            feeds = dict(zip(input_names, [t.numpy() for t in model_inputs]))

            sess = ExtendedReferenceEvaluator(onx, verbose=0)
            got = sess.run(None, feeds)
            d = max_diff(p1, got[0])
            assert (
                d["abs"] < 1e-5
            ), f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}"

            model_inputs = (
                torch.from_numpy(knn_imputer._mask_fit_X),
                torch.from_numpy(knn_imputer._valid_mask),
                torch.from_numpy(knn_imputer._fit_X),
                Y[1:2],
            )
            p1 = knn_imputer.transform(Y[1:2])
            p2 = model.transform(*model_inputs)
            d = max_diff(p1, p2)
            assert d["abs"] < 1e-5, f"Discrepancies for size={size} and sizey={sizey}, d={d}"
            feeds = dict(zip(input_names, [t.numpy() for t in model_inputs]))
            got = sess.run(None, feeds)
            d = max_diff(p1, got[0])
            assert (
                d["abs"] < 1e-5
            ), f"ONNX Discrepancies for size={size} and sizey={sizey}, d={d}"

        validate_onnx(5, 10, onx)
        validate_onnx(50, 40, onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
