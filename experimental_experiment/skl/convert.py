import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from onnx import ModelProto, save_model
from onnx.model_container import ModelContainer
import sklearn
from ..xbuilder import GraphBuilder, FunctionOptions, OptimizationOptions


def to_onnx(
    model: sklearn.base.BaseEstimator,
    args: Optional[Sequence["torch.Tensor"]] = None,  # noqa: F821
    target_opset: Optional[Union[int, Dict[str, int]]] = None,
    as_function: bool = False,
    options: Optional[OptimizationOptions] = None,
    optimize: bool = True,
    filename: Optional[str] = None,
    inline: bool = False,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[List[str]] = None,
    large_model: bool = False,
    verbose: int = 0,
    return_builder: bool = False,
    raise_list: Optional[Set[str]] = None,
    external_threshold: int = 1024,
    return_optimize_report: bool = False,
    function_options: Optional[FunctionOptions] = None,
) -> Union[
    Union[ModelProto, ModelContainer],
    Tuple[Union[ModelProto, ModelContainer], GraphBuilder],
]:
    """
    Exports a :epkg:`scikit-learn` model into ONNX.

    :param model: estimator
    :param args: input arguments
    :param kwargs: keyword attributes
    :param input_names: input names
    :param target_opset: targeted opset or targeted opsets as a dictionary
    :param as_function: export as a ModelProto or a FunctionProto
    :param options: optimization options
    :param verbose: verbosity level
    :param return_builder: returns the builder as well
    :param raise_list: the builder stops any time a name falls into that list,
        this is a debbuging tool
    :param optimize: optimize the model before exporting into onnx
    :param large_model: if True returns a :class:`onnx.model_container.ModelContainer`,
        it lets the user to decide later if the weights should be part of the model
        or saved as external weights
    :param external_threshold: if large_model is True, every tensor above this limit
        is stored as external
    :param return_optimize_report: returns statistics on the optimization as well
    :param filename: if specified, stores the model into that file
    :param inline: inline the model before converting to onnx, this is done before
            any optimization takes place
    :param export_options: to apply differents options before to get the exported program
    :param function_options: to specify what to do with the initializers in local functions,
        add them as constants or inputs
    :param output_names: to rename the output names
    :return: onnx model
    """
    assert isinstance(
        model, sklearn.base.BaseEstimator
    ), f"Unexpected model type {type(model)}"
    import skl2onnx

    if output_names is None:
        if hasattr(model, "get_feature_names_out"):
            output_names = model.get_feature_names_out()

    if args is None:
        if hasattr(model, "n_features_in_"):
            n = model.n_features_in_
        else:
            raise NotImplementedError(
                f"Unable to guess the number of input features for model type {type(model)}"
            )
        args = np.random.randn(2, n).astype(np.float32)

    if isinstance(
        model,
        (
            sklearn.pipeline.Pipeline,
            sklearn.pipeline.FeatureUnion,
            sklearn.compose.ColumnTransformer,
            sklearn.compose.TransformedTargetRegressor,
        ),
    ):
        raise NotImplementedError(f"not implemented yet for {type(model)}")

    add_stats = {}
    begin = time.perf_counter()
    if verbose:
        print(f"[skl.to_onnx] convert {model.__class__.__name__}")
    proto = skl2onnx.to_onnx(
        model,
        args[0],
        target_opset=target_opset,
        options={"zipmap": False} if sklearn.base.is_classifier(model) else None,
        verbose=max(verbose - 1, 0),
    )
    t = time.perf_counter()
    add_stats["time_export"] = t - begin
    add_stats[f"time_export_{model.__class__.__name__}"] = t - begin
    begin = t

    if verbose:
        print(f"[skl.to_onnx] builds {model.__class__.__name__}")
    builder = GraphBuilder(
        target_opset_or_existing_proto=proto,
        as_function=as_function,
        optimization_options=options,
        args=args,
        kwargs=None,
        verbose=verbose,
        raise_list=raise_list,
        graph_module=model,
        output_names=output_names,
    )

    if input_names:
        renames = dict(zip(builder.input_names, input_names))
        if verbose:
            print(f"[skl.to_onnx] renames {renames}")
        builder.rename_names(renames)

    t = time.perf_counter()
    add_stats["time_builder"] = t - begin
    add_stats[f"time_builder_{model.__class__.__name__}"] = t - begin
    begin = t

    if verbose:
        print(f"[skl.to_onnx] make_proto for {model.__class__.__name__}")
    onx, stats = builder.to_onnx(
        optimize=optimize,
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=True,
        inline=inline,
        function_options=function_options,
    )
    t = time.perf_counter()
    add_stats["time_builder_to_onnx"] = t - begin
    add_stats[f"time_builder_to_onnx_{model.__class__.__name__}"] = t - begin
    begin = time.perf_counter()

    if verbose:
        print(f"[skl.to_onnx] done {model.__class__.__name__}")

    all_stats = dict(builder=builder.statistics_)
    if stats:
        add_stats["optimization"] = stats
    t = time.perf_counter()
    add_stats["time_export_to_onnx"] = t - begin

    if verbose:
        proto = onx if isinstance(onx, ModelProto) else onx.model_proto
        print(
            f"[to_onnx] to_onnx done in {t - begin}s "
            f"and {len(proto.graph.node)} nodes, "
            f"{len(proto.graph.initializer)} initializers, "
            f"{len(proto.graph.input)} inputs, "
            f"{len(proto.graph.output)} outputs"
        )
        if verbose >= 10:
            print(builder.get_debug_msg())

    if filename:
        if isinstance(onx, ModelProto):
            save_model(onx, filename)
        else:
            onx.save(filename, all_tensors_to_one_file=True)
    all_stats.update(add_stats)
    if return_builder:
        return (onx, builder, all_stats) if return_optimize_report else (onx, builder)
    return (onx, all_stats) if return_optimize_report else onx
