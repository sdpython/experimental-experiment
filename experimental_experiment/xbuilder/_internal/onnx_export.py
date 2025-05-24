import textwrap
from typing import Optional
import numpy
import onnx
from onnx.helper import printable_graph, make_node, np_dtype_to_tensor_dtype
from onnx import numpy_helper, ModelProto
from onnx.helper import tensor_dtype_to_np_dtype
from .onnx_export_templates import get_numpy_template
from .numpy_helper import make_numpy_code


_keywords = {
    "False",
    "await",
    "else",
    "import",
    "pass",
    "None",
    "break",
    "except",
    "in",
    "raise",
    "True",
    "class",
    "finally",
    "is",
    "return",
    "and",
    "continue",
    "for",
    "lambda",
    "try",
    "as",
    "def",
    "from",
    "nonlocal",
    "while",
    "assert",
    "del",
    "global",
    "not",
    "with",
    "async",
    "elif",
    "if",
    "or",
    "yield",
}


def get_attribute_value(at: onnx.AttributeProto):
    if at.type == onnx.AttributeProto.INT:
        return at.i
    if at.type == onnx.AttributeProto.FLOAT:
        return at.f
    if at.type == onnx.AttributeProto.INTS:
        return list(at.ints)
    raise NotImplementedError(f"Unable to get the value of {at!r}")


def _rename_var(var, empty="None"):
    if var in _keywords:
        return "r_" + var
    if var == "":
        return empty
    return var


def select_attribute(ens, att, sort=False, unique=False, skip=None):
    """
    Returns the list of the same attribute.
    `[el.att for el in ens]`.

    :param ens: list
    :param att: attribute name
    :param sort: sort the array
    :param unique: returns the unique values
    :param skip: to skip some names
    :return: something like `[el.att for el in ens]`
    """
    if len(ens) == 0:
        return []
    if isinstance(ens[0], dict):
        atts = [el[att] for el in ens]
    else:
        atts = [getattr(el, att) for el in ens]
    if unique:
        atts = list(set(atts))
    if sort:
        atts.sort()
    if skip is None:
        return atts
    return [a for a in atts if a not in skip]


def _nodes(
    graph,
    rename_name,
    used,
    output_names,
    use_onnx_tensor,
    templates,
    verbose,
    opset,
    rename,
    name,
    subgraphs,
    unique_operators,
    opsets=None,
):
    if opsets is None:
        raise ValueError("opsets cannot be None.")
    nodes = []
    for node in list(graph.node):
        for i_raw_name in node.input:
            if len(i_raw_name) == 0:
                i = "None"
            else:
                i = rename_name(i_raw_name, out=False)
                if i not in used:
                    used[i] = []
                used[i].append(node)
        attributes = []
        for at in node.attribute:
            value = get_attribute_value(at)
            if node.op_type in {"Scan", "Loop"} and at.name == "body":
                if "{{ inputs[0][0] }}" in str(templates):
                    attributes.append((at.name, at.g))
                    continue
                fname = "_create_" + node.op_type + "_" + node.name + "_body"
                body = export_template(
                    value,
                    templates,
                    opset=opset,
                    verbose=verbose,
                    name=name,
                    rename=rename,
                    use_onnx_tensor=use_onnx_tensor,
                    function_name=fname,
                    opsets=opsets,
                )
                subgraphs.append((body, node.op_type + "_" + node.name + "_body"))
                attributes.append((at.name, fname + "()"))
                continue
            if node.op_type == "If" and at.name in {"then_branch", "else_branch"}:
                if "{{ inputs[0][0] }}" in str(templates):
                    attributes.append((at.name, at.g))
                    continue
                fname = "_create_if_" + node.name + "_" + at.name
                body = export_template(
                    value,
                    templates,
                    opset=opset,
                    verbose=verbose,
                    name=name,
                    rename=rename,
                    use_onnx_tensor=use_onnx_tensor,
                    function_name=fname,
                    opsets=opsets,
                )
                subgraphs.append((body, "if_" + node.name + "_" + at.name))
                attributes.append((at.name, fname + "()"))
                continue
            if use_onnx_tensor:
                if node.op_type == "Cast" and at.name == "to":
                    attributes.append((at.name, str(int(value))))
                    continue
            if isinstance(value, str):
                attributes.append((at.name, f"{value!r}"))
            else:
                if isinstance(value, numpy.ndarray):
                    if use_onnx_tensor and at.name == "value":
                        onnx_dtype = str(np_dtype_to_tensor_dtype(value.dtype))
                        value = (
                            'make_tensor("value", %s, dims=%r, vals=%r)'
                            ""
                            % (  # noqa: ISC001
                                onnx_dtype,
                                list(value.shape),
                                value.tolist(),
                            )
                        )
                        attributes.append((at.name, value))
                    else:
                        attributes.append((at.name, repr(value.tolist())))
                else:
                    attributes.append((at.name, repr(value)))

        attributes_str = ", ".join(f"{k}={v}" for k, v in attributes)
        d = dict(
            name=node.name,
            op_type=node.op_type,
            domain=node.domain,
            onnx_node=node,
            inputs=[rename_name(n, out=False) for n in node.input if len(n) > 0],
            outputs=[rename_name(n, out=True) for n in node.output],
            output_names=[rename_name(n, out=True) for n in node.output if n in output_names],
            attributes=attributes,
            attributes_str=attributes_str,
        )
        nodes.append(d)
    return nodes


def _xop_make_node_name(domain, name):
    from ..npy.xop import _domain_to_class_name

    class_name = "Onnx" + _domain_to_class_name(domain) + name
    return class_name


def _python_make_node_name(domain, version, name, node=False):
    if node:
        if version is None:
            version = 1
        if not isinstance(version, int):
            raise TypeError(
                "version must be an integer not %r for domain=%r and name=%r."
                % (version, domain, name)
            )
        if domain == "":
            return "opset%d.%s" % (version, name)
        return "%s%d.%s" % (domain.replace(".", "_"), version, name)
    return name


def _python_make_node_graph(graph, opsets, indent=0, output_names=None):
    """
    Translates a GraphProto into python.
    """
    code = []
    sindent = "    " * indent
    for init in graph.initializer:
        node = make_node("Constant", [], [_rename_var(init.name)], value=init)
        code.append(_python_make_node(node, opsets, indent=indent))
    assert (
        len(graph.sparse_initializer) == 0
    ), "Unable to convert sparse_initilizer into python."
    for node in list(graph.node):
        code.append(_python_make_node(node, opsets, indent=indent))
    if output_names is not None:
        for fr, to in zip(graph.output, output_names):
            code.append(f"{sindent}{_rename_var(to)} = {_rename_var(fr.name)}")
    return "\n".join(code)


def _python_make_node_make_attribute_str(node):
    attributes = []
    for at in node.attribute:
        value = get_attribute_value(at)
        if isinstance(value, str):
            attributes.append((at.name, f"{value.decode('utf-8')!r}"))
            continue
        if isinstance(value, numpy.ndarray):
            if at.name == "value":
                onnx_dtype = str(np_dtype_to_tensor_dtype(value.dtype))
                value = 'make_tensor("value", %s, dims=%r, vals=%r)' "" % (  # noqa: ISC001
                    onnx_dtype,
                    list(value.shape),
                    value.ravel().tolist(),
                )
                attributes.append((at.name, value))
                continue
            attributes.append((at.name, repr(value.tolist())))
            continue
        attributes.append((at.name, repr(value)))

    return ", ".join(f"{k}={v}" for k, v in attributes)


def _python_make_node_if(node, opsets, indent=0):
    """
    Translates a node If into python.
    """
    sindent = "    " * indent
    code = [f"{sindent}if {node.input[0]}:"]
    if len(node.attribute) != 2:
        raise RuntimeError(
            f"Node {node.op_type!r} expected two attributes not {len(node.attribute)}."
        )
    atts = node.attribute
    if atts[0].name == "else_branch":
        else_branch, then_branch = atts[0].g, atts[1].g
    else:
        else_branch, then_branch = atts[1].g, atts[0].g
    code.append(
        _python_make_node_graph(
            then_branch, opsets, indent=indent + 1, output_names=node.output
        )
    )
    code.append(f"{sindent}else:")
    code.append(
        _python_make_node_graph(
            else_branch, opsets, indent=indent + 1, output_names=node.output
        )
    )
    return "\n".join(code)


def _python_make_node_loop(node, opsets, indent=0):
    """
    Translates a node Loop into python.
    """
    raise NotImplementedError()


def _python_make_node_scan(node, opsets, indent=0):
    """
    Translates a node Scan into python.
    """
    raise NotImplementedError()


def _python_make_node(onnx_node, opsets, indent=0):
    if isinstance(onnx_node, dict):
        node = onnx_node["onnx_node"]
    else:
        node = onnx_node
    version = opsets[node.domain]
    if node.op_type in {"If", "Loop", "Scan"}:
        # If, Loop, Scan
        if node.op_type == "If":
            return _python_make_node_if(node, opsets, indent=indent)
        if node.op_type == "Loop":
            return _python_make_node_loop(node, opsets, indent=indent)
        if node.op_type == "Scan":
            return _python_make_node_scan(node, opsets, indent=indent)
        raise RuntimeError(f"Unable to export node type {node.op_type!r} into python.")

    if any(
        map((hasattr(att, "g") and att.g and att.g.ByteSize() > 0) for att in node.attribute)
    ):
        raise RuntimeError(f"Unable to export node type {node.op_type!r} into python.")
    ops = {
        "Add": "+",
        "Sub": "-",
        "Mul": "*",
        "MatMul": "@",
        "Div": "/",
        "Pow": "**",
        "And": "&",
        "Or": "|",
        "Greater": ">",
        "Equal": "==",
        "Lesser": "<",
        "GreaterOrEqual": ">=",
        "LessOrEqual": "<=",
    }
    sindent = "    " * indent
    if node.op_type in ops:
        return "%s%s = %s" % (
            sindent,
            _rename_var(node.output[0], empty="_"),
            (" %s " % ops[node.op_type]).join(map(_rename_var, node.input)),
        )
    name = _python_make_node_name(node.domain, version, node.op_type, node=True)
    attributes_str = _python_make_node_make_attribute_str(node)
    if len(node.input) > 0 and len(attributes_str) > 0:
        attributes_str = ", " + attributes_str
    output = ", ".join(_rename_var(s, empty="_") for s in node.output)
    text = [
        sindent,
        output,
        " = ",
        name,
        "(",
        ", ".join(map(_rename_var, node.input)),
        attributes_str,
        ")",
    ]
    return "".join(text)


def export_template(
    model_onnx: ModelProto,
    templates,
    opset: Optional[int] = None,
    verbose=True,
    name=None,
    rename=False,
    use_onnx_tensor=False,
    function_name="create_model",
    clean_code=True,
    opsets=None,
):
    """
    Exports an ONNX model to the onnx syntax.

    :param model_onnx: string or ONNX graph
    :param templates: exporting templates
    :param opset: opset to export to
        (None to select the one from the graph)
    :param opsets: nodes uses these opsets
    :param verbose: insert prints
    :param name: to overwrite onnx name
    :param rename: rename the names to get shorter names
    :param use_onnx_tensor: when an attribute is an array
        and its name is `'value'`, it converts that array into an
        ONNX tensor to avoid type mismatch, (operator *ConstantOfShape*, ...)
    :param function_name: main function name in the code
    :param clean_code: clean the code
    :return: python code
    """
    assert opset is None or isinstance(opset, int), f"Wrong type for opset={opset!r}"

    def number2name(n):
        n += 1
        seq = []
        while n >= 1:
            r = n % 26
            seq.append(r)
            n = (n - r) // 26
        return "".join(chr(65 + i) for i in reversed(seq))

    def rename_name(name, out):
        if len(name) == 0:
            if out:
                return "__"
            return "_"
        if name in dict_names:
            return dict_names[name]
        if rename:
            i = 0
            new_name = number2name(i)
            while new_name in dict_names:
                i += 1
                new_name = number2name(i)
            if len(new_name) == 0:
                raise ValueError("Unable to rename name=%r i=%d." % (name, i))
            dict_names[name] = new_name
            dict_names[new_name] = new_name
            return new_name
        return name

    # unique_function_domain_version
    unique_function_domain_version = set()
    if hasattr(model_onnx, "functions"):
        for f in model_onnx.functions:
            unique_function_domain_version.add((f.domain, 1))
    unique_function_domain_version = list(sorted(unique_function_domain_version))

    # containers
    context = {
        "main_model": model_onnx,
        "printable_graph": printable_graph,
        "xop_make_node_name": _xop_make_node_name,
        "python_make_node": _python_make_node,
        "python_make_node_name": _python_make_node_name,
        "unique_function_domain_version": unique_function_domain_version,
        "rename_var": _rename_var,
    }
    used = {}

    # opset
    if hasattr(model_onnx, "opset_import"):
        if opsets is None:
            opsets = {}
        else:
            opsets = opsets.copy()
        for oimp in model_onnx.opset_import:
            if oimp.domain == "" and opset is None:
                opsets[oimp.domain] = oimp.version
                opset = oimp.version
            else:
                opsets[oimp.domain] = opset
        context["opsets"] = opsets
        assert isinstance(opset, int), f"Wrong type for opset={opset!r}"
        context["target_opset"] = opset
    else:
        context["opsets"] = opsets
    if opsets is None:
        raise ValueError("opsets cannot be None.")

    if hasattr(model_onnx, "graph"):
        graph = model_onnx.graph
    else:
        graph = model_onnx
    dict_names = {}
    if rename:
        for o in graph.input:
            dict_names[o.name] = o.name
        for o in graph.output:
            dict_names[o.name] = o.name

    # inits
    unique_operators = set()
    initializers = []
    for init in graph.initializer:
        init_name = rename_name(init.name, out=False)
        value = numpy_helper.to_array(init)
        initializers.append((init_name, value))
    context["initializers"] = initializers
    context["initializers_dict"] = dict(initializers)

    # functions
    functions = []
    fct_dict = {}
    if hasattr(model_onnx, "functions") and model_onnx.functions:
        for fct in model_onnx.functions:
            used = {}
            opsets_fct = {}
            for oimp in fct.opset_import:
                if oimp.domain == "" and opset is None:
                    opsets_fct[oimp.domain] = oimp.version
                else:
                    opsets_fct[oimp.domain] = opset
            functions.append(
                (
                    fct.domain,
                    fct.name,
                    {
                        "proto": fct,
                        "opsets": opsets_fct,
                        "nodes": _nodes(
                            fct,
                            rename_name,
                            used,
                            fct.output,
                            use_onnx_tensor,
                            templates,
                            verbose,
                            opset,
                            rename,
                            fct.name,
                            [],
                            unique_operators,
                            opsets=opsets,
                        ),
                    },
                )
            )
            if fct.name in fct_dict:
                fct_dict[fct.name].append(fct)
            else:
                fct_dict[fct.name] = [fct]
    context["functions"] = functions
    context["functions_dict"] = fct_dict

    # inputs
    inputs = []
    for inp in graph.input:
        elem_type = inp.type.tensor_type.elem_type
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        inputs.append((inp.name, elem_type, shape))
    context["inputs"] = inputs

    # outputs
    outputs = []
    for inp in graph.output:
        elem_type = inp.type.tensor_type.elem_type
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        outputs.append((inp.name, elem_type, shape))
    context["outputs"] = outputs

    # node
    output_names = set(o.name for o in graph.output)
    subgraphs = []
    context["graph"] = graph
    context["nodes"] = _nodes(
        graph,
        rename_name,
        used,
        output_names,
        use_onnx_tensor,
        templates,
        verbose,
        opset,
        rename,
        name,
        subgraphs,
        unique_operators,
        opsets=opsets,
    )

    # graph
    context["name"] = (name or graph.name).replace("(", "_").replace(")", "")
    context["function_name"] = function_name
    context["indent"] = textwrap.indent
    if hasattr(model_onnx, "graph"):
        context["ir_version"] = model_onnx.ir_version
        context["producer_name"] = model_onnx.producer_name
        context["domain"] = model_onnx.domain
        context["model_version"] = model_onnx.model_version
        context["doc_string"] = model_onnx.doc_string
        context["metadata"] = {p.key: p.value for p in model_onnx.metadata_props}
    else:
        # subgraph
        context["ir_version"] = None
        context["producer_name"] = None
        context["domain"] = None
        context["model_version"] = None
        context["doc_string"] = ""
        context["metadata"] = {}

    # common context
    context["unique_operators"] = [
        dict(domain=o[0], name=o[1], classname=o[2]) for o in sorted(unique_operators)
    ]
    context["skip_inits"] = {}
    context["subgraphs"] = subgraphs

    mark_inits = {}

    # First rendering to detect any unused or replaced initializer.
    from jinja2 import Template  # delayed import

    template = Template(templates)
    final = template.render(
        enumerate=enumerate,
        sorted=sorted,
        len=len,
        map=map,
        select_attribute=select_attribute,
        repr=repr,
        tensor_dtype_to_np_dtype=tensor_dtype_to_np_dtype,
        make_numpy_code=lambda *args, **kwargs: make_numpy_code(
            *args, context=context, used=used, mark_inits=mark_inits, **kwargs
        ),
        verbose=verbose,
        **context,
    )

    skip_inits = set()
    for k, v in mark_inits.items():
        if len(v) == len(used[k]):
            # One initializers was removed.
            skip_inits.add(k)

    if len(skip_inits) > 0:
        # Second rendering if needed when an initializer was replaced
        # or removed.
        context["skip_inits"] = skip_inits
        # Again with skip_inits.
        final = template.render(
            enumerate=enumerate,
            sorted=sorted,
            len=len,
            make_numpy_code=lambda *args, **kwargs: make_numpy_code(
                *args, context=context, used=used, mark_inits=mark_inits, **kwargs
            ),
            verbose=verbose,
            **context,
        )

    final += "\n"
    if not verbose:
        rows = final.split("\n")
        final = "\n".join(_ for _ in rows if not _.endswith("#  verbose"))
    return final


def export2numpy(
    model_onnx: ModelProto,
    opset: Optional[int] = None,
    verbose: int = 0,
    name: str = "onnx_exported_to_numpy",
    rename: bool = False,
):
    """
    Exports an ONNX model to the :epkg:`numpy` syntax.
    The exports does not work with all operators.

    :param model_onnx: string or ONNX graph
    :param opset: opset to export to
        (None to select the one from the graph)
    :param verbose: inserts prints
    :param name: to overwrite onnx name
    :param rename: rename the names to get shorter names
    :return: python code

    .. runpython::
        :showcode:
        :process:

        import numpy
        from sklearn.cluster import KMeans
        from skl2onnx import to_onnx
        from experimental_experiment.xbuilder._internal.onnx_export import export2numpy

        X = numpy.arange(20).reshape(10, 2).astype(numpy.float32)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        onx = to_onnx(tr, X, target_opset=14)
        code = export2numpy(onx)

        print(code)
    """
    if isinstance(model_onnx, str):
        model_onnx = onnx.load(model_onnx)

    if not isinstance(model_onnx, ModelProto):
        raise TypeError(f"The function expects a ModelProto not {type(model_onnx)!r}.")
    code = export_template(
        model_onnx,
        templates=get_numpy_template(),
        opset=opset,
        verbose=verbose,
        name=name,
        rename=rename,
    )
    for i in range(-6, 6):
        code = code.replace("axis=tuple([%d])" % i, "axis=%d" % i)
        code = code.replace("tuple([%d])" % i, "(%d, )" % i)
    return code
