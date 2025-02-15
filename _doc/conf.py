import os
import sys
import packaging.version as pv
from sphinx_runpython.github_link import make_linkcode_resolve
from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
import torch
from experimental_experiment import __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/logo.png"
source_suffix = ".rst"
master_doc = "index"
project = "experimental-experiment"
copyright = "2023-2024"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_static_path = ["_static"]
html_sourcelink_suffix = ""

issues_github_path = "sdpython/experimental-experiment"

# suppress_warnings = [
#     "tutorial.exported_onnx",
#     "tutorial.exported_onnx_dynamic",
#     "tutorial.exported_program",
#     "tutorial.exported_program_dynamic",
# ]

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "experimental-experiment",
    (
        "https://github.com/sdpython/experimental-experiment/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

intersphinx_mapping = {
    "experimental_experiment": (
        "https://sdpython.github.io/doc/experimental-experiment/dev/",
        None,
    ),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "onnx_array_api": ("https://sdpython.github.io/doc/onnx-array-api/dev/", None),
    "onnx_extended": ("https://sdpython.github.io/doc/onnx-extended/dev/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skl2onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "torch": ("https://pytorch.org/docs/main/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "dtype"),
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "Argument"),
    ("py:class", "onnxscript.ir.Tuple"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
    ("py:class", "ModelProto"),
    ("py:class", "Module"),
    ("py:class", "torch.fx.passes.operator_support.OperatorSupport"),
    ("py:class", "torch.fx.proxy.TracerBase"),
    ("py:class", "torch.utils._pytree.Context"),
    ("py:class", "torch.utils._pytree.KeyEntry"),
    ("py:class", "torch.utils._pytree.TreeSpec"),
    ("py:class", "transformers.cache_utils.Cache"),
    ("py:class", "transformers.cache_utils.DynamicCache"),
    ("py:class", "transformers.cache_utils.MambaCache"),
    ("py:func", "torch.export._draft_export.draft_export"),
    ("py:func", "torch._export.tools.report_exportability"),
]

nitpick_ignore_regex = [
    ("py:func", ".*numpy[.].*"),
    ("py:func", ".*scipy[.].*"),
    ("py:func", ".*torch.ops.higher_order.*"),
    ("py:class", ".*onnxruntime[.].*"),
]


sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        os.path.join(os.path.dirname(__file__), "examples"),
        os.path.join(os.path.dirname(__file__), "recipes"),
    ],
    # path where to save gallery generated examples
    "gallery_dirs": [
        "auto_examples",
        "auto_recipes",
    ],
    # no parallelization to avoid conflict with environment variables
    "parallel": 1,
    # sorting
    "within_subsection_order": "ExampleTitleSortKey",
    # errors
    "abort_on_example_error": True,
    # recommendation
    "recommender": {"enable": True, "n_examples": 5, "min_df": 3, "max_df": 0.9},
    # ignore capture for matplotib axes
    "ignore_repr_types": "matplotlib\\.(text|axes)",
    # robubstness
    "reset_modules_order": "both",
    "reset_modules": ("matplotlib", "experimental_experiment.reset_torch_transformers"),
}

if int(os.environ.get("UNITTEST_GOING", "0")):
    sphinx_gallery_conf["ignore_pattern"] = (
        ".*((_oe_)|(dort)|(diff)|(exe)|(llama)|(aot)|(compile)|(export_201)|"
        "(c_phi2)|(oe_custom_ops_inplace)|(oe_scan)|(draft_mode)).*"
    )
    # it fails if not run in standalone mode
    sphinx_gallery_conf["ignore_pattern"] = (
        f"{sphinx_gallery_conf['ignore_pattern'][:-3]}|"
        f"(torch_sklearn_201)|(plot_exporter_exporter_with_dynamic_cache)).*"
    )
elif pv.Version(torch.__version__) < pv.Version("2.8"):
    sphinx_gallery_conf["ignore_pattern"] = (
        ".*((_oe_((modules)|(custom)))|(_executorch_)|(oe_scan)).*"
    )


epkg_dictionary = {
    "aten functions": "https://pytorch.org/cppdocs/api/namespace_at.html#functions",
    "azure pipeline": "https://azure.microsoft.com/en-us/products/devops/pipelines",
    "Custom Backends": "https://pytorch.org/docs/stable/torch.compiler_custom_backends.html",
    "diffusers": "https://github.com/huggingface/diffusers",
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "executorch": "https://pytorch.org/executorch/stable/intro-overview.html",
    "ExecuTorch": "https://pytorch.org/executorch/stable/intro-overview.html",
    "ExecuTorch Runtime Python API Reference": "https://pytorch.org/executorch/stable/runtime-python-api-reference.html",
    "ExecuTorch Tutorial": "https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "FunctionProto": "https://onnx.ai/onnx/api/classes.html#functionproto",
    "graph break": "https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks",
    "GraphModule": "https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule",
    "Linux": "https://www.linux.org/",
    "monai": "https://monai.io/",
    "numpy": "https://numpy.org/",
    "onnx": "https://onnx.ai/onnx/",
    "onnx.helper": "https://onnx.ai/onnx/api/helper.html",
    "ONNX": "https://onnx.ai/",
    "ONNX Operators": "https://onnx.ai/onnx/operators/",
    "onnxrt backend": "https://pytorch.org/docs/stable/onnx_dynamo_onnxruntime_backend.html",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-training": "https://onnxruntime.ai/docs/get-started/training-on-device.html",
    "onnx-array-api": "https://sdpython.github.io/doc/onnx-array-api/dev/",
    "onnx-extended": "https://sdpython.github.io/doc/onnx-extended/dev/",
    "onnx-script": "https://github.com/microsoft/onnxscript",
    "onnxscript": "https://github.com/microsoft/onnxscript",
    "onnxscript Tutorial": "https://onnxscript.ai/tutorial/index.html",
    "Pattern-based Rewrite Using Rules With onnxscript": "https://onnxscript.ai/tutorial/rewriter/rewrite_patterns.html",
    "opsets": "https://onnx.ai/onnx/intro/concepts.html#what-is-an-opset-version",
    "pyinstrument": "https://pyinstrument.readthedocs.io/en/latest/",
    "psutil": "https://psutil.readthedocs.io/en/latest/",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "run_with_ortvaluevector": "https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_inference_collection.py#L339",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sklearn-onnx": "https://onnx.ai/sklearn-onnx/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "Supported Operators and Data Types": "https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md",
    "sympy": "https://www.sympy.org/en/index.html",
    "timm": "https://github.com/huggingface/pytorch-image-models",
    "torch": "https://pytorch.org/docs/stable/torch.html",
    "torchbench": "https://github.com/pytorch/benchmark",
    "torch.compile": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
    "torch.compiler": "https://pytorch.org/docs/stable/torch.compiler.html",
    "torch.export.export": "https://pytorch.org/docs/stable/export.html#torch.export.export",
    "torch.onnx": "https://pytorch.org/docs/stable/onnx.html",
    "transformers": "https://huggingface.co/docs/transformers/en/index",
    "vocos": "https://github.com/gemelo-ai/vocos",
    "Windows": "https://www.microsoft.com/windows",
}
