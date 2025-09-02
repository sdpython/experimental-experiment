import copy
import unittest
from typing import Any, Dict, List, Tuple
import numpy as np
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
)


class TestIssuesPytorch2024Export(ExtTestCase):

    @skipif_ci_windows("not working")
    @requires_torch("2.8")
    def test_export_index_put_none_decompositions(self):
        # see issue https://github.com/pytorch/pytorch/issues/141336
        import torch

        class UpdateModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.params = torch.zeros((4, 4, 10))

            def forward(self, update, index1, index2):
                copy = self.params.clone()
                copy[index1, torch.tensor([1, 2], dtype=torch.int64), index2] = update
                return copy

        model = UpdateModel()

        update = (torch.arange(2) + 10).reshape((2,)).to(torch.float32)
        index1 = torch.from_numpy(np.array([1, 2])).to(torch.int64)
        index2 = torch.from_numpy(np.array([7, 8])).to(torch.int64)
        model(update, index1, index2)

        ep = torch.export.export(model, (update, index1, index2))
        # print(ep.graph)
        ep.run_decompositions()  # Fails here

    @skipif_ci_windows("not working")
    @requires_torch("2.10")
    def test_export_mistral_nousers(self):
        import onnx
        import torch
        import transformers
        import onnxruntime as ort
        from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
        from onnx_diagnostic.torch_export_patches import torch_export_patches

        def assert_close(actual, desired):
            if isinstance(desired, torch.Tensor):
                torch.testing.assert_close(actual, desired)
            elif isinstance(desired, tuple):
                assert isinstance(actual, tuple)
                assert len(actual) == len(desired)
                for a, d in zip(actual, desired):
                    torch.testing.assert_close(a, d)
            else:
                raise NotImplementedError(f"Not implemented for class {type(desired)}")

        def flatten(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, tuple):
                res = []
                for o in obj:
                    if isinstance(o, torch.Tensor):
                        res.append(o)
                    else:
                        res.extend(flatten(o))
                return tuple(res)
            if isinstance(obj, list):
                res = []
                for o in obj:
                    if isinstance(o, torch.Tensor):
                        res.append(o)
                    else:
                        res.extend(flatten(o))
                return res
            if isinstance(obj, transformers.cache_utils.DynamicCache):
                obj = CacheKeyValue(obj)
                return [*obj.key_cache, *obj.value_cache]
            raise NotImplementedError(f"Not implemented for class {type(obj)}")

        config = transformers.MistralConfig(
            hidden_size=32,
            num_hidden_layers=2,
            vocab_size=99,
            intermediate_size=16,
            max_position_embeddings=512,
            num_attention_heads=2,
            num_key_value_heads=2,
            sliding_window=4096,
        )
        config._attn_implementation = "eager"
        model = transformers.MistralModel(config)
        shape = (1, 30)
        input_ids = torch.randint(0, 99, shape).to(torch.int64)
        attention_mask = torch.ones(shape)
        expected = model(input_ids, attention_mask)
        with torch_export_patches(patch_transformers=True, patch_torch=True):
            ep = torch.export.export(model, copy.deepcopy((input_ids, attention_mask)))

            # assert "[num_users=0]" not in str(ep.graph), f"One output is unused:\n{ep.graph}"
            mod = ep.module()
            # got = mod(input_ids, attention_mask)
            # assert_close(got.to_tuple(), expected.to_tuple())

            expected2 = model(*copy.deepcopy((input_ids, attention_mask * 0)))
            got2 = mod(*copy.deepcopy((input_ids, attention_mask * 0)))
            self.assertEqualAny(got2.to_tuple(), expected2.to_tuple())
            # assert_close(expected2.to_tuple(), expected.to_tuple())

            onx = to_onnx(
                model,
                copy.deepcopy((input_ids, attention_mask)),
                export_options=ExportOptions(
                    aten_as_function=True, decomposition_table="default"
                ),
                optimize=False,
            )

        filename = self.get_dump_file("test_mistral_nousers_aten.onnx")
        onnx.save(onx, filename)
        sess = ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
        got = sess.run(
            None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
        )
        assert_close(tuple(torch.from_numpy(t) for t in got), flatten(expected.to_tuple()))

        got = sess.run(
            None,
            {"input_ids": input_ids.numpy(), "attention_mask": (attention_mask * 0).numpy()},
        )
        assert_close(tuple(torch.from_numpy(t) for t in got), flatten(expected2.to_tuple()))

        # ep = torch.onnx.export(mod, (input_ids, attention_mask), dynamo=True)
        # ep.optimize()
        # onnx.save(ep.model_proto, "test_mistral_nousers.onnx")
        # sess = ort.InferenceSession(
        #     "test_mistral_nousers.onnx", providers=["CPUExecutionProvider"]
        # )
        # got = sess.run(
        #     None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
        # )
        # assert_close(tuple(torch.from_numpy(t) for t in got), flatten(expected.to_tuple()))

    def test_export_inplace_raise(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                y = xc[:, :2] * 2
                xc[:, :2] = y
                return x

        model = Model()
        x = torch.ones((4, 4))
        ep = torch.export.export(model, (x,), strict=False)
        # this test should fail but it does not because torch.ops.aten.copy_.default
        # is executed inplace.
        torch.testing.assert_close(model(x), ep.module()(x))
        # print(ep.graph)

        class MyProxy(torch.fx.proxy.Proxy):
            def __setitem__(self, *args, **kwargs):
                raise AssertionError(
                    f"This should fail with args={args!r}, kwargs={kwargs}, "
                    f"self.node={self.node}, node.meta={self.node.meta}"
                )

        class MyTracer(torch.fx.Tracer):
            def proxy(self, node: torch.fx.Node) -> torch.fx.Proxy:
                return MyProxy(node, self)

        # torch.fx.proxy.Proxy.__setitem__ = setitem
        self.assertRaise(lambda: MyTracer().trace(model), AssertionError, "This should fail")
        # print(graph)
        """
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default]
                (args = (%x,), kwargs = {})
            %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_2 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_1, 1, 0, 2), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor]
                (args = (%slice_2, 2), kwargs = {})
            %slice_3 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_4 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_3, 1, 0, 2), kwargs = {})
            %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default]
                (args = (%slice_4, %mul), kwargs = {})
            return (clone,)  <---- This is wrong.

        This is what is expected:

        graph():
            %x : [num_users=1] = placeholder[target=x]
            %clone : [num_users=4] = call_function[target=torch.ops.aten.clone.default]
                (args = (%x,), kwargs = {})
            %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_2 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_1, 1, 0, 2), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor]
                (args = (%slice_2, 2), kwargs = {})
            %slice_3 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_4 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%slice_3, 1, 0, 2), kwargs = {})
            %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default]
                (args = (%slice_4, %mul), kwargs = {})
            %slice_5 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
            %slice_scatter : [num_users=1] = call_function
                [target=torch.ops.aten.slice_scatter.default]
                (args = (%slice_5, %copy, 1, 0, 2), kwargs = {})
            %slice_scatter_1 : [num_users=1] = call_function
                [target=torch.ops.aten.slice_scatter.default]
                (args = (%clone, %slice_scatter, 0, 0, 9223372036854775807), kwargs = {})
            return (slice_scatter_1,)
        """

    def test_export_list(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, yz):
                return x + yz[0] + yz[1]

        model = Model()
        x = torch.ones((4, 4))
        x2 = x * x
        # torch.export.export(model, (x,[x2, x2])) fails because
        # the export detect a duplicated input and reuse whatever is possible.
        ep = torch.export.export(model, (x, [x2, x2 * 2]))
        print(ep.graph)
        # this test should fail but it does not because torch.ops.aten.copy_.default
        # is executed inplace.
        torch.testing.assert_close(model(x, [x * 2, x * 3]), ep.module()(x, [x * 2, x * 3]))
        ep = ep.run_decompositions()
        print(ep.graph)
        self.assertIn("yz_0", str(ep.graph))
        print("-----")
        print(torch.fx.Tracer().trace(model))

    def test_export_inplace_add_(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([2])
                x.add_(a)
                return a

        model = Model()
        x = torch.ones((4, 4))
        ep = torch.export.export(model, (x,))
        ep = ep.run_decompositions({})
        self.assertNotIn("add_", str(ep.graph))

    def test_export_inplace_setitem(self):
        import operator
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                xc = x.clone()
                y = xc[:, :2] * 2
                xc[:, :2] = y
                return xc + 2

        model = Model()
        x = torch.ones((4, 4))
        ep = torch.export.export(model, (x,))
        print(ep.graph)
        ep = ep.run_decompositions({})
        print(ep.graph)
        self.assertIn("slice_scatter", str(ep.graph))

        # this test should fail but it does not because torch.ops.aten.copy_.default
        # is executed inplace.
        torch.testing.assert_close(model(x), ep.module()(x))

        class MyProxy(torch.fx.proxy.Proxy):
            def __setitem__(self, *args, **kwargs):
                assert not kwargs, f"Unexpected not empty kwargs={kwargs!r}"
                assert len(args) == 2, f"Unexpected number of args={len(args)}: {args}"
                indices, values = args
                node = self.tracer.create_node(
                    "call_function", operator.setitem, args=(indices, values.node), kwargs={}
                )
                # node_to_replace = self.node
                return self.tracer.proxy(node)

        class MyTracer(torch.fx.Tracer):
            def proxy(self, node: torch.fx.Node) -> torch.fx.Proxy:
                return MyProxy(node, self)

        graph = MyTracer(autowrap_functions=(operator.setitem,)).trace(
            model,
        )
        self.assertIn("operator.setitem", str(graph))

    def test_dynamic_cache(self):
        import torch

        class MyCache:
            def __init__(self, key_cache=None, value_cache=None):
                self.key_cache = key_cache
                self.value_cache = value_cache

        def flatten_my_cache(cache):
            flat = [
                (k, getattr(cache, k))
                for k in ["key_cache", "value_cache"]
                if hasattr(cache, k)
            ]
            return [f[1] for f in flat], [f[0] for f in flat]

        def unflatten_my_cache(
            values: List[Any],
            context: "_torch_pytree.Context",  # noqa: F821
            output_type=None,
        ) -> MyCache:
            cache = MyCache()
            values = dict(zip(context, values))
            for k, v in values.items():
                setattr(cache, k, v)
            return cache

        def flatten_with_keys_my_cache(d: Dict[Any, Any]) -> Tuple:
            values, context = flatten_my_cache(d)
            return [
                (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
            ], context

        torch.utils._pytree.register_pytree_node(
            MyCache,
            flatten_my_cache,
            unflatten_my_cache,
            serialized_type_name=f"{MyCache.__module__}.{MyCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_my_cache,
        )

        torch.fx._pytree.register_pytree_flatten_spec(
            MyCache, lambda x, _: [x.key_cache, x.value_cache]
        )

        class Model(torch.nn.Module):
            def forward(self, x, cache: MyCache):
                kcat = torch.cat(cache.key_cache, axis=0)
                vcat = torch.cat(cache.value_cache, axis=0)
                s1 = kcat.sum(axis=1)
                s2 = vcat.sum(axis=1)
                return x @ (s1 + s2)

        cache = MyCache(
            [torch.ones([4, 4]), torch.ones([4, 4]) * 2],
            [-torch.ones([4, 4]), -torch.ones([4, 4]) * 2],
        )
        x = torch.ones((2, 8))
        model = Model()
        expected = model(x, cache)

        # static shape
        ep = torch.export.export(model, (x, cache))
        ep = ep.run_decompositions()
        mod = ep.module()
        got = mod(x, cache)
        self.assertEqualArray(expected, got)

        # dynamic shape 1
        ep = torch.export.export(
            model,
            (x, cache),
            dynamic_shapes=({0: torch.export.Dim("batch")}, [[{}, {}], [{}, {}]]),
        )
        ep = ep.run_decompositions()
        mod = ep.module()
        got = mod(x, cache)
        self.assertEqualArray(expected, got)

        # dynamic shape 2
        ep = torch.export.export(
            model,
            (x, cache),
            dynamic_shapes={
                "x": {0: torch.export.Dim("batch")},
                "cache": [[{}, {}], [{}, {}]],
            },
        )
        ep = ep.run_decompositions()
        mod = ep.module()
        got = mod(x, cache)
        self.assertEqualArray(expected, got)

        torch.utils._pytree.SUPPORTED_NODES.pop(MyCache)
        torch.fx._pytree.SUPPORTED_NODES.pop(MyCache)
        torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH.pop(MyCache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
