"""Unit coverage for the V3 strict authoring and graph compiler contract."""

from __future__ import annotations

import pickle
from typing import Annotated, Any, NamedTuple

import pytest

from rayorch.experimental.multigrain_v3 import (
    CompileError,
    Map,
    OpaqueValue,
    Pipeline,
    expand,
    filter,
    reduce,
)
from rayorch.experimental.multigrain_v3.benchmark.mineru import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
)
from rayorch.experimental.multigrain_v3.model.graph import (
    BoolShape,
    ExpandOp,
    MapOp,
    OpaqueShape,
    ParameterKind,
    ReduceOp,
    ReturnKind,
    SerializableCallableRef,
    StructuralListShape,
    infer_run_schemas,
    thaw_config_value,
)
from rayorch.experimental.multigrain_v3.model.semantics import (
    PREIMAGE_DEBUG_LEDGER_LIMIT,
    GraphFingerprint,
    preimage_debug_ledger_size,
)
from rayorch.experimental.multigrain_v3.ray.worker import normalize_outputs


class SignatureOp:
    """UDF exercising positional-only, positional/keyword, and keyword-only."""

    def run(
        self,
        left: list[str],
        /,
        right: list[str],
        *,
        suffix: list[str],
    ) -> list[str]:
        """Return one aligned output column."""

        raise NotImplementedError


class SignaturePipeline(Pipeline[str]):
    """Pipeline binding the same source through all supported call forms."""

    def __init__(self) -> None:
        self.operation = Map(SignatureOp)

    def forward(self, source):
        """Bind positional-only and explicit keyword roles."""

        result = self.operation(source, right=source, suffix=source)
        return {"result": result}


class AnonymousTupleOp:
    """UDF with a fixed anonymous tuple return."""

    def run(self, values: list[str]) -> tuple[list[int], list[str]]:
        """Return two aligned output columns."""

        raise NotImplementedError


class NamedOutputs(NamedTuple):
    """NamedTuple output shape mirrored with SymbolicPort leaves."""

    accepted: list[str]
    mask: list[bool]


class NamedTupleOp:
    """UDF with a concrete importable NamedTuple return."""

    def run(self, values: list[str]) -> NamedOutputs:
        """Return payload and strict bool mask columns."""

        raise NotImplementedError


class MultiReturnPipeline(Pipeline[str]):
    """Pipeline retaining anonymous and named return leaves."""

    def __init__(self) -> None:
        self.anonymous = Map(AnonymousTupleOp)
        self.named = Map(NamedTupleOp)
        self.observed_named_type = None

    def forward(self, source):
        """Use tuple unpacking and NamedTuple field access."""

        number, text = self.anonymous(source)
        named = self.named(source)
        self.observed_named_type = type(named)
        kept = filter(named.mask, named.accepted)
        return {
            "number": number,
            "text": text,
            "kept": kept,
        }


class ChainedFilterPipeline(Pipeline[str]):
    """Pipeline incorrectly reusing a system FILTER bool as another mask."""

    def __init__(self) -> None:
        self.named = Map(NamedTupleOp)

    def forward(self, source):
        """Attempt a chained FILTER mask not produced directly by MAP."""

        named = self.named(source)
        system_bool = filter(named.mask, named.mask)
        return {"invalid": filter(system_bool, named.accepted)}


class NestedListOp:
    """UDF returning two Python list layers beneath the physical batch."""

    def run(self, values: list[str]) -> list[list[list[int]]]:
        """Expose only the first per-grain list layer structurally."""

        raise NotImplementedError


class AnyGroupsOp:
    """UDF exposing a structural list whose opaque elements are Any."""

    def run(self, values: list[str]) -> list[list[Any]]:
        """Return one structurally expandable wildcard group per grain."""

        raise NotImplementedError


class AnyIdentityOp:
    """UDF consuming wildcard items after one structural expansion."""

    def run(self, values: list[Any]) -> list[Any]:
        """Preserve one wildcard item per occurrence."""

        raise NotImplementedError


class AnyGroupsPipeline(Pipeline[str]):
    """Pipeline proving structural Any leaves remain runtime wildcards."""

    def __init__(self) -> None:
        self.groups = Map(AnyGroupsOp)
        self.identity = Map(AnyIdentityOp)

    def forward(self, source):
        """Expand wildcard items once, map them, then close their scope."""

        items = expand(self.groups(source))
        return {"items": reduce(self.identity(items))}


class NestedListPipeline(Pipeline[str]):
    """Pipeline expanding the one exposed nested-list layer."""

    def __init__(self) -> None:
        self.nested = Map(NestedListOp)

    def forward(self, source):
        """Expand once and collect the opaque inner-list items."""

        groups = self.nested(source)
        inner_lists = expand(groups)
        return {"collected": reduce(inner_lists)}


class InvalidSecondExpandPipeline(Pipeline[str]):
    """Pipeline incorrectly trying to expand an opaque inner Python list."""

    def __init__(self) -> None:
        self.nested = Map(NestedListOp)

    def forward(self, source):
        """Attempt a forbidden second direct EXPAND."""

        first = expand(self.nested(source))
        return {"invalid": expand(first)}


class OpaqueListOp:
    """UDF explicitly marking a per-grain list as opaque."""

    def run(
        self,
        values: list[str],
    ) -> list[Annotated[list[int], OpaqueValue]]:
        """Return a Python list payload with no structural layout metadata."""

        raise NotImplementedError


class OpaquePipeline(Pipeline[str]):
    """Pipeline returning an opaque list payload without expansion."""

    def __init__(self) -> None:
        self.operation = Map(OpaqueListOp)

    def forward(self, source):
        """Keep the opaque list at root scope."""

        return {"tokens": self.operation(source)}


class InvalidOpaqueExpandPipeline(Pipeline[str]):
    """Pipeline trying to EXPAND an OpaqueValue-marked output."""

    def __init__(self) -> None:
        self.operation = Map(OpaqueListOp)

    def forward(self, source):
        """Attempt an invalid structural operation."""

        return {"tokens": expand(self.operation(source))}


class OuterGroupsOp:
    """Create the outer dynamic list."""

    def run(self, values: list[str]) -> list[list[int]]:
        """Return one structural list of ints per source record."""

        raise NotImplementedError


class InnerGroupsOp:
    """Create a nested dynamic list for each outer item."""

    def run(self, values: list[int]) -> list[list[str]]:
        """Return one structural list of strings per outer occurrence."""

        raise NotImplementedError


class InnerMapOp:
    """Map inner scalar strings before inner reduction."""

    def run(self, values: list[str]) -> list[str]:
        """Return one string per nested occurrence."""

        raise NotImplementedError


class AssembleOuterOp:
    """Combine an outer scalar with its inner REDUCE result."""

    def run(
        self,
        outer: list[int],
        inner: list[list[str]],
    ) -> list[str]:
        """Return one value in the restored outer occurrence domain."""

        raise NotImplementedError


class PathLocalScopePipeline(Pipeline[str]):
    """Nested and sibling reductions proving path-local LIFO behavior."""

    def __init__(self) -> None:
        self.outer = Map(OuterGroupsOp)
        self.inner = Map(InnerGroupsOp)
        self.inner_map = Map(InnerMapOp)
        self.assemble = Map(AssembleOuterOp)

    def forward(self, source):
        """Close inner first while another branch closes outer directly."""

        outer_groups = self.outer(source)
        outer_items = expand(outer_groups)
        inner_groups = self.inner(outer_items)
        inner_items = expand(inner_groups)
        inner_values = self.inner_map(inner_items)
        reduced_inner = reduce(inner_values)
        assembled = self.assemble(outer_items, reduced_inner)
        assembled_root = reduce(assembled)
        direct_root = reduce(outer_items)
        return {
            "assembled": assembled_root,
            "direct": direct_root,
        }


class FrozenConfigOp:
    """UDF with nested mutable constructor configuration."""

    def __init__(
        self,
        config: dict[str, object],
        *,
        labels: list[str],
    ) -> None:
        """Capture constructor values that a Worker must receive as containers."""

        self.config = config
        self.labels = labels

    def run(self, values: list[str]) -> list[str]:
        """Return one scalar output per input grain."""

        raise NotImplementedError


class ExceptionAtomicOp:
    """UDF declaring actor replacement is unnecessary after exceptions."""

    exception_atomic = True

    def run(self, values: list[str]) -> list[str]:
        """Return one scalar output per input grain."""

        raise NotImplementedError


class FrozenConfigPipeline(Pipeline[str]):
    """Pipeline compiling one externally configured immutable Map spec."""

    def __init__(self, operation: Map) -> None:
        self.operation = operation

    def forward(self, source):
        """Expose the configured MAP output at root scope."""

        return {"result": self.operation(source)}


class MissingParameterAnnotationOp:
    """Invalid UDF with an unannotated data parameter."""

    def run(self, values) -> list[str]:
        """Invalid run contract."""

        raise NotImplementedError


class MissingReturnAnnotationOp:
    """Invalid UDF with no return annotation."""

    def run(self, values: list[str]):
        """Invalid run contract."""

        raise NotImplementedError


class ScalarParameterOp:
    """Invalid UDF whose input is not a physical batch column."""

    def run(self, values: str) -> list[str]:
        """Invalid run contract."""

        raise NotImplementedError


class ScalarReturnOp:
    """Invalid UDF whose return is not a physical batch column."""

    def run(self, values: list[str]) -> str:
        """Invalid run contract."""

        raise NotImplementedError


class VariadicOp:
    """Invalid UDF using a dynamic parameter list."""

    def run(self, *values: list[str]) -> list[str]:
        """Invalid run contract."""

        raise NotImplementedError


class StaticRunOp:
    """Invalid UDF exposing run as a static method."""

    @staticmethod
    def run(values: list[str]) -> list[str]:
        """Invalid run descriptor."""

        raise NotImplementedError


class InvalidUdfPipeline(Pipeline[str]):
    """Minimal pipeline compiling one selected invalid UDF."""

    def __init__(self, op_cls: type) -> None:
        self.operation = Map(op_cls)

    def forward(self, source):
        """Invoke the selected invalid operation."""

        return {"result": self.operation(source)}


def _map_nodes(graph):
    """Return frozen MAP nodes in topological order."""

    return tuple(node for node in graph.nodes if isinstance(node.op, MapOp))


def test_signature_inference_freezes_binding_and_shapes() -> None:
    graph = SignaturePipeline().compile()
    map_node = _map_nodes(graph)[0]

    assert tuple(binding.role for binding in map_node.inputs) == (
        "left",
        "right",
        "suffix",
    )
    assert tuple(
        parameter.kind for parameter in map_node.op.call_schema.parameters
    ) == (
        ParameterKind.POSITIONAL_ONLY,
        ParameterKind.POSITIONAL_OR_KEYWORD,
        ParameterKind.KEYWORD_ONLY,
    )
    assert map_node.op.call_schema.positional_count == 1
    assert map_node.op.call_schema.keyword_roles == ("right", "suffix")
    assert map_node.op.return_schema.kind is ReturnKind.SINGLE
    assert isinstance(map_node.outputs[0].shape, OpaqueShape)


def test_tuple_and_named_tuple_returns_are_mirrored() -> None:
    pipeline = MultiReturnPipeline()
    graph = pipeline.compile()
    anonymous, named = _map_nodes(graph)

    assert pipeline.observed_named_type is NamedOutputs
    assert anonymous.op.return_schema.kind is ReturnKind.TUPLE
    assert tuple(port.name for port in anonymous.outputs) == (
        "output_0",
        "output_1",
    )
    assert named.op.return_schema.kind is ReturnKind.NAMED_TUPLE
    assert tuple(port.name for port in named.outputs) == ("accepted", "mask")
    assert isinstance(named.outputs[1].shape, BoolShape)
    assert named.op.physical_outputs[1].emit_control_bits


def test_nested_python_list_exposes_exactly_one_structural_layer() -> None:
    graph = NestedListPipeline().compile()
    nested_map = _map_nodes(graph)[0]
    map_shape = nested_map.outputs[0].shape

    assert isinstance(map_shape, StructuralListShape)
    assert isinstance(map_shape.element, OpaqueShape)
    assert map_shape.element.type_ref is not None
    assert map_shape.element.type_ref.qualname == "list"
    expand_node = next(
        node for node in graph.nodes if isinstance(node.op, ExpandOp)
    )
    assert expand_node.outputs[0].shape == map_shape.element

    with pytest.raises(
        CompileError,
        match="ExpandRequiresStructuralList",
    ):
        InvalidSecondExpandPipeline().compile()


def test_structural_list_any_element_stays_opaque_wildcard() -> None:
    graph = AnyGroupsPipeline().compile()
    groups, identity = _map_nodes(graph)
    group_shape = groups.outputs[0].shape

    assert isinstance(group_shape, StructuralListShape)
    assert group_shape.element == OpaqueShape(None)
    expand_node = next(
        node for node in graph.nodes if isinstance(node.op, ExpandOp)
    )
    assert expand_node.outputs[0].shape == OpaqueShape(None)
    assert identity.inputs[0].parameter.item_shape == OpaqueShape(None)
    assert identity.outputs[0].shape == OpaqueShape(None)

    sentinel = object()
    columns, layouts = normalize_outputs(
        [[sentinel, {"arbitrary": "payload"}]],
        batch_size=1,
        return_schema=groups.op.return_schema,
        output_schema=groups.op.physical_outputs,
    )
    assert columns[0][0] is sentinel
    assert columns[0][1] == {"arbitrary": "payload"}
    assert layouts[0].offsets == (0, 2)


def test_opaque_value_marker_prevents_structural_expand() -> None:
    graph = OpaquePipeline().compile()
    output_shape = _map_nodes(graph)[0].outputs[0].shape

    assert isinstance(output_shape, OpaqueShape)
    assert output_shape.type_ref is not None
    assert output_shape.type_ref.qualname == "list"
    with pytest.raises(
        CompileError,
        match="ExpandRequiresStructuralList",
    ):
        InvalidOpaqueExpandPipeline().compile()


def test_map_configuration_is_defensively_copied_at_both_boundaries() -> None:
    runtime_env = {
        "env_vars": {"MODE": "before"},
        "py_modules": ["/first/module"],
        "tags": {"stable", "worker"},
    }
    constructor_config = {
        "nested": {"values": [1, 2]},
        "flags": {"first"},
    }
    labels = ["alpha"]
    operation = Map(
        FrozenConfigOp,
        runtime_env=runtime_env,
    ).pre_init(constructor_config, labels=labels)
    graph = FrozenConfigPipeline(operation).compile()
    fingerprint = graph.fingerprint

    runtime_env["env_vars"]["MODE"] = "after"
    runtime_env["py_modules"].append("/second/module")
    runtime_env["tags"].add("mutated")
    constructor_config["nested"]["values"].append(3)
    constructor_config["flags"].add("second")
    labels.append("beta")

    frozen_map_env = dict(operation.resources.runtime_env)
    map_env = thaw_config_value(frozen_map_env)
    assert map_env == {
        "env_vars": {"MODE": "before"},
        "py_modules": ["/first/module"],
        "tags": {"stable", "worker"},
    }
    assert thaw_config_value(operation.init_args) == (
        {
            "nested": {"values": [1, 2]},
            "flags": {"first"},
        },
    )
    assert thaw_config_value(dict(operation.init_kwargs)) == {
        "labels": ["alpha"]
    }
    assert not isinstance(frozen_map_env["env_vars"], dict)
    assert not isinstance(frozen_map_env["py_modules"], list)
    assert not isinstance(frozen_map_env["tags"], set)

    map_op = _map_nodes(graph)[0].op
    graph_env = thaw_config_value(
        dict(map_op.execution.resources.runtime_env)
    )
    assert graph_env == map_env
    assert isinstance(graph_env["env_vars"], dict)
    assert isinstance(graph_env["py_modules"], list)
    assert isinstance(graph_env["tags"], set)
    assert map_op.udf.init_args == operation.init_args
    assert thaw_config_value(dict(map_op.udf.init_kwargs)) == {
        "labels": ["alpha"]
    }

    recompiled = FrozenConfigPipeline(operation).compile()
    assert recompiled.fingerprint == fingerprint
    assert recompiled == graph
    graph.verify()
    restored = pickle.loads(pickle.dumps(graph))
    assert restored == graph
    restored.verify()
    assert pickle.dumps(restored, protocol=5) == pickle.dumps(graph, protocol=5)


def test_filter_mask_must_be_a_direct_map_bool_output() -> None:
    with pytest.raises(
        CompileError,
        match="FilterMaskNotMapOutput",
    ) as captured:
        ChainedFilterPipeline().compile()

    assert captured.value.code == "FilterMaskNotMapOutput"


def test_equivalent_configurations_have_canonical_graph_identity() -> None:
    left = Map(
        FrozenConfigOp,
        runtime_env={
            "env_vars": {"B": "2", "A": "1"},
            "tags": {"beta", "alpha"},
        },
    ).pre_init(
        {"second": [2], "first": [1]},
        labels=["same"],
    )
    right = Map(
        FrozenConfigOp,
        runtime_env={
            "tags": {"alpha", "beta"},
            "env_vars": {"A": "1", "B": "2"},
        },
    ).pre_init(
        {"first": [1], "second": [2]},
        labels=["same"],
    )

    left_graph = FrozenConfigPipeline(left).compile()
    right_graph = FrozenConfigPipeline(right).compile()
    assert left_graph.fingerprint == right_graph.fingerprint
    assert left_graph == right_graph


@pytest.mark.parametrize("entry_module", ["__main__", "__mp_main__"])
def test_process_entry_module_udfs_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    entry_module: str,
) -> None:
    monkeypatch.setattr(SignatureOp, "__module__", entry_module)

    with pytest.raises(CompileError, match="NonCanonicalTypeRef"):
        SignaturePipeline().compile()
    with pytest.raises(ValueError, match="process entry module"):
        SerializableCallableRef(entry_module, "SignatureOp")


def test_exception_atomic_is_frozen_into_udf_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = InvalidUdfPipeline(ExceptionAtomicOp).compile()
    monkeypatch.setattr(ExceptionAtomicOp, "exception_atomic", False)

    assert _map_nodes(graph)[0].op.udf.exception_atomic is True


def test_identity_preimage_debug_ledger_is_bounded() -> None:
    for index in range(PREIMAGE_DEBUG_LEDGER_LIMIT + 50):
        GraphFingerprint.derive("bounded-ledger-test", index)

    assert preimage_debug_ledger_size() <= PREIMAGE_DEBUG_LEDGER_LIMIT


def test_current_mineru_annotations_match_strict_compiler_shapes() -> None:
    render_call, render_return = infer_run_schemas(MinerUPdfToPages)
    ocr_call, ocr_return = infer_run_schemas(MinerUVlmOcrPage)
    assemble_call, assemble_return = infer_run_schemas(MinerUAssembleDoc)

    assert isinstance(render_return.leaves[0].item_shape, StructuralListShape)
    render_item = render_return.leaves[0].item_shape.element
    assert isinstance(render_item, OpaqueShape)
    assert render_item.type_ref is not None
    assert render_item.type_ref.qualname == "dict"
    assert ocr_call.parameters[0].item_shape == render_item
    assert ocr_return.leaves[0].item_shape == OpaqueShape(None)

    assert assemble_call.parameters[0].item_shape == OpaqueShape(
        render_call.parameters[0].item_shape.type_ref
    )
    contents_shape = assemble_call.parameters[1].item_shape
    pages_shape = assemble_call.parameters[2].item_shape
    assert contents_shape == StructuralListShape(OpaqueShape(None))
    assert isinstance(pages_shape, StructuralListShape)
    assert pages_shape.element == render_item
    assert isinstance(assemble_return.leaves[0].item_shape, OpaqueShape)


def test_scope_paths_are_path_local_lifo() -> None:
    graph = PathLocalScopePipeline().compile()
    expand_nodes = tuple(
        node for node in graph.nodes if isinstance(node.op, ExpandOp)
    )
    reduce_nodes = tuple(
        node for node in graph.nodes if isinstance(node.op, ReduceOp)
    )

    assert len(expand_nodes) == 2
    outer_scope = expand_nodes[0].op.scope
    inner_scope = expand_nodes[1].op.scope
    assert expand_nodes[0].outputs[0].scope_path == (outer_scope,)
    assert expand_nodes[1].outputs[0].scope_path == (
        outer_scope,
        inner_scope,
    )
    assert reduce_nodes[0].op.closes_scope == inner_scope
    assert reduce_nodes[0].outputs[0].scope_path == (outer_scope,)
    outer_reducers = tuple(
        node for node in reduce_nodes if node.op.closes_scope == outer_scope
    )
    assert len(outer_reducers) == 2
    assert set(graph.scope_plans[outer_scope].reducers) == {
        node.id for node in outer_reducers
    }
    assert all(not graph.port(output.port).scope_path for output in graph.outputs)


@pytest.mark.parametrize(
    ("op_cls", "error_code"),
    [
        (MissingParameterAnnotationOp, "MissingRunParameterAnnotation"),
        (MissingReturnAnnotationOp, "MissingRunReturnAnnotation"),
        (ScalarParameterOp, "UnsupportedBatchColumnAnnotation"),
        (ScalarReturnOp, "UnsupportedReturnStructure"),
        (VariadicOp, "UnsupportedRunParameter"),
        (StaticRunOp, "UnsupportedRunDescriptor"),
    ],
)
def test_invalid_run_annotations_are_compile_errors(
    op_cls: type,
    error_code: str,
) -> None:
    with pytest.raises(CompileError, match=error_code) as captured:
        InvalidUdfPipeline(op_cls).compile()

    assert captured.value.code == error_code
