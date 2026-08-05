"""Multigrain v3.3 的单写者、事件驱动 Arena 语义状态机。"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainOutcome,
    GrainPhase,
    GrainRef,
    InputMode,
    ItemOutcome,
    ItemRef,
    PortRef,
    ShapeState,
)
from ..program import (
    BroadcastOrigin,
    CallConsumer,
    CallOutputOrigin,
    ExpandOrigin,
    FilterOrigin,
    GroupOrigin,
    Program,
    ViewConsumer,
)
from ..protocol import (
    CallFailureReport,
    CallReport,
    GroupTake,
    InvocationPlan,
    MissingTake,
    OutputReport,
    RowBinding,
    ScalarTake,
    WorkerReport,
)
from .state import (
    ArenaLimits,
    CommitError,
    EntityOrigin,
    GrainRecord,
    GroupBinding,
    GroupShape,
    ItemRecord,
    PendingInvocation,
    RuntimeState,
    ShapeKey,
    ShapeRecord,
    ValueBinding,
)


@dataclass(frozen=True, slots=True)
class _ExpansionCommit:
    source_port: PortRef
    child_port: PortRef
    child_domain: DomainRef
    rows: tuple[RowBinding, ...]


class ArenaEngine:
    """拥有一个 source microbatch 的全部语义事实与唯一写入口。"""

    def __init__(
        self,
        program: Program,
        *,
        limits: ArenaLimits | None = None,
    ) -> None:
        self.program = program
        self.limits = limits or ArenaLimits()
        self.state = RuntimeState()
        self._receipts: deque[ItemRef] = deque()
        self._ready: deque[GrainRef] = deque()
        self._entities_by_domain: dict[DomainRef, list[EntityRef]] = defaultdict(list)
        self._children_by_shape: dict[ShapeKey, tuple[EntityRef, ...]] = {}
        self._mask_values: dict[ItemRef, bool] = {}
        self._group_slots = 0
        self._admission_closed = False

        self._expanded_by_source: dict[PortRef, list[PortRef]] = defaultdict(list)
        self._groups_by_child_domain: dict[DomainRef, list[PortRef]] = defaultdict(list)
        self._broadcasts_by_target_domain: dict[DomainRef, list[PortRef]] = defaultdict(list)
        for port, spec in program.ports.items():
            origin = spec.origin
            if isinstance(origin, ExpandOrigin):
                self._expanded_by_source[origin.group_port].append(port)
            elif isinstance(origin, GroupOrigin):
                child_domain = program.port(origin.value_port).domain
                self._groups_by_child_domain[child_domain].append(port)
            elif isinstance(origin, BroadcastOrigin):
                self._broadcasts_by_target_domain[spec.domain].append(port)

    @property
    def ready_count(self) -> int:
        """返回当前可被执行器预留的 Grain 数。"""

        return len(self._ready)

    @property
    def entity_count(self) -> int:
        """返回 Arena 已创建的 Entity 总数，供审计与 benchmark 使用。"""

        return sum(len(entities) for entities in self._entities_by_domain.values())

    @property
    def item_count(self) -> int:
        """返回 Arena 已终态化的 Item 总数。"""

        return len(self.state.items)

    @property
    def shape_count(self) -> int:
        """返回 Arena 已创建的 fan-out Shape 总数。"""

        return len(self.state.shapes)

    def ready_calls(self) -> tuple[CallRef, ...]:
        """按 ready queue 首次出现顺序返回可调度 Call。"""

        seen: set[CallRef] = set()
        calls = []
        for grain in self._ready:
            if grain.call not in seen and self.state.grains[grain].phase is GrainPhase.READY:
                seen.add(grain.call)
                calls.append(grain.call)
        return tuple(calls)

    def reserve_batch(
        self,
        call: CallRef,
        *,
        max_size: int,
        parent_bound: bool = False,
    ) -> tuple[GrainRef, ...]:
        """预留 READY Grains；parent_bound 只改变 packing，不改变身份。"""

        if max_size <= 0:
            raise ValueError("max_size must be positive")
        selected: list[GrainRef] = []
        remaining: deque[GrainRef] = deque()
        parent: EntityRef | None = None
        while self._ready:
            grain = self._ready.popleft()
            record = self.state.grains[grain]
            if record.phase is not GrainPhase.READY:
                continue
            if grain.call != call or len(selected) >= max_size:
                remaining.append(grain)
                continue
            candidate_parent = self._batch_parent(grain.entity)
            if parent_bound and parent is not None and candidate_parent != parent:
                remaining.append(grain)
                continue
            if parent is None:
                parent = candidate_parent
            record.phase = GrainPhase.IN_FLIGHT
            record.active_attempt = record.generation
            selected.append(grain)
        self._ready = remaining
        return tuple(selected)

    def close_admission(self) -> None:
        """声明本 Arena 不再接纳 source，允许完整性判定成立。"""

        self._admission_closed = True

    def is_complete(self) -> bool:
        """按完整合同审计 Arena，而非仅检查 ready queue。"""

        if not self._admission_closed:
            return False
        if self._receipts or self.state.pending or self._ready:
            return False
        if any(record.phase is not GrainPhase.SEALED for record in self.state.grains.values()):
            return False
        if any(shape.state is ShapeState.OPEN for shape in self.state.shapes.values()):
            return False
        for port in self._output_ports(self.program.output_tree):
            for entity in self._entities_by_domain.get(self.program.port(port).domain, ()):
                if ItemRef(port, entity) not in self.state.items:
                    return False
        return True

    def _output_ports(self, tree: object) -> tuple[PortRef, ...]:
        if isinstance(tree, PortRef):
            return (tree,)
        if isinstance(tree, tuple):
            return tuple(
                port for item in tree for port in self._output_ports(item)
            )
        raise CommitError("invalid Program.output_tree")

    def _batch_parent(self, entity: EntityRef) -> EntityRef:
        parent = self._parent_of(entity)
        return entity if parent is None else parent

    def entities(self, domain: DomainRef) -> tuple[EntityRef, ...]:
        """返回 Domain 内已经创建的 Entity，顺序只用于审计。"""

        return tuple(self._entities_by_domain.get(domain, ()))

    def invocation_plan(self, grain: GrainRef) -> InvocationPlan:
        """把 Arena 语义事实投影为 Worker 可消费的纯物理输入计划。"""

        record = self.state.grains[grain]
        call = self.program.call(grain.call)
        takes = []
        for input_spec, item in zip(call.inputs, record.inputs):
            receipt = self.state.items[item]
            if (
                input_spec.mode is InputMode.OPTIONAL
                and receipt.outcome is ItemOutcome.DROPPED
            ):
                takes.append(MissingTake())
                continue
            binding = self.state.values[item]
            if isinstance(binding, RowBinding):
                takes.append(ScalarTake(binding))
                continue
            if isinstance(binding, GroupBinding):
                rows = tuple(self.state.values[leaf] for leaf in binding.flat_items)
                if not all(isinstance(row, RowBinding) for row in rows):
                    raise CommitError("canonical group leaves must resolve to rows")
                takes.append(
                    GroupTake(
                        tuple(row for row in rows if isinstance(row, RowBinding)),
                        binding.shape.offsets_by_level,
                    )
                )
                continue
            raise CommitError(f"unsupported ValueBinding: {binding!r}")
        return InvocationPlan(grain, record.generation, tuple(takes))

    def ordered_items(self, port: PortRef) -> tuple[ItemRef, ...]:
        """按 source/ordinal coordinate 返回一个 Port 的全部 Item。"""

        domain = self.program.port(port).domain
        entities = sorted(
            self._entities_by_domain.get(domain, ()),
            key=self.entity_coordinate,
        )
        return tuple(ItemRef(port, entity) for entity in entities)

    def item_outcome(self, item: ItemRef) -> ItemOutcome:
        """读取已经终态化 Item 的 outcome。"""

        return self.state.items[item].outcome

    def value_binding(self, item: ItemRef) -> ValueBinding:
        """读取 PRESENT Item 的物理 binding；调用方不得据此改变语义状态。"""

        return self.state.values[item]

    def group_rows(self, binding: GroupBinding) -> tuple[RowBinding, ...]:
        """把 canonical group 叶子解析为行引用，不读取业务 payload。"""

        rows = tuple(self.state.values[leaf] for leaf in binding.flat_items)
        if not all(isinstance(row, RowBinding) for row in rows):
            raise CommitError("canonical group leaves must resolve to rows")
        return tuple(row for row in rows if isinstance(row, RowBinding))

    def entity_coordinate(self, entity: EntityRef) -> tuple[int, ...]:
        """沿显式 lineage 计算稳定的 root/ordinal path。"""

        path = []
        cursor = entity
        while cursor in self.state.entity_lineage:
            origin = self.state.entity_lineage[cursor]
            path.append(origin.ordinal)
            cursor = origin.parent_entity
        return (cursor.value, *reversed(path))

    def release_values(self) -> int:
        """完成后释放物理 bindings，同时保留 Item/Shape/Grain 语义事实。"""

        if not self.is_complete():
            raise CommitError("cannot release values before Arena completion")
        released = len(self.state.values)
        self.state.values.clear()
        return released

    def infra_failures(self, grain: GrainRef) -> int:
        """返回 Grain 已发生的基础设施失败次数。"""

        return self.state.grains[grain].infra_failures

    def progress_summary(self) -> str:
        """返回不含业务 payload 的死锁诊断摘要。"""

        return (
            f"pending={len(self.state.pending)}, "
            f"grains={len(self.state.grains)}, "
            f"shapes={len(self.state.shapes)}"
        )

    def admit_sources(
        self,
        bindings: Mapping[PortRef, tuple[RowBinding, ...]],
    ) -> tuple[EntityRef, ...]:
        """原子接纳行对齐 source bindings，并触发第一轮事件传播。"""

        if set(bindings) != set(self.program.source_ports):
            raise CommitError("source bindings must exactly match Program.source_ports")
        counts = {len(rows) for rows in bindings.values()}
        if len(counts) != 1:
            raise CommitError("row-aligned sources must have equal cardinality")
        count = next(iter(counts))
        if len(self._entities_by_domain[self.program.port(self.program.source_ports[0]).domain]):
            raise CommitError("sources have already been admitted")
        if count > self.limits.max_entities:
            raise CommitError("source admission exceeds max_entities")
        self._ensure_item_capacity(count * len(bindings))

        root_domain = self.program.port(self.program.source_ports[0]).domain
        entities = tuple(EntityRef(root_domain, index) for index in range(count))
        self._entities_by_domain[root_domain].extend(entities)
        for source, rows in bindings.items():
            for entity, row in zip(entities, rows):
                self._publish(
                    ItemRef(source, entity),
                    ItemOutcome.PRESENT,
                    binding=row,
                )
        self.advance()
        return entities

    def advance(self) -> None:
        """消费 publication receipts，直到结构传播达到局部不动点。"""

        while self._receipts:
            item = self._receipts.popleft()
            for consumer in self.program.consumers_by_port.get(item.port, ()):
                if isinstance(consumer, CallConsumer):
                    self._accept_call_input(consumer, item)
                elif isinstance(consumer, ViewConsumer):
                    self._advance_view(consumer.port, item)

    def reserve_ready(self) -> GrainRef:
        """预留一个 READY Grain，并记录当前 generation 的 attempt。"""

        while self._ready:
            grain = self._ready.popleft()
            record = self.state.grains[grain]
            if record.phase is GrainPhase.READY:
                record.phase = GrainPhase.IN_FLIGHT
                record.active_attempt = record.generation
                return grain
        raise LookupError("no READY Grain")

    def commit_report(self, report: WorkerReport) -> None:
        """按报告类型进入唯一的成功/失败语义提交路径。"""

        if isinstance(report, CallFailureReport):
            self.commit_failure(
                report.grain,
                report.cause,
                generation=report.generation,
            )
        else:
            self.commit_success(report)

    def commit_success(self, report: CallReport) -> None:
        """校验并发布一个成功 Grain 的全部输出、Shape 与 child Entities。"""

        grain = report.grain
        record = self.state.grains.get(grain)
        if record is None or record.phase is not GrainPhase.IN_FLIGHT:
            raise CommitError("success report requires one IN_FLIGHT Grain")
        if report.generation != record.generation:
            raise CommitError("stale generation")

        expected_outputs = self.program.outputs_by_call[grain.call]
        by_port = {output.port: output for output in report.outputs}
        if len(by_port) != len(report.outputs) or set(by_port) != set(expected_outputs):
            raise CommitError("report outputs must exactly match Call outputs")

        scalar_commits: list[tuple[ItemRef, OutputReport]] = []
        expansion_commits: list[_ExpansionCommit] = []
        counts_by_shape: dict[ShapeKey, list[int]] = defaultdict(list)
        reporters_by_shape: dict[ShapeKey, set[PortRef]] = defaultdict(set)

        for output_port in expected_outputs:
            output = by_port[output_port]
            expanded_ports = tuple(self._expanded_by_source.get(output_port, ()))
            expanded_by_port = {item.port: item for item in output.expansions}
            if len(expanded_by_port) != len(output.expansions):
                raise CommitError("duplicate expanded output report")
            if set(expanded_by_port) != set(expanded_ports):
                raise CommitError("expanded reports do not match Port demand")
            if expanded_ports and output.scalar is not None:
                raise CommitError("expanded Call output must not duplicate scalar payload")
            if not expanded_ports and output.scalar is None:
                raise CommitError("non-expanded Call output requires one scalar binding")

            parent_item = ItemRef(output_port, grain.entity)
            scalar_commits.append((parent_item, output))
            for child_port in expanded_ports:
                rows = expanded_by_port[child_port].rows
                child_domain = self.program.port(child_port).domain
                shape = ShapeKey(child_domain, grain.entity)
                counts_by_shape[shape].append(len(rows))
                reporters_by_shape[shape].add(output_port)
                expansion_commits.append(
                    _ExpansionCommit(output_port, child_port, child_domain, rows)
                )

        for shape, counts in counts_by_shape.items():
            if len(set(counts)) != 1:
                raise CommitError("aligned expansion cardinality mismatch")
            expected = set(self.program.shape_reporters_by_domain[shape.child_domain])
            if reporters_by_shape[shape] != expected:
                raise CommitError("aligned expansion reporters are incomplete")
            if shape in self.state.shapes:
                raise CommitError("Shape has already been published")
            count = counts[0]
            domain_limit = self.program.domain(shape.child_domain).max_fanout
            if domain_limit is not None and count > domain_limit:
                raise CommitError("expansion exceeds Domain max_fanout")

        new_entities = sum(counts[0] for counts in counts_by_shape.values())
        if len(self.state.entity_lineage) + new_entities > self.limits.max_entities:
            raise CommitError("commit exceeds max_entities")
        # 每个 Call output 发布一个 parent Item；每个 expanded row 另外发布
        # 一个 child Item。先统一预留，防止 multi-output 中途越界。
        self._ensure_item_capacity(
            len(expected_outputs)
            + sum(len(commit.rows) for commit in expansion_commits)
        )

        # 至此所有 report/shape/cardinality/limit 均已验证；后续 publication
        # 对状态机而言是一个不可分割的逻辑 turn。
        record.phase = GrainPhase.SEALED
        record.outcome = GrainOutcome.SUCCESS
        record.active_attempt = None

        commits_by_shape: dict[ShapeKey, list[_ExpansionCommit]] = defaultdict(list)
        for commit in expansion_commits:
            commits_by_shape[ShapeKey(commit.child_domain, grain.entity)].append(commit)

        for shape, commits in commits_by_shape.items():
            count = len(commits[0].rows)
            reporters = self.program.shape_reporters_by_domain[shape.child_domain]
            self.state.shapes[shape] = ShapeRecord(
                ShapeState.SUCCEEDED,
                count,
                reporters,
                {commit.source_port: count for commit in commits},
            )
            children = self._create_children(shape, count)
            for commit in commits:
                leaves = []
                for child, row in zip(children, commit.rows):
                    item = ItemRef(commit.child_port, child)
                    leaves.append(item)
                    self._publish(item, ItemOutcome.PRESENT, binding=row)
                parent_item = ItemRef(commit.source_port, grain.entity)
                self._publish(
                    parent_item,
                    ItemOutcome.PRESENT,
                    binding=GroupBinding(
                        GroupShape.one_level(count),
                        tuple(leaves),
                    ),
                    control=by_port[commit.source_port].control,
                )
            self._shape_terminal(shape)

        expanded_sources = {commit.source_port for commit in expansion_commits}
        for item, output in scalar_commits:
            if item.port in expanded_sources:
                continue
            self._publish(
                item,
                ItemOutcome.PRESENT,
                binding=output.scalar,
                control=output.control,
            )
        self.advance()

    def commit_failure(
        self,
        grain: GrainRef,
        cause: object,
        *,
        generation: int | None = None,
    ) -> None:
        """发布计算失败；fan-out 前失败以 unknown cardinality 封闭 Shape。"""

        record = self.state.grains.get(grain)
        if record is None or record.phase is not GrainPhase.IN_FLIGHT:
            raise CommitError("failure requires one IN_FLIGHT Grain")
        if generation is not None and generation != record.generation:
            raise CommitError("stale generation")

        outputs = self.program.outputs_by_call[grain.call]
        self._ensure_item_capacity(len(outputs))
        record.phase = GrainPhase.SEALED
        record.outcome = GrainOutcome.FAILED
        record.active_attempt = None
        for output in outputs:
            self._publish(ItemRef(output, grain.entity), ItemOutcome.FAILED, cause=cause)
            for child_port in self._expanded_by_source.get(output, ()):
                shape = ShapeKey(self.program.port(child_port).domain, grain.entity)
                if shape not in self.state.shapes:
                    self.state.shapes[shape] = ShapeRecord(
                        ShapeState.FAILED,
                        None,
                        self.program.shape_reporters_by_domain[shape.child_domain],
                        cause=cause,
                    )
                    self._shape_terminal(shape)
        self.advance()

    def retry(self, grain: GrainRef) -> None:
        """保留 Grain 身份并提升 generation，以 fence 隔离旧报告。"""

        record = self.state.grains[grain]
        if record.phase is not GrainPhase.IN_FLIGHT:
            raise CommitError("only an IN_FLIGHT Grain can be retried")
        record.generation += 1
        record.infra_failures += 1
        record.active_attempt = None
        record.phase = GrainPhase.READY
        self._ready.append(grain)

    def _ensure_item_capacity(self, additional: int) -> None:
        """在一个逻辑 turn 发布前统一检查 ItemTable hard limit。"""

        if additional < 0:
            raise ValueError("additional item count must be non-negative")
        if len(self.state.items) + additional > self.limits.max_items:
            raise CommitError("commit exceeds max_items")

    def _publish(
        self,
        item: ItemRef,
        outcome: ItemOutcome,
        *,
        binding: ValueBinding | None = None,
        cause: object | None = None,
        control: bool | None = None,
    ) -> None:
        existing = self.state.items.get(item)
        record = ItemRecord(outcome, cause)
        if existing is not None:
            existing_binding = self.state.values.get(item)
            if existing != record or existing_binding != binding:
                raise CommitError(f"conflicting Item publication: {item}")
            return
        if outcome is ItemOutcome.PRESENT and binding is None:
            raise CommitError("PRESENT Item requires a ValueBinding")
        if outcome is not ItemOutcome.PRESENT and binding is not None:
            raise CommitError("non-PRESENT Item cannot have a ValueBinding")
        if len(self.state.items) + 1 > self.limits.max_items:
            raise CommitError("commit exceeds max_items")
        self.state.items[item] = record
        if binding is not None:
            self.state.values[item] = binding
        if control is not None:
            if outcome is not ItemOutcome.PRESENT:
                raise CommitError("control manifest requires PRESENT Item")
            self._mask_values[item] = bool(control)
        self._receipts.append(item)

    def _accept_call_input(self, edge: CallConsumer, item: ItemRef) -> None:
        call = self.program.call(edge.call)
        if item.entity.domain != call.execution_domain:
            raise CommitError("Call input receipt has the wrong Domain")
        grain = GrainRef(edge.call, item.entity)
        if grain in self.state.grains:
            return
        pending = self.state.pending.setdefault(
            grain,
            PendingInvocation([None] * len(call.inputs)),
        )
        current = pending.slots[edge.input_index]
        if current is not None and current != item:
            raise CommitError("Call input slot received conflicting Items")
        pending.slots[edge.input_index] = item
        if any(slot is None for slot in pending.slots):
            return

        inputs = tuple(slot for slot in pending.slots if slot is not None)
        del self.state.pending[grain]
        self._classify_call(grain, inputs)

    def _classify_call(self, grain: GrainRef, inputs: tuple[ItemRef, ...]) -> None:
        if len(self.state.grains) + 1 > self.limits.max_grains:
            raise CommitError("commit exceeds max_grains")
        call = self.program.call(grain.call)
        receipts = tuple(self.state.items[item] for item in inputs)

        # driving input 被丢弃时，本 Grain 与输出同步 DROPPED；其他 REQUIRED
        # 输入异常则产生 SUPPRESSED，二者不能合并成同一种传播语义。
        driving = receipts[call.driving_input]

        if driving.outcome is ItemOutcome.DROPPED:
            self.state.grains[grain] = GrainRecord(
                grain,
                inputs,
                GrainPhase.SEALED,
                GrainOutcome.SUPPRESSED,
            )
            self._publish_call_outputs(grain, ItemOutcome.DROPPED, driving.cause)
            return

        suppress_cause = None
        for input_spec, receipt in zip(call.inputs, receipts):
            if receipt.outcome is ItemOutcome.PRESENT:
                continue
            if (
                input_spec.mode is InputMode.OPTIONAL
                and receipt.outcome is ItemOutcome.DROPPED
            ):
                continue
            suppress_cause = receipt.cause or inputs[call.inputs.index(input_spec)]
            break
        if suppress_cause is not None:
            self.state.grains[grain] = GrainRecord(
                grain,
                inputs,
                GrainPhase.SEALED,
                GrainOutcome.SUPPRESSED,
            )
            self._publish_call_outputs(
                grain,
                ItemOutcome.SUPPRESSED,
                suppress_cause,
            )
            return

        self.state.grains[grain] = GrainRecord(
            grain,
            inputs,
            GrainPhase.READY,
        )
        self._ready.append(grain)

    def _publish_call_outputs(
        self,
        grain: GrainRef,
        outcome: ItemOutcome,
        cause: object | None,
    ) -> None:
        outputs = self.program.outputs_by_call[grain.call]
        self._ensure_item_capacity(len(outputs))
        for output in outputs:
            self._publish(ItemRef(output, grain.entity), outcome, cause=cause)
            for child_port in self._expanded_by_source.get(output, ()):
                shape = ShapeKey(self.program.port(child_port).domain, grain.entity)
                state = (
                    ShapeState.DROPPED
                    if outcome is ItemOutcome.DROPPED
                    else ShapeState.FAILED
                )
                self.state.shapes[shape] = ShapeRecord(
                    state,
                    None,
                    self.program.shape_reporters_by_domain[shape.child_domain],
                    cause=cause,
                )
                self._shape_terminal(shape)

    def _advance_view(self, target_port: PortRef, trigger: ItemRef) -> None:
        spec = self.program.port(target_port)
        origin = spec.origin
        if isinstance(origin, FilterOrigin):
            self._try_filter(target_port, trigger.entity)
        elif isinstance(origin, BroadcastOrigin):
            self._propagate_broadcast_source(target_port, trigger)
        elif isinstance(origin, GroupOrigin):
            parent = self._parent_of(trigger.entity)
            if parent is not None:
                self._try_group(target_port, parent)
        elif isinstance(origin, ExpandOrigin):
            # CallOutput expansion 只能由 commit_success 原子发布；第一版
            # 编译器拒绝非 Call source 的 Expand，避免第二条隐式写路径。
            return

    def _try_filter(self, target_port: PortRef, entity: EntityRef) -> None:
        target = ItemRef(target_port, entity)
        if target in self.state.items:
            return
        origin = self.program.port(target_port).origin
        assert isinstance(origin, FilterOrigin)
        source = ItemRef(origin.source_port, entity)
        mask = ItemRef(origin.mask_port, entity)
        if source not in self.state.items or mask not in self.state.items:
            return
        source_record = self.state.items[source]
        mask_record = self.state.items[mask]
        if source_record.outcome is not ItemOutcome.PRESENT:
            self._publish(target, source_record.outcome, cause=source_record.cause)
            return
        if mask_record.outcome is not ItemOutcome.PRESENT:
            self._publish(target, mask_record.outcome, cause=mask_record.cause)
            return
        if mask not in self._mask_values:
            raise CommitError("Filter mask Port requires a worker control manifest")
        if self._mask_values[mask]:
            self._publish(
                target,
                ItemOutcome.PRESENT,
                binding=self.state.values[source],
            )
        else:
            self._publish(target, ItemOutcome.DROPPED, cause=mask)

    def _try_group(self, target_port: PortRef, parent: EntityRef) -> None:
        target = ItemRef(target_port, parent)
        if target in self.state.items:
            return
        spec = self.program.port(target_port)
        origin = spec.origin
        assert isinstance(origin, GroupOrigin)
        child_domain = self.program.port(origin.value_port).domain
        shape_key = ShapeKey(child_domain, parent)
        shape = self.state.shapes.get(shape_key)
        # Reduce 只在 fan-out cardinality 已终态后判断成员，避免空 group
        # 与尚未完成的 group 混淆。
        if shape is None or shape.state is ShapeState.OPEN:
            return
        if shape.state is ShapeState.DROPPED:
            self._publish(target, ItemOutcome.DROPPED, cause=shape.cause)
            return
        if shape.state is ShapeState.FAILED:
            self._publish(target, ItemOutcome.SUPPRESSED, cause=shape.cause or shape_key)
            return

        children = self._children_by_shape[shape_key]
        member_items = tuple(ItemRef(origin.members_port, child) for child in children)
        if any(item not in self.state.items for item in member_items):
            return
        for item in member_items:
            outcome = self.state.items[item].outcome
            if outcome in {ItemOutcome.FAILED, ItemOutcome.SUPPRESSED}:
                self._publish(target, ItemOutcome.SUPPRESSED, cause=item)
                return

        # members Port 决定成员资格和稳定顺序；value Port 只提供 payload，
        # 因而 filter 后 reduce 不需要复制或重写 lineage。
        survivors = tuple(
            child
            for child, member in zip(children, member_items)
            if self.state.items[member].outcome is ItemOutcome.PRESENT
        )
        value_items = tuple(ItemRef(origin.value_port, child) for child in survivors)
        if any(item not in self.state.items for item in value_items):
            return
        for item in value_items:
            if self.state.items[item].outcome is not ItemOutcome.PRESENT:
                self._publish(target, ItemOutcome.SUPPRESSED, cause=item)
                return

        bindings = tuple(self.state.values[item] for item in value_items)
        # 一层 reduce 收集 RowBinding；多层 reduce 则拼接子 GroupShape，
        # 最终仍保持一个规范 CSR shape 和一份扁平叶子引用。
        value_depth = self._group_depth(origin.value_port)
        if not bindings and value_depth > 0:
            group_shape = GroupShape.nest((), child_depth=value_depth)
            flat_items = ()
        elif not bindings or all(isinstance(value, RowBinding) for value in bindings):
            group_shape = GroupShape.one_level(len(bindings))
            flat_items = value_items
        elif all(isinstance(value, GroupBinding) for value in bindings):
            groups = tuple(value for value in bindings if isinstance(value, GroupBinding))
            depth = groups[0].shape.depth if groups else self._group_depth(origin.value_port)
            group_shape = GroupShape.nest(
                tuple(group.shape for group in groups),
                child_depth=depth,
            )
            flat_items = tuple(
                item for group in groups for item in group.flat_items
            )
        else:
            raise CommitError("group values mix scalar and grouped realizations")

        # 先完成 group slot 与 Item 上限检查，再发布唯一结果，保证越界时
        # 不留下半成品 GroupBinding。
        next_group_slots = self._group_slots + len(flat_items)
        if next_group_slots > self.limits.max_group_slots:
            raise CommitError("group realization exceeds max_group_slots")
        self._ensure_item_capacity(1)
        self._group_slots = next_group_slots
        self._publish(
            target,
            ItemOutcome.PRESENT,
            binding=GroupBinding(group_shape, flat_items),
        )

    def _propagate_broadcast_source(
        self,
        target_port: PortRef,
        source_item: ItemRef,
    ) -> None:
        target_domain = self.program.port(target_port).domain
        for entity in self._entities_by_domain.get(target_domain, ()):
            if self._ancestor_entity(entity, source_item.entity.domain) == source_item.entity:
                self._publish_broadcast(target_port, entity)

    def _publish_broadcast(self, target_port: PortRef, entity: EntityRef) -> None:
        target = ItemRef(target_port, entity)
        if target in self.state.items:
            return
        origin = self.program.port(target_port).origin
        assert isinstance(origin, BroadcastOrigin)
        source_domain = self.program.port(origin.source_port).domain
        ancestor = self._ancestor_entity(entity, source_domain)
        if ancestor is None:
            raise CommitError("broadcast target has no source-domain ancestor")
        source = ItemRef(origin.source_port, ancestor)
        if source not in self.state.items:
            return
        record = self.state.items[source]
        binding = (
            self.state.values[source]
            if record.outcome is ItemOutcome.PRESENT
            else None
        )
        self._publish(
            target,
            record.outcome,
            binding=binding,
            cause=record.cause,
            control=self._mask_values.get(source),
        )

    def _create_children(
        self,
        shape: ShapeKey,
        count: int,
    ) -> tuple[EntityRef, ...]:
        children = tuple(
            EntityRef(shape.child_domain, self._pair(shape.parent_entity.value, ordinal))
            for ordinal in range(count)
        )
        for ordinal, child in enumerate(children):
            self.state.entity_lineage[child] = EntityOrigin(
                shape.child_domain,
                shape.parent_entity,
                ordinal,
                shape,
            )
            self._entities_by_domain[shape.child_domain].append(child)
            for broadcast in self._broadcasts_by_target_domain.get(
                shape.child_domain,
                (),
            ):
                self._publish_broadcast(broadcast, child)
        self._children_by_shape[shape] = children
        return children

    def _shape_terminal(self, shape: ShapeKey) -> None:
        for group_port in self._groups_by_child_domain.get(shape.child_domain, ()):
            self._try_group(group_port, shape.parent_entity)

    def _parent_of(self, entity: EntityRef) -> EntityRef | None:
        origin = self.state.entity_lineage.get(entity)
        return None if origin is None else origin.parent_entity

    def _ancestor_entity(
        self,
        entity: EntityRef,
        domain: DomainRef,
    ) -> EntityRef | None:
        cursor = entity
        while cursor.domain != domain:
            origin = self.state.entity_lineage.get(cursor)
            if origin is None:
                return None
            cursor = origin.parent_entity
        return cursor

    def _group_depth(self, port: PortRef) -> int:
        origin = self.program.port(port).origin
        if not isinstance(origin, GroupOrigin):
            return 0
        return 1 + self._group_depth(origin.value_port)

    @staticmethod
    def _pair(left: int, right: int) -> int:
        """Cantor pairing 让 child identity 与异步完成顺序无关。"""

        total = left + right
        return total * (total + 1) // 2 + right


__all__ = ["ArenaEngine", "CommitError"]
