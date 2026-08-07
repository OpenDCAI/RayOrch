"""Multigrain v3.5 的单写者、事件驱动 Arena 语义状态机。"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Mapping, TypeAlias, assert_never

from ..model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainRef,
    InputMode,
    ItemOutcome,
    ItemRef,
    PortRef,
    ShapeState,
)
from ..transitions import (
    CallAction,
    FilterCause,
    GroupCause,
    InvalidTransition,
    broadcast_transition,
    call_transition,
    expansion_shape_transition,
    filter_transition,
    group_transition,
    item_transition,
    shape_transition,
)
from ..plan import (
    BroadcastEffect,
    CallInputEffect,
    FilterEffect,
    GroupEffect,
    ItemEffect,
    RuntimePlan,
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
from ..recovery import RecoveryAction, RecoveryPolicy
from .dispatch import DispatchSelection, DispatchState, GrainSnapshot
from .state import (
    CommitError,
    EntityOrigin,
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
    controls: tuple[bool, ...] | None


_FactEvent: TypeAlias = ItemRef | ShapeKey | EntityRef


class ArenaEngine:
    """拥有一个 source microbatch 的全部语义事实与唯一写入口。"""

    def __init__(
        self,
        plan: RuntimePlan,
    ) -> None:
        self.plan = plan
        self.state = RuntimeState()
        self._dispatch = DispatchState()
        self._facts: deque[_FactEvent] = deque()
        self._entities_by_domain: dict[
            DomainRef, dict[EntityRef, None]
        ] = defaultdict(dict)
        self._admission_closed = False

    @property
    def ready_count(self) -> int:
        """返回当前可被执行器预留的 Grain 数。"""

        return self._dispatch.ready_count

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

    @property
    def grain_count(self) -> int:
        """返回 DispatchState 独占的 Grain 总数。"""

        return self._dispatch.grain_count

    def grain_snapshot(self, grain: GrainRef) -> GrainSnapshot:
        """返回一个 Grain 的不可变物理状态副本。"""

        return self._dispatch.snapshot(grain)

    def grain_snapshots(self) -> Mapping[GrainRef, GrainSnapshot]:
        """返回当前全部 Grain 的不可变 point-in-time snapshot。"""

        return self._dispatch.snapshots()

    def dispatch_priority(self, call: CallRef) -> int | None:
        """Return immediate/normal/tail priority for one Call, if runnable."""

        return self._dispatch.priority(call)

    def reserve_dispatch(
        self,
        call: CallRef,
        *,
        max_size: int,
        parent_bound: bool = False,
    ) -> DispatchSelection:
        """Reserve immediate recovery, normal work, then deferred recovery."""

        selection = self._dispatch.reserve(
            call,
            max_size=max_size,
            parent_bound=parent_bound,
        )
        if selection is None:
            raise LookupError(f"no READY dispatch for {call!r}")
        return selection

    def close_admission(self) -> None:
        """声明本 Arena 不再接纳 source，允许完整性判定成立。"""

        self._admission_closed = True

    def is_complete(self) -> bool:
        """按完整合同审计 Arena，而非仅检查 ready queue。"""

        if not self._admission_closed:
            return False
        if (
            self._facts
            or self.state.pending
            or not self._dispatch.is_idle
        ):
            return False
        if not self._dispatch.all_sealed:
            return False
        for port in self._output_ports(self.plan.output_tree):
            for entity in self._entities_by_domain.get(self.plan.port_domain(port), ()):
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

        call = self.plan.call(grain.call)
        takes = []
        for input_spec in call.ordered_inputs:
            item = ItemRef(input_spec.port, grain.entity)
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
        return InvocationPlan(
            grain,
            self._dispatch.generation(grain),
            tuple(takes),
        )

    def ordered_items(self, port: PortRef) -> tuple[ItemRef, ...]:
        """按 source/ordinal coordinate 返回一个 Port 的全部 Item。"""

        domain = self.plan.port_domain(port)
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

    def progress_summary(self) -> str:
        """返回不含业务 payload 的死锁诊断摘要。"""

        return (
            f"pending={len(self.state.pending)}, "
            f"ready={self.ready_count}, "
            f"grains={self.grain_count}, "
            f"shapes={len(self.state.shapes)}"
        )

    def admit_sources(
        self,
        bindings: Mapping[PortRef, tuple[RowBinding, ...]],
        *,
        controls: Mapping[PortRef, tuple[bool, ...]] | None = None,
    ) -> tuple[EntityRef, ...]:
        """原子接纳行对齐 source bindings 与所需 control manifests。"""

        if set(bindings) != set(self.plan.source_ports):
            raise CommitError("source bindings must exactly match Program.source_ports")
        counts = {len(rows) for rows in bindings.values()}
        if len(counts) != 1:
            raise CommitError("row-aligned sources must have equal cardinality")
        count = next(iter(counts))
        controls = {} if controls is None else controls
        demanded = set(self.plan.source_ports).intersection(
            self.plan.control_ports
        )
        if set(controls) != demanded:
            raise CommitError("source controls must exactly match control demand")
        if any(len(values) != count for values in controls.values()):
            raise CommitError("source controls must be row-aligned")
        if any(type(value) is not bool for values in controls.values() for value in values):
            raise CommitError("source control manifest must contain bool values")
        root_domain = self.plan.port_domain(self.plan.source_ports[0])
        if len(self._entities_by_domain[root_domain]):
            raise CommitError("sources have already been admitted")

        entities = tuple(EntityRef(root_domain, index) for index in range(count))
        for entity in entities:
            self._publish_entity(entity)
        for source, rows in bindings.items():
            for entity, row in zip(entities, rows):
                self._publish_item(
                    ItemRef(source, entity),
                    ItemOutcome.PRESENT,
                    binding=row,
                    control=(controls[source][entity.value] if source in controls else None),
                )
        self.advance()
        return entities

    def advance(self) -> None:
        """消费封闭 FactEvent 联合，直到结构传播达到局部不动点。"""

        while self._facts:
            fact = self._facts.popleft()
            match fact:
                case ItemRef():
                    for effect in self.plan.effects_by_item_port.get(
                        fact.port, ()
                    ):
                        self._apply_item_effect(effect, fact)
                case ShapeKey():
                    for effect in self.plan.effects_by_shape_domain.get(
                        fact.child_domain, ()
                    ):
                        self._try_group(effect, fact.parent_entity)
                case EntityRef():
                    for effect in self.plan.effects_by_entity_domain.get(
                        fact.domain, ()
                    ):
                        self._try_broadcast_to_entity(effect, fact)
                case _:
                    assert_never(fact)

    def _apply_item_effect(self, effect: ItemEffect, item: ItemRef) -> None:
        """穷尽解释一个由 Item publication 触发的完整编译期 Effect。"""

        match effect:
            case CallInputEffect():
                self._accept_call_input(effect, item)
            case FilterEffect():
                self._try_filter(effect, item.entity)
            case BroadcastEffect():
                self._try_broadcast_from_source(effect, item)
            case GroupEffect():
                parent = self._parent_of(item.entity)
                if parent is not None:
                    self._try_group(effect, parent)
            case _:
                assert_never(effect)

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
        self._dispatch.validate_in_flight(grain, report.generation)

        expected_outputs = self.plan.outputs_by_call[grain.call]
        by_port = {output.port: output for output in report.outputs}
        if len(by_port) != len(report.outputs) or set(by_port) != set(expected_outputs):
            raise CommitError("report outputs must exactly match Call outputs")

        scalar_commits: list[tuple[ItemRef, OutputReport]] = []
        expansion_commits: list[_ExpansionCommit] = []
        counts_by_shape: dict[ShapeKey, list[int]] = defaultdict(list)
        reporters_by_shape: dict[ShapeKey, set[PortRef]] = defaultdict(set)

        for output_port in expected_outputs:
            output = by_port[output_port]
            expansion_rules = self.plan.expansions_by_source.get(output_port, ())
            expanded_ports = tuple(rule.port for rule in expansion_rules)
            rules_by_port = {rule.port: rule for rule in expansion_rules}
            expanded_by_port = {item.port: item for item in output.expansions}
            if len(expanded_by_port) != len(output.expansions):
                raise CommitError("duplicate expanded output report")
            if set(expanded_by_port) != set(expanded_ports):
                raise CommitError("expanded reports do not match Port demand")
            if expanded_ports and output.scalar is not None:
                raise CommitError("expanded Call output must not duplicate scalar payload")
            if not expanded_ports and output.scalar is None:
                raise CommitError("non-expanded Call output requires one scalar binding")
            scalar_control_required = (
                not expanded_ports and output_port in self.plan.control_ports
            )
            if scalar_control_required != (output.control is not None):
                raise CommitError("scalar control manifest does not match Port demand")
            if output.control is not None and type(output.control) is not bool:
                raise CommitError("control manifest must be bool")

            parent_item = ItemRef(output_port, grain.entity)
            scalar_commits.append((parent_item, output))
            for child_port in expanded_ports:
                rows = expanded_by_port[child_port].rows
                controls = expanded_by_port[child_port].controls
                rule = rules_by_port[child_port]
                control_required = rule.control_required
                if control_required != (controls is not None):
                    raise CommitError(
                        "expanded control manifest does not match Port demand"
                    )
                if controls is not None:
                    if len(controls) != len(rows):
                        raise CommitError("expanded controls must align with rows")
                    if any(type(value) is not bool for value in controls):
                        raise CommitError("expanded control manifest must contain bool")
                child_domain = rule.child_domain
                shape = ShapeKey(child_domain, grain.entity)
                counts_by_shape[shape].append(len(rows))
                reporters_by_shape[shape].add(output_port)
                expansion_commits.append(
                    _ExpansionCommit(
                        output_port,
                        child_port,
                        child_domain,
                        rows,
                        controls,
                    )
                )

        for shape, counts in counts_by_shape.items():
            if len(set(counts)) != 1:
                raise CommitError("aligned expansion cardinality mismatch")
            expected = set(self.plan.shape_reporters_by_domain[shape.child_domain])
            if reporters_by_shape[shape] != expected:
                raise CommitError("aligned expansion reporters are incomplete")
            if shape in self.state.shapes:
                raise CommitError("Shape has already been published")
        # 至此所有 report/shape/cardinality 均已验证；后续 publication
        # 对状态机而言是一个不可分割的逻辑 turn。
        self._dispatch.seal(grain, report.generation)

        commits_by_shape: dict[ShapeKey, list[_ExpansionCommit]] = defaultdict(list)
        for commit in expansion_commits:
            commits_by_shape[ShapeKey(commit.child_domain, grain.entity)].append(commit)

        for shape, commits in commits_by_shape.items():
            count = len(commits[0].rows)
            children = self._create_children(shape, count)
            self._publish_shape(shape, ShapeState.SUCCEEDED, children=children)
            for commit in commits:
                leaves = []
                controls = commit.controls or (None,) * len(commit.rows)
                for child, row, control in zip(children, commit.rows, controls):
                    item = ItemRef(commit.child_port, child)
                    leaves.append(item)
                    self._publish_item(
                        item,
                        ItemOutcome.PRESENT,
                        binding=row,
                        control=control,
                    )
                parent_item = ItemRef(commit.source_port, grain.entity)
                self._publish_item(
                    parent_item,
                    ItemOutcome.PRESENT,
                    binding=GroupBinding(
                        GroupShape.one_level(count),
                        tuple(leaves),
                    ),
                )

        expanded_sources = {commit.source_port for commit in expansion_commits}
        for item, output in scalar_commits:
            if item.port in expanded_sources:
                continue
            self._publish_item(
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

        self._dispatch.validate_in_flight(grain, generation)

        outputs = self.plan.outputs_by_call[grain.call]
        self._dispatch.seal(grain, generation)
        for output in outputs:
            self._publish_item(
                ItemRef(output, grain.entity),
                ItemOutcome.FAILED,
                cause=cause,
            )
            for expansion in self.plan.expansions_by_source.get(output, ()):
                shape = ShapeKey(expansion.child_domain, grain.entity)
                if shape not in self.state.shapes:
                    self._publish_shape(
                        shape,
                        ShapeState.FAILED,
                        cause=cause,
                    )
        self.advance()

    def apply_udf_recovery(
        self,
        selection: DispatchSelection,
        action: RecoveryAction,
        cause: object,
    ) -> int:
        """Apply one policy action at the physical/semantic ownership boundary."""

        match action:
            case RecoveryAction.FAIL_SINGLETON:
                if len(selection.grains) != 1:
                    raise CommitError("FAIL_SINGLETON requires one Grain")
                self.commit_failure(selection.grains[0], cause)
                return 0
            case (
                RecoveryAction.RETRY_IMMEDIATE
                | RecoveryAction.RETRY_TAIL
                | RecoveryAction.SPLIT_TAIL
            ):
                return self._dispatch.recover_udf(selection, action)
            case RecoveryAction.ABORT:
                raise CommitError("ABORT is terminal and cannot mutate an Arena")

    def retry_infrastructure_dispatch(
        self,
        selection: DispatchSelection,
        policy: RecoveryPolicy,
    ) -> bool:
        """Pure-policy preflight followed by one atomic physical requeue."""

        failures = self._dispatch.infrastructure_failures(selection)
        if not policy.allows_infrastructure_retry(failures):
            return False
        self._dispatch.recover_infrastructure(selection)
        return True

    def _publish_item(
        self,
        item: ItemRef,
        outcome: ItemOutcome,
        *,
        binding: ValueBinding | None = None,
        cause: object | None = None,
        control: bool | None = None,
    ) -> None:
        existing = self.state.items.get(item)
        record = ItemRecord(outcome, cause, control)
        if existing is not None:
            try:
                item_transition(existing.outcome, outcome)
            except InvalidTransition as error:
                raise CommitError(str(error)) from error
            existing_binding = self.state.values.get(item)
            if existing != record or existing_binding != binding:
                raise CommitError(f"conflicting Item publication: {item}")
            return
        if outcome is ItemOutcome.PRESENT and binding is None:
            raise CommitError("PRESENT Item requires a ValueBinding")
        if outcome is not ItemOutcome.PRESENT and binding is not None:
            raise CommitError("non-PRESENT Item cannot have a ValueBinding")
        if control is not None:
            if outcome is not ItemOutcome.PRESENT:
                raise CommitError("control manifest requires PRESENT Item")
            if type(control) is not bool:
                raise CommitError("control manifest must be bool")
        self.state.items[item] = record
        if binding is not None:
            self.state.values[item] = binding
        self._facts.append(item)

    def _publish_shape(
        self,
        shape: ShapeKey,
        state: ShapeState,
        *,
        children: tuple[EntityRef, ...] | None = None,
        cause: object | None = None,
    ) -> None:
        """单调发布一个 Shape 终态，并入队唯一的事实传播入口。"""

        record = ShapeRecord(state, children, cause)
        existing = self.state.shapes.get(shape)
        try:
            shape_transition(None if existing is None else existing.state, state)
        except InvalidTransition as error:
            raise CommitError(str(error)) from error
        if existing is not None:
            if existing != record:
                raise CommitError(f"conflicting Shape publication: {shape}")
            return
        self.state.shapes[shape] = record
        self._facts.append(shape)

    def _accept_call_input(self, effect: CallInputEffect, item: ItemRef) -> None:
        call = self.plan.call(effect.call)
        if item.entity.domain != call.execution_domain:
            raise CommitError("Call input receipt has the wrong Domain")
        grain = GrainRef(effect.call, item.entity)
        if self._dispatch.contains(grain):
            return
        pending = self.state.pending.setdefault(
            grain,
            PendingInvocation([None] * len(call.ordered_inputs)),
        )
        current = pending.slots[effect.input_index]
        if current is not None and current != item:
            raise CommitError("Call input slot received conflicting Items")
        pending.slots[effect.input_index] = item
        if self._classify_call(grain, tuple(pending.slots)):
            del self.state.pending[grain]

    def _classify_call(
        self,
        grain: GrainRef,
        inputs: tuple[ItemRef | None, ...],
    ) -> bool:
        """按纯输入代数封闭 Grain；返回是否已离开 WAITING。"""

        call = self.plan.call(grain.call)
        outcomes = tuple(
            None if item is None else self.state.items[item].outcome
            for item in inputs
        )
        decision = call_transition(
            tuple(input_.mode for input_ in call.ordered_inputs),
            outcomes,
        )
        if decision.action is CallAction.WAIT:
            return False
        if decision.action is CallAction.READY:
            self._dispatch.inputs_ready(grain, self._batch_parent(grain.entity))
            return True

        assert decision.decisive_input is not None
        decisive_item = inputs[decision.decisive_input]
        assert decisive_item is not None
        receipt = self.state.items[decisive_item]
        self._dispatch.inputs_terminal(grain)
        output = (
            ItemOutcome.DROPPED
            if decision.action is CallAction.DROP_OUTPUTS
            else ItemOutcome.SUPPRESSED
        )
        self._publish_call_outputs(
            grain,
            output,
            decisive_item if receipt.cause is None else receipt.cause,
        )
        return True

    def _publish_call_outputs(
        self,
        grain: GrainRef,
        outcome: ItemOutcome,
        cause: object | None,
    ) -> None:
        outputs = self.plan.outputs_by_call[grain.call]
        shapes: list[ShapeKey] = []
        for output in outputs:
            self._publish_item(
                ItemRef(output, grain.entity),
                outcome,
                cause=cause,
            )
            for expansion in self.plan.expansions_by_source.get(output, ()):
                shape = ShapeKey(expansion.child_domain, grain.entity)
                if shape not in shapes:
                    shapes.append(shape)
        state = expansion_shape_transition(outcome)
        if state is ShapeState.SUCCEEDED:
            raise CommitError("PRESENT expansion requires a successful Worker report")
        for shape in shapes:
            self._publish_shape(shape, state, cause=cause)

    def _try_filter(self, effect: FilterEffect, entity: EntityRef) -> None:
        target = ItemRef(effect.target_port, entity)
        if target in self.state.items:
            return
        source = ItemRef(effect.source_port, entity)
        mask = ItemRef(effect.mask_port, entity)
        source_record = self.state.items.get(source)
        mask_record = self.state.items.get(mask)
        try:
            decision = filter_transition(
                None if source_record is None else source_record.outcome,
                None if mask_record is None else mask_record.outcome,
                None if mask_record is None else mask_record.control,
            )
        except InvalidTransition as error:
            raise CommitError(str(error)) from error
        if decision.outcome is None:
            return
        if decision.outcome is ItemOutcome.PRESENT:
            assert source_record is not None
            control = None
            if effect.copy_source_control:
                control = source_record.control
                if control is None:
                    raise CommitError(
                        "filtered control Port requires a source control manifest"
                    )
            self._publish_item(
                target,
                ItemOutcome.PRESENT,
                binding=self.state.values[source],
                control=control,
            )
            return

        cause_item = source if decision.cause is FilterCause.SOURCE else mask
        cause_record = self.state.items.get(cause_item)
        cause = None if cause_record is None else cause_record.cause
        self._publish_item(
            target,
            decision.outcome,
            cause=cause_item if cause is None else cause,
        )

    def _try_group(self, effect: GroupEffect, parent: EntityRef) -> None:
        target = ItemRef(effect.target_port, parent)
        if target in self.state.items:
            return
        shape_key = ShapeKey(effect.child_domain, parent)
        shape = self.state.shapes.get(shape_key)
        if shape is None:
            return
        children = () if shape.children is None else shape.children
        member_items = tuple(
            ItemRef(effect.members_port, child) for child in children
        )
        value_items = tuple(
            ItemRef(effect.value_port, child) for child in children
        )
        decision = group_transition(
            shape.state,
            tuple(
                None if item not in self.state.items else self.state.items[item].outcome
                for item in member_items
            ),
            tuple(
                None if item not in self.state.items else self.state.items[item].outcome
                for item in value_items
            ),
        )
        if decision.outcome is None:
            return
        if decision.outcome is not ItemOutcome.PRESENT:
            cause: object = shape_key if shape.cause is None else shape.cause
            if decision.cause is GroupCause.MEMBER:
                assert decision.cause_index is not None
                cause = member_items[decision.cause_index]
            elif decision.cause is GroupCause.VALUE:
                assert decision.cause_index is not None
                cause = value_items[decision.cause_index]
            self._publish_item(target, decision.outcome, cause=cause)
            return

        # members 决定资格，value 仅为纯状态机选出的 survivors 提供 payload。
        survivor_items = tuple(value_items[index] for index in decision.survivors)

        bindings = tuple(self.state.values[item] for item in survivor_items)
        # 一层 reduce 收集 RowBinding；多层 reduce 则拼接子 GroupShape，
        # 最终仍保持一个规范 CSR shape 和一份扁平叶子引用。
        value_depth = effect.value_depth
        if not bindings and value_depth > 0:
            group_shape = GroupShape.nest((), child_depth=value_depth)
            flat_items = ()
        elif not bindings or all(isinstance(value, RowBinding) for value in bindings):
            group_shape = GroupShape.one_level(len(bindings))
            flat_items = survivor_items
        elif all(isinstance(value, GroupBinding) for value in bindings):
            groups = tuple(value for value in bindings if isinstance(value, GroupBinding))
            depth = groups[0].shape.depth if groups else effect.value_depth
            group_shape = GroupShape.nest(
                tuple(group.shape for group in groups),
                child_depth=depth,
            )
            flat_items = tuple(
                item for group in groups for item in group.flat_items
            )
        else:
            raise CommitError("group values mix scalar and grouped realizations")

        self._publish_item(
            target,
            ItemOutcome.PRESENT,
            binding=GroupBinding(group_shape, flat_items),
        )

    def _try_broadcast_from_source(
        self,
        effect: BroadcastEffect,
        source_item: ItemRef,
    ) -> None:
        for entity in self._entities_by_domain.get(effect.target_domain, ()):
            if (
                self._ancestor_entity(entity, effect.source_domain)
                == source_item.entity
            ):
                self._try_broadcast_to_entity(effect, entity)

    def _try_broadcast_to_entity(
        self,
        effect: BroadcastEffect,
        entity: EntityRef,
    ) -> None:
        target = ItemRef(effect.target_port, entity)
        if target in self.state.items:
            return
        ancestor = self._ancestor_entity(entity, effect.source_domain)
        if ancestor is None:
            raise CommitError("broadcast target has no source-domain ancestor")
        source = ItemRef(effect.source_port, ancestor)
        if source not in self.state.items:
            return
        record = self.state.items[source]
        binding = (
            self.state.values[source]
            if record.outcome is ItemOutcome.PRESENT
            else None
        )
        self._publish_item(
            target,
            broadcast_transition(record.outcome),
            binding=binding,
            cause=record.cause,
            control=record.control if effect.copy_source_control else None,
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
            self._publish_entity(
                child,
                EntityOrigin(shape.parent_entity, ordinal),
            )
        return children

    def _publish_entity(
        self,
        entity: EntityRef,
        origin: EntityOrigin | None = None,
    ) -> None:
        """单调发布 Entity 事实；root 无 origin，child 显式记录 lineage。"""

        entities = self._entities_by_domain[entity.domain]
        if entity in entities:
            if self.state.entity_lineage.get(entity) != origin:
                raise CommitError(f"conflicting Entity publication: {entity}")
            return
        if entity in self.state.entity_lineage:
            raise CommitError(f"conflicting Entity publication: {entity}")
        if origin is not None:
            self.state.entity_lineage[entity] = origin
        entities[entity] = None
        self._facts.append(entity)

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

    @staticmethod
    def _pair(left: int, right: int) -> int:
        """Cantor pairing 让 child identity 与异步完成顺序无关。"""

        total = left + right
        return total * (total + 1) // 2 + right


__all__ = ["ArenaEngine", "CommitError"]
