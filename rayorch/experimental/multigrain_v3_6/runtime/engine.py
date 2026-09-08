"""Multigrain v3.6 的单写者、事件驱动 microbatch 语义状态机。

Engine 是 RuntimeState、Entity 枚举索引和事实 FIFO 的唯一写入口。纯 outcome/phase
决策位于 transitions.py，Grain 物理队列位于 dispatch.py；本模块只把编译后的 Effect
应用到一个 microbatch 的 canonical facts，绝不回读 Logical Origin。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import AbstractSet, Mapping, TypeAlias, assert_never

from ..model import (
    CallRef,
    DomainRef,
    EntityRef,
    GrainRef,
    InputMode,
    ItemOutcome,
    ItemRef,
    PortRef,
    ExpansionOutcome,
)
from .transitions import (
    CallAction,
    FilterCause,
    ReduceCause,
    InvalidTransition,
    broadcast_transition,
    call_transition,
    expansion_outcome_from_item,
    filter_transition,
    reduce_transition,
    item_transition,
    expansion_transition,
)
from ..program.plan import (
    BroadcastEffect,
    CallInputEffect,
    FilterEffect,
    ReduceEffect,
    ItemEffect,
    RuntimePlan,
)
from ..protocol import (
    GrainFailureReport,
    GrainReport,
    NestedGroupInput,
    GrainInvocation,
    MissingInput,
    PortOutputReport,
    RowBinding,
    WorkerReport,
)
from ..recovery import RecoveryAction, RecoveryPolicy
from .dispatch import DispatchBatch, DispatchState, GrainSnapshot
from .state import (
    CommitError,
    EntityParent,
    NestedGroupBinding,
    NestedGroupLayout,
    ItemRecord,
    PendingGrain,
    RuntimeState,
    ExpansionRef,
    ExpansionRecord,
    ValueBinding,
)


@dataclass(frozen=True, slots=True)
class _ExpandedOutputCommit:
    source_port: PortRef
    child_port: PortRef
    child_domain: DomainRef
    rows: tuple[RowBinding, ...]
    controls: tuple[bool, ...] | None


@dataclass(frozen=True, slots=True)
class _PreparedGrainSuccess:
    report: GrainReport
    scalar_commits: tuple[tuple[ItemRef, PortOutputReport], ...]
    expansion_commits: tuple[_ExpandedOutputCommit, ...]


class _SuppressionBarrierIndex:
    """Microbatch-local index of monotonic ``(Call, parent anchor)`` barriers."""

    def __init__(self) -> None:
        self._causes_by_call: dict[
            CallRef,
            dict[EntityRef, object | None],
        ] = {}

    def establish(
        self,
        call: CallRef,
        parent_anchor: EntityRef,
        cause: object | None,
    ) -> None:
        self._causes_by_call.setdefault(call, {}).setdefault(parent_anchor, cause)

    def is_barriered(self, call: CallRef, parent_anchor: EntityRef) -> bool:
        return parent_anchor in self._causes_by_call.get(call, {})

    def cause(self, call: CallRef, parent_anchor: EntityRef) -> object | None:
        return self._causes_by_call[call][parent_anchor]

    def anchors_for(self, call: CallRef) -> AbstractSet[EntityRef]:
        causes = self._causes_by_call.get(call)
        return frozenset() if causes is None else causes.keys()


_FactEvent: TypeAlias = ItemRef | ExpansionRef | EntityRef


class MicrobatchEngine:
    """拥有一个 source microbatch 的全部语义事实与唯一写入口。

    阅读顺序建议是 admission/advance、Worker report commit、三个 publication
    gateway，最后再看 Filter/Reduce/Broadcast 的 Effect interpreter。
    """

    def __init__(
        self,
        plan: RuntimePlan,
        *,
        ready_fifo: bool = True,
    ) -> None:
        self.plan = plan
        self._state = RuntimeState()
        self._dispatch = DispatchState(ready_fifo=ready_fifo)
        self._suppression_barriers = _SuppressionBarrierIndex()
        self._fact_queue: deque[_FactEvent] = deque()
        self._entities_by_domain: dict[
            DomainRef, dict[EntityRef, None]
        ] = defaultdict(dict)
        self._admission_closed = False

    # ── Read-only projections for Executor and materialization ──────────

    @property
    def ready_count(self) -> int:
        """返回当前可被执行器预留的 Grain 数。"""

        return self._dispatch.ready_count

    @property
    def entity_count(self) -> int:
        """返回本 microbatch 已创建的 Entity 总数。"""

        return sum(len(entities) for entities in self._entities_by_domain.values())

    @property
    def item_count(self) -> int:
        """返回本 microbatch 已终态化的 Item 总数。"""

        return len(self._state.items)

    @property
    def expansion_count(self) -> int:
        """返回本 microbatch 已终态化的 Expansion 总数。"""

        return len(self._state.expansions)

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
        """Return immediate-retry/ready/deferred-recovery priority, if runnable."""

        return self._dispatch.priority(call)

    def reserve_dispatch(
        self,
        call: CallRef,
        *,
        max_size: int,
        pack_by_parent: bool = False,
    ) -> DispatchBatch | None:
        """Reserve live work and publish lazy parent suppression in one turn."""

        batch, suppressed = self._dispatch.reserve_with_barriers(
            call,
            max_size=max_size,
            pack_by_parent=pack_by_parent,
            barriered_anchors=self._suppression_barriers.anchors_for(call),
        )
        if suppressed:
            self._publish_parent_suppression(suppressed)
            self.advance()
        if batch is None and not suppressed:
            raise LookupError(f"no READY dispatch for {call!r}")
        return batch

    def close_admission(self) -> None:
        """声明本 microbatch 不再接纳 source。"""

        self._admission_closed = True

    def is_complete(self) -> bool:
        """按完整合同审计 microbatch，而非仅检查 ready queue。"""

        if not self._admission_closed:
            return False
        if (
            self._fact_queue
            or self._state.pending_grains
            or not self._dispatch.is_idle
        ):
            return False
        if not self._dispatch.all_sealed:
            return False
        for port in self._output_ports(self.plan.output_tree):
            for entity in self._entities_by_domain.get(self.plan.port_domain(port), ()):
                if ItemRef(port, entity) not in self._state.items:
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

    def _parent_anchor(self, entity: EntityRef) -> EntityRef:
        """Return the direct parent, or the root Entity itself when parentless."""

        parent = self._parent_of(entity)
        return entity if parent is None else parent

    def entities(self, domain: DomainRef) -> tuple[EntityRef, ...]:
        """返回 Domain 内已经创建的 Entity，顺序只用于审计。"""

        return tuple(self._entities_by_domain.get(domain, ()))

    def grain_invocation(self, grain: GrainRef) -> GrainInvocation:
        """把语义事实投影为 Worker 可消费的纯物理输入计划。"""

        call = self.plan.call(grain.call)
        inputs = []
        for input_spec in call.ordered_inputs:
            item = ItemRef(input_spec.port, grain.entity)
            receipt = self._state.items[item]
            if (
                input_spec.mode is InputMode.OPTIONAL
                and receipt.outcome is ItemOutcome.DROPPED
            ):
                inputs.append(MissingInput())
                continue
            binding = self._state.values[item]
            if isinstance(binding, RowBinding):
                inputs.append(binding)
                continue
            if isinstance(binding, NestedGroupBinding):
                rows = tuple(self._state.values[leaf] for leaf in binding.flat_items)
                if not all(isinstance(row, RowBinding) for row in rows):
                    raise CommitError(
                        "canonical nested-group leaves must resolve to rows"
                    )
                inputs.append(
                    NestedGroupInput(
                        tuple(row for row in rows if isinstance(row, RowBinding)),
                        binding.layout.offsets_by_level,
                    )
                )
                continue
            raise CommitError(f"unsupported ValueBinding: {binding!r}")
        return GrainInvocation(
            grain,
            self._dispatch.generation(grain),
            tuple(inputs),
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

        return self._state.items[item].outcome

    def value_binding(self, item: ItemRef) -> ValueBinding:
        """读取 PRESENT Item 的物理 binding；调用方不得据此改变语义状态。"""

        return self._state.values[item]

    def nested_group_rows(
        self,
        binding: NestedGroupBinding,
    ) -> tuple[RowBinding, ...]:
        """把 canonical nested-group 叶子解析为行引用，不读取业务 payload。"""

        rows = tuple(self._state.values[leaf] for leaf in binding.flat_items)
        if not all(isinstance(row, RowBinding) for row in rows):
            raise CommitError("canonical nested-group leaves must resolve to rows")
        return tuple(row for row in rows if isinstance(row, RowBinding))

    def entity_coordinate(self, entity: EntityRef) -> tuple[int, ...]:
        """沿显式 lineage 计算稳定的 root/ordinal path。"""

        path = []
        cursor = entity
        while cursor in self._state.entity_lineage:
            origin = self._state.entity_lineage[cursor]
            path.append(origin.ordinal)
            cursor = origin.parent_entity
        return (cursor.value, *reversed(path))

    def release_values(self) -> int:
        """完成后释放物理 bindings，同时保留语义计数与 Grain 状态。"""

        if not self.is_complete():
            raise CommitError("cannot release values before microbatch completion")
        released = len(self._state.values)
        self._state.values.clear()
        return released

    def progress_summary(self) -> str:
        """返回不含业务 payload 的死锁诊断摘要。"""

        return (
            f"pending={len(self._state.pending_grains)}, "
            f"ready={self.ready_count}, "
            f"grains={self.grain_count}, "
            f"expansions={len(self._state.expansions)}"
        )

    # ── Source admission and the sole fact fixed-point loop ─────────────

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

        while self._fact_queue:
            fact = self._fact_queue.popleft()
            match fact:
                case ItemRef():
                    for effect in self.plan.item_effects_by_source.get(
                        fact.port, ()
                    ):
                        self._apply_item_effect(effect, fact)
                case ExpansionRef():
                    for effect in self.plan.reduce_effects_by_child_domain.get(
                        fact.child_domain, ()
                    ):
                        self._try_reduce(effect, fact.parent_entity)
                case EntityRef():
                    for effect in self.plan.broadcast_effects_by_target_domain.get(
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
            case ReduceEffect():
                parent = self._parent_of(item.entity)
                if parent is not None:
                    self._try_reduce(effect, parent)
            case _:
                assert_never(effect)

    # ── Worker report preflight and commit boundary ─────────────────────

    def commit_reports(
        self,
        dispatch_batch: DispatchBatch,
        reports: tuple[WorkerReport, ...],
    ) -> None:
        """Preflight all reports for one DispatchBatch before ordered publication."""

        if len(reports) != len(dispatch_batch.grains):
            raise CommitError("Worker reports must exactly cover the dispatch batch")
        by_grain: dict[GrainRef, WorkerReport] = {}
        for report in reports:
            if not isinstance(report, (GrainReport, GrainFailureReport)):
                raise CommitError("unsupported Worker report")
            if report.grain in by_grain:
                raise CommitError("duplicate Grain report")
            by_grain[report.grain] = report
        if set(by_grain) != set(dispatch_batch.grains):
            raise CommitError("Worker reports must exactly cover the dispatch batch")

        ordered = tuple(by_grain[grain] for grain in dispatch_batch.grains)
        for report in ordered:
            self._dispatch.validate_in_flight(report.grain, report.generation)

        # Discover every new barrier before preparing any success. Stable
        # DispatchBatch order, not report tuple order, chooses the canonical cause.
        pending_barriers: dict[tuple[CallRef, EntityRef], object] = {}
        for report in ordered:
            if isinstance(report, GrainFailureReport) and report.suppress_siblings:
                key = (report.grain.call, self._dispatch.parent_anchor(report.grain))
                pending_barriers.setdefault(key, report.cause)

        prepared: dict[GrainRef, _PreparedGrainSuccess] = {}
        for report in ordered:
            if not isinstance(report, GrainReport):
                continue
            key = (report.grain.call, self._dispatch.parent_anchor(report.grain))
            if key in pending_barriers or self._suppression_barriers.is_barriered(*key):
                continue
            prepared[report.grain] = self._prepare_success(report)

        # Mutation frontier: identities and every live success payload are now
        # valid. Explicit failures remain FAILED even when their parent is barriered.
        for (call, parent_anchor), cause in pending_barriers.items():
            self._suppression_barriers.establish(call, parent_anchor, cause)
        for report in ordered:
            if isinstance(report, GrainFailureReport):
                self._apply_failure(
                    report.grain,
                    report.cause,
                    generation=report.generation,
                )
                continue
            if self._is_parent_barriered(report.grain):
                self._apply_suppression(
                    report.grain,
                    self._barrier_cause(report.grain),
                    generation=report.generation,
                )
                continue
            self._apply_success(prepared[report.grain])
        self.advance()

    def _prepare_success(self, report: GrainReport) -> _PreparedGrainSuccess:
        """Validate one success and freeze publication intents without mutation."""

        grain = report.grain
        self._dispatch.validate_in_flight(grain, report.generation)

        # Phase 1: validate every per-output scalar/expanded/control contract
        # and build publication intents without mutating canonical state.
        expected_outputs = self.plan.outputs_by_call[grain.call]
        by_port = {output.port: output for output in report.outputs}
        if len(by_port) != len(report.outputs) or set(by_port) != set(expected_outputs):
            raise CommitError("report outputs must exactly match Call outputs")

        scalar_commits: list[tuple[ItemRef, PortOutputReport]] = []
        expansion_commits: list[_ExpandedOutputCommit] = []
        counts_by_expansion: dict[ExpansionRef, list[int]] = defaultdict(list)
        reporters_by_expansion: dict[ExpansionRef, set[PortRef]] = defaultdict(set)

        for output_port in expected_outputs:
            output = by_port[output_port]
            expansion_rules = self.plan.expand_effects_by_source.get(output_port, ())
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
                expansion = ExpansionRef(child_domain, grain.entity)
                counts_by_expansion[expansion].append(len(rows))
                reporters_by_expansion[expansion].add(output_port)
                expansion_commits.append(
                    _ExpandedOutputCommit(
                        output_port,
                        child_port,
                        child_domain,
                        rows,
                        controls,
                    )
                )

        # Phase 2: aligned outputs share one Expansion, so cardinality and the
        # complete reporter set must agree across all participating outputs.
        for expansion, counts in counts_by_expansion.items():
            if len(set(counts)) != 1:
                raise CommitError("aligned expansion cardinality mismatch")
            expected = set(
                self.plan.expansion_sources_by_domain[expansion.child_domain]
            )
            if reporters_by_expansion[expansion] != expected:
                raise CommitError("aligned expansion reporters are incomplete")
            if expansion in self._state.expansions:
                raise CommitError("Expansion has already been published")
        return _PreparedGrainSuccess(
            report,
            tuple(scalar_commits),
            tuple(expansion_commits),
        )

    def _apply_success(self, prepared: _PreparedGrainSuccess) -> None:
        """Apply one previously validated success without running advance()."""

        report = prepared.report
        grain = report.grain
        scalar_commits = prepared.scalar_commits
        expansion_commits = prepared.expansion_commits
        self._dispatch.seal(grain, report.generation)

        commits_by_expansion: dict[
            ExpansionRef, list[_ExpandedOutputCommit]
        ] = defaultdict(list)
        for commit in expansion_commits:
            commits_by_expansion[
                ExpansionRef(commit.child_domain, grain.entity)
            ].append(commit)

        # Publish Expansion -> child Entity -> child/parent Item in dependency
        # order. The caller owns the single fact-propagation fixed point.
        for expansion, commits in commits_by_expansion.items():
            count = len(commits[0].rows)
            children = self._create_children(expansion, count)
            self._publish_expansion(
                expansion,
                ExpansionOutcome.SUCCEEDED,
                children=children,
            )
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
                    binding=NestedGroupBinding(
                        NestedGroupLayout.one_level(count),
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

    # ── Failure/recovery handoff to the sole DispatchState ──────────────

    def _apply_failure(
        self,
        grain: GrainRef,
        cause: object,
        *,
        generation: int | None = None,
    ) -> None:
        """Seal and publish one explicit failure without running advance()."""

        self._dispatch.seal(grain, generation)
        self._publish_call_outputs(grain, ItemOutcome.FAILED, cause)

    def _apply_suppression(
        self,
        grain: GrainRef,
        cause: object | None,
        *,
        generation: int | None = None,
    ) -> None:
        """Seal one in-flight sibling and publish no successful payload."""

        self._dispatch.seal(grain, generation)
        self._publish_call_outputs(grain, ItemOutcome.SUPPRESSED, cause)

    def _publish_parent_suppression(
        self,
        grains: tuple[GrainRef, ...],
    ) -> None:
        """Publish READY/in-flight grains already sealed by DispatchState."""

        for grain in grains:
            self._publish_call_outputs(
                grain,
                ItemOutcome.SUPPRESSED,
                self._barrier_cause(grain),
            )

    def _is_parent_barriered(self, grain: GrainRef) -> bool:
        return self._suppression_barriers.is_barriered(
            grain.call,
            self._dispatch.parent_anchor(grain),
        )

    def _barrier_cause(self, grain: GrainRef) -> object | None:
        return self._suppression_barriers.cause(
            grain.call,
            self._dispatch.parent_anchor(grain),
        )

    def apply_udf_recovery(
        self,
        dispatch_batch: DispatchBatch,
        action: RecoveryAction,
        cause: object,
    ) -> int:
        """Suppress barriered peers, then apply one action to the exact live subset."""

        live, barriered = self._partition_in_flight(dispatch_batch)
        if action is RecoveryAction.ABORT:
            raise CommitError("ABORT is terminal and cannot mutate a microbatch")
        if live is None:
            self._seal_and_publish_suppressed(barriered)
            self.advance()
            return 0
        if action is RecoveryAction.FAIL_SINGLETON and len(live.grains) != 1:
            raise CommitError("FAIL_SINGLETON requires one live Grain")
        if action is RecoveryAction.SPLIT_TAIL and (
            live.udf_retries == 0 or len(live.grains) <= 1
        ):
            raise CommitError("split requires one failed live recovery batch")

        self._seal_and_publish_suppressed(barriered)

        match action:
            case RecoveryAction.FAIL_SINGLETON:
                self._apply_failure(live.grains[0], cause)
                retried = 0
            case (
                RecoveryAction.RETRY_IMMEDIATE
                | RecoveryAction.RETRY_TAIL
                | RecoveryAction.SPLIT_TAIL
            ):
                retried = self._dispatch.recover_udf(live, action)
            case RecoveryAction.ABORT:
                raise AssertionError("ABORT was rejected before mutation")
        if barriered or action is RecoveryAction.FAIL_SINGLETON:
            self.advance()
        return retried

    def live_recovery_batch(
        self,
        dispatch_batch: DispatchBatch,
    ) -> DispatchBatch | None:
        """Return the current live subset for a pure RecoveryPolicy decision."""

        live, _ = self._partition_in_flight(dispatch_batch)
        return live

    def suppress_barriered_batch(self, dispatch_batch: DispatchBatch) -> None:
        """Suppress an in-flight batch whose every Grain is behind a barrier."""

        live, barriered = self._partition_in_flight(dispatch_batch)
        if live is not None:
            raise CommitError("dispatch still contains live Grains")
        self._seal_and_publish_suppressed(barriered)
        self.advance()

    def retry_infrastructure_dispatch(
        self,
        dispatch_batch: DispatchBatch,
        policy: RecoveryPolicy,
    ) -> int | None:
        """Return retried live Grains, or None when the live budget is exhausted."""

        live, barriered = self._partition_in_flight(dispatch_batch)
        if live is not None:
            failures = self._dispatch.infrastructure_failures(live)
            if not policy.allows_infrastructure_retry(failures):
                return None
            self._seal_and_publish_suppressed(barriered)
        retried = (
            0 if live is None else self._dispatch.recover_infrastructure(live)
        )
        if barriered:
            self.advance()
        return retried

    def _partition_in_flight(
        self,
        dispatch_batch: DispatchBatch,
    ) -> tuple[DispatchBatch | None, tuple[GrainRef, ...]]:
        return self._dispatch.partition_in_flight(
            dispatch_batch,
            self._suppression_barriers.anchors_for(dispatch_batch.grains[0].call),
        )

    def _seal_and_publish_suppressed(
        self,
        barriered: tuple[GrainRef, ...],
    ) -> None:
        self._dispatch.seal_in_flight(barriered)
        self._publish_parent_suppression(barriered)

    # ── Canonical Item and Expansion publication gateways ───────────────

    def _publish_item(
        self,
        item: ItemRef,
        outcome: ItemOutcome,
        *,
        binding: ValueBinding | None = None,
        cause: object | None = None,
        control: bool | None = None,
    ) -> None:
        """Monotonically publish one Item and enqueue its identity once.

        Outcome, binding, cause and control are committed together here; the
        FIFO carries only ItemRef and therefore never becomes a second truth.
        """

        existing = self._state.items.get(item)
        record = ItemRecord(outcome, cause, control)
        if existing is not None:
            try:
                item_transition(existing.outcome, outcome)
            except InvalidTransition as error:
                raise CommitError(str(error)) from error
            existing_binding = self._state.values.get(item)
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
        self._state.items[item] = record
        if binding is not None:
            self._state.values[item] = binding
        self._fact_queue.append(item)

    def _publish_expansion(
        self,
        expansion: ExpansionRef,
        outcome: ExpansionOutcome,
        *,
        children: tuple[EntityRef, ...] | None = None,
        cause: object | None = None,
    ) -> None:
        """单调发布一个 Expansion 终态，并入队唯一的事实传播入口。"""

        record = ExpansionRecord(outcome, children, cause)
        existing = self._state.expansions.get(expansion)
        try:
            expansion_transition(
                None if existing is None else existing.outcome,
                outcome,
            )
        except InvalidTransition as error:
            raise CommitError(str(error)) from error
        if existing is not None:
            if existing != record:
                raise CommitError(
                    f"conflicting Expansion publication: {expansion}"
                )
            return
        self._state.expansions[expansion] = record
        self._fact_queue.append(expansion)

    # ── Call input algebra and structural Effect interpreters ───────────

    def _accept_call_input(self, effect: CallInputEffect, item: ItemRef) -> None:
        call = self.plan.call(effect.call)
        if item.entity.domain != call.execution_domain:
            raise CommitError("Call input receipt has the wrong Domain")
        grain = GrainRef(effect.call, item.entity)
        if self._dispatch.contains(grain):
            return
        pending_grain = self._state.pending_grains.setdefault(
            grain,
            PendingGrain([None] * len(call.ordered_inputs)),
        )
        current = pending_grain.slots[effect.input_index]
        if current is not None and current != item:
            raise CommitError("Call input slot received conflicting Items")
        pending_grain.slots[effect.input_index] = item
        if self._classify_call(grain, tuple(pending_grain.slots)):
            del self._state.pending_grains[grain]

    def _classify_call(
        self,
        grain: GrainRef,
        inputs: tuple[ItemRef | None, ...],
    ) -> bool:
        """按纯输入代数封闭 Grain；返回是否已离开 WAITING。"""

        call = self.plan.call(grain.call)
        parent_anchor = self._parent_anchor(grain.entity)
        if self._suppression_barriers.is_barriered(grain.call, parent_anchor):
            self._dispatch.inputs_terminal(grain)
            self._publish_call_outputs(
                grain,
                ItemOutcome.SUPPRESSED,
                self._suppression_barriers.cause(grain.call, parent_anchor),
            )
            return True
        outcomes = tuple(
            None if item is None else self._state.items[item].outcome
            for item in inputs
        )
        decision = call_transition(
            tuple(input_.mode for input_ in call.ordered_inputs),
            outcomes,
        )
        if decision.action is CallAction.WAIT:
            return False
        if decision.action is CallAction.READY:
            self._dispatch.inputs_ready(grain, self._parent_anchor(grain.entity))
            return True

        assert decision.decisive_input is not None
        decisive_item = inputs[decision.decisive_input]
        assert decisive_item is not None
        receipt = self._state.items[decisive_item]
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
        expansions: list[ExpansionRef] = []
        for output in outputs:
            self._publish_item(
                ItemRef(output, grain.entity),
                outcome,
                cause=cause,
            )
            for expansion in self.plan.expand_effects_by_source.get(output, ()):
                expansion_ref = ExpansionRef(expansion.child_domain, grain.entity)
                if expansion_ref not in expansions:
                    expansions.append(expansion_ref)
        expansion_outcome = expansion_outcome_from_item(outcome)
        if expansion_outcome is ExpansionOutcome.SUCCEEDED:
            raise CommitError("PRESENT expansion requires a successful Worker report")
        for expansion in expansions:
            self._publish_expansion(expansion, expansion_outcome, cause=cause)

    def _try_filter(self, effect: FilterEffect, entity: EntityRef) -> None:
        """Publish Filter once both same-Entity inputs determine a terminal result."""

        target = ItemRef(effect.target_port, entity)
        if target in self._state.items:
            return
        source = ItemRef(effect.source_port, entity)
        mask = ItemRef(effect.mask_port, entity)
        source_record = self._state.items.get(source)
        mask_record = self._state.items.get(mask)
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
                binding=self._state.values[source],
                control=control,
            )
            return

        cause_item = source if decision.cause is FilterCause.SOURCE else mask
        cause_record = self._state.items.get(cause_item)
        cause = None if cause_record is None else cause_record.cause
        self._publish_item(
            target,
            decision.outcome,
            cause=cause_item if cause is None else cause,
        )

    def _try_reduce(self, effect: ReduceEffect, parent: EntityRef) -> None:
        """Restore one parent nested-group value after its facts settle."""

        target = ItemRef(effect.target_port, parent)
        if target in self._state.items:
            return
        expansion_ref = ExpansionRef(effect.child_domain, parent)
        expansion = self._state.expansions.get(expansion_ref)
        if expansion is None:
            return
        children = () if expansion.children is None else expansion.children
        member_items = tuple(
            ItemRef(effect.members_port, child) for child in children
        )
        value_items = tuple(
            ItemRef(effect.value_port, child) for child in children
        )
        decision = reduce_transition(
            expansion.outcome,
            tuple(
                None if item not in self._state.items else self._state.items[item].outcome
                for item in member_items
            ),
            tuple(
                None if item not in self._state.items else self._state.items[item].outcome
                for item in value_items
            ),
        )
        if decision.outcome is None:
            return
        if decision.outcome is not ItemOutcome.PRESENT:
            cause: object = (
                expansion_ref if expansion.cause is None else expansion.cause
            )
            if decision.cause is ReduceCause.MEMBER:
                assert decision.cause_index is not None
                cause = member_items[decision.cause_index]
            elif decision.cause is ReduceCause.VALUE:
                assert decision.cause_index is not None
                cause = value_items[decision.cause_index]
            self._publish_item(target, decision.outcome, cause=cause)
            return

        # members 决定资格，value 仅为纯状态机选出的 survivors 提供 payload。
        survivor_items = tuple(value_items[index] for index in decision.survivors)

        bindings = tuple(self._state.values[item] for item in survivor_items)
        # 一层 reduce 收集 RowBinding；多层 reduce 则拼接子 NestedGroupLayout，
        # 最终仍保持一个规范 CSR layout 和一份扁平叶子引用。
        value_depth = effect.value_depth
        if not bindings and value_depth > 0:
            group_layout = NestedGroupLayout.nest((), child_depth=value_depth)
            flat_items = ()
        elif not bindings or all(isinstance(value, RowBinding) for value in bindings):
            group_layout = NestedGroupLayout.one_level(len(bindings))
            flat_items = survivor_items
        elif all(isinstance(value, NestedGroupBinding) for value in bindings):
            groups = tuple(value for value in bindings if isinstance(value, NestedGroupBinding))
            depth = groups[0].layout.depth if groups else effect.value_depth
            group_layout = NestedGroupLayout.nest(
                tuple(group.layout for group in groups),
                child_depth=depth,
            )
            flat_items = tuple(
                item for group in groups for item in group.flat_items
            )
        else:
            raise CommitError(
                "nested-group values mix scalar and grouped realizations"
            )

        self._publish_item(
            target,
            ItemOutcome.PRESENT,
            binding=NestedGroupBinding(group_layout, flat_items),
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
        if target in self._state.items:
            return
        ancestor = self._ancestor_entity(entity, effect.source_domain)
        if ancestor is None:
            raise CommitError("broadcast target has no source-domain ancestor")
        source = ItemRef(effect.source_port, ancestor)
        if source not in self._state.items:
            return
        record = self._state.items[source]
        binding = (
            self._state.values[source]
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

    # ── Entity creation and lineage navigation ──────────────────────────

    def _create_children(
        self,
        expansion: ExpansionRef,
        count: int,
    ) -> tuple[EntityRef, ...]:
        children = tuple(
            EntityRef(
                expansion.child_domain,
                self._pair(expansion.parent_entity.value, ordinal),
            )
            for ordinal in range(count)
        )
        for ordinal, child in enumerate(children):
            self._publish_entity(
                child,
                EntityParent(expansion.parent_entity, ordinal),
            )
        return children

    def _publish_entity(
        self,
        entity: EntityRef,
        origin: EntityParent | None = None,
    ) -> None:
        """单调发布 Entity 事实；root 无 origin，child 显式记录 lineage。"""

        entities = self._entities_by_domain[entity.domain]
        if entity in entities:
            if self._state.entity_lineage.get(entity) != origin:
                raise CommitError(f"conflicting Entity publication: {entity}")
            return
        if entity in self._state.entity_lineage:
            raise CommitError(f"conflicting Entity publication: {entity}")
        if origin is not None:
            self._state.entity_lineage[entity] = origin
        entities[entity] = None
        self._fact_queue.append(entity)

    def _parent_of(self, entity: EntityRef) -> EntityRef | None:
        origin = self._state.entity_lineage.get(entity)
        return None if origin is None else origin.parent_entity

    def _ancestor_entity(
        self,
        entity: EntityRef,
        domain: DomainRef,
    ) -> EntityRef | None:
        cursor = entity
        while cursor.domain != domain:
            origin = self._state.entity_lineage.get(cursor)
            if origin is None:
                return None
            cursor = origin.parent_entity
        return cursor

    @staticmethod
    def _pair(left: int, right: int) -> int:
        """Cantor pairing 让 child identity 与异步完成顺序无关。"""

        total = left + right
        return total * (total + 1) // 2 + right


__all__ = ["MicrobatchEngine", "CommitError"]
