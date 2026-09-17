"""Single-writer, event-driven semantic state machine for one input batch.

The engine is the sole writer of RuntimeState, entity indexes, and the fact
queue. Pure outcome and phase decisions live in ``transitions.py``; physical
Grain queues live in ``dispatch.py``. This module applies compiled Effects to
canonical facts and never reinterprets logical origins.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import AbstractSet, Iterator, Mapping, TypeAlias, assert_never

from .._model import (
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
from .._program.plan import (
    BroadcastEffect,
    CallInputEffect,
    FilterEffect,
    ReduceEffect,
    ItemEffect,
    RuntimePlan,
)
from .._protocol import (
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
from .dispatch import ExecutionMicrobatch, DispatchState, GrainSnapshot
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


@dataclass(slots=True)
class _ReduceProgress:
    """Rebuildable summary of immutable facts for one unfinished Reduce."""

    pending_members: int
    first_failed_member: int | None = None
    next_value_index: int = 0

    def accept_member(self, ordinal: int, outcome: ItemOutcome) -> None:
        self.pending_members -= 1
        if outcome in {ItemOutcome.FAILED, ItemOutcome.SUPPRESSED}:
            if self.first_failed_member is None or ordinal < self.first_failed_member:
                self.first_failed_member = ordinal


class _SuppressionBarrierIndex:
    """InputBatch-local index of monotonic ``(Call, parent anchor)`` barriers."""

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


class InputBatchEngine:
    """Own every semantic fact and mutation for one source input batch.

    A useful reading order is source admission and ``advance``, Worker report
    commit, the publication gateways, and finally the structural Effect
    interpreters.
    """

    def __init__(self, plan: RuntimePlan) -> None:
        self.plan = plan
        self._state = RuntimeState()
        self._dispatch = DispatchState()
        self._suppression_barriers = _SuppressionBarrierIndex()
        self._fact_queue: deque[_FactEvent] = deque()
        self._reduce_progress: dict[ItemRef, _ReduceProgress] = {}
        self._entities_by_domain: dict[
            DomainRef, dict[EntityRef, None]
        ] = defaultdict(dict)
        self._admission_closed = False

    # ── Read-only projections for Executor and materialization ──────────

    @property
    def ready_count(self) -> int:
        """Return the number of Grains visible to the executor."""

        return self._dispatch.ready_count

    @property
    def entity_count(self) -> int:
        """Return the number of Entities created in this input batch."""

        return sum(len(entities) for entities in self._entities_by_domain.values())

    @property
    def item_count(self) -> int:
        """Return the number of terminal Items in this input batch."""

        return len(self._state.items)

    @property
    def expansion_count(self) -> int:
        """Return the number of terminal Expansions in this input batch."""

        return len(self._state.expansions)

    @property
    def grain_count(self) -> int:
        """Return the number of Grains owned by DispatchState."""

        return self._dispatch.grain_count

    def grain_snapshot(self, grain: GrainRef) -> GrainSnapshot:
        """Return an immutable physical-state snapshot for one Grain."""

        return self._dispatch.snapshot(grain)

    def grain_snapshots(self) -> Mapping[GrainRef, GrainSnapshot]:
        """Return an immutable point-in-time snapshot of every Grain."""

        return self._dispatch.snapshots()

    def dispatch_priority(self, call: CallRef) -> int | None:
        """Return immediate-retry/ready/deferred-recovery priority, if runnable."""

        return self._dispatch.priority(call)

    def reserve_dispatch(
        self,
        call: CallRef,
        *,
        max_size: int,
    ) -> ExecutionMicrobatch | None:
        """Reserve visible READY work and publish lazy parent suppression."""

        batch, suppressed = self._dispatch.reserve_with_barriers(
            call,
            max_size=max_size,
            barriered_anchors=self._suppression_barriers.anchors_for(call),
        )
        if suppressed:
            self._publish_parent_suppression(suppressed)
            self.advance()
        if batch is None and not suppressed:
            raise LookupError(f"no READY dispatch for {call!r}")
        return batch

    def close_admission(self) -> None:
        """Declare that this input batch will accept no more source rows."""

        self._admission_closed = True

    def is_complete(self) -> bool:
        """Audit the full completion contract, not merely an empty READY queue."""

        if not self._admission_closed:
            return False
        if (
            self._fact_queue
            or self._state.pending_grains
            or not self._dispatch.queues_empty
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
        """Return created Entities in a Domain for diagnostics."""

        return tuple(self._entities_by_domain.get(domain, ()))

    def grain_invocation(self, grain: GrainRef) -> GrainInvocation:
        """Project semantic facts into the physical inputs consumed by a Worker."""

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
        """Return a Port's Items in stable source/ordinal order."""

        domain = self.plan.port_domain(port)
        entities = sorted(
            self._entities_by_domain.get(domain, ()),
            key=self.entity_coordinate,
        )
        return tuple(ItemRef(port, entity) for entity in entities)

    def item_outcome(self, item: ItemRef) -> ItemOutcome:
        """Read the outcome of a terminal Item."""

        return self._state.items[item].outcome

    def item_cause(self, item: ItemRef) -> object | None:
        """Follow the selected provenance to its cause without mutating facts."""

        cause = self._state.items[item].cause
        seen: set[ItemRef | ExpansionRef] = set()
        while isinstance(cause, (ItemRef, ExpansionRef)):
            if cause in seen:
                raise CommitError("cyclic result cause")
            seen.add(cause)
            record = (
                self._state.items[cause]
                if isinstance(cause, ItemRef)
                else self._state.expansions[cause]
            )
            cause = record.cause
        return cause

    def value_binding(self, item: ItemRef) -> ValueBinding:
        """Read a PRESENT Item's binding without exposing mutation authority."""

        return self._state.values[item]

    def nested_group_rows(
        self,
        binding: NestedGroupBinding,
    ) -> tuple[RowBinding, ...]:
        """Resolve canonical nested-group leaves to rows without reading payloads."""

        rows = tuple(self._state.values[leaf] for leaf in binding.flat_items)
        if not all(isinstance(row, RowBinding) for row in rows):
            raise CommitError("canonical nested-group leaves must resolve to rows")
        return tuple(row for row in rows if isinstance(row, RowBinding))

    def entity_coordinate(self, entity: EntityRef) -> tuple[int, ...]:
        """Return the stable root/ordinal coordinate encoded by explicit lineage."""

        path = []
        cursor = entity
        while cursor in self._state.entity_lineage:
            origin = self._state.entity_lineage[cursor]
            path.append(origin.ordinal)
            cursor = origin.parent_entity
        return (cursor.value, *reversed(path))

    def release_values(self) -> int:
        """Release physical bindings after completion while retaining semantics."""

        if not self.is_complete():
            raise CommitError("cannot release values before input batch completion")
        released = len(self._state.values)
        self._state.values.clear()
        return released

    def progress_summary(self) -> str:
        """Return a deadlock diagnostic that contains no business payload."""

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
        """Atomically admit row-aligned source bindings and control manifests."""

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
        """Consume the closed FactEvent union to a local fixed point."""

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
        """Exhaustively interpret a compiled Effect triggered by Item publication."""

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
        execution_microbatch: ExecutionMicrobatch,
        reports: tuple[WorkerReport, ...],
    ) -> None:
        """Preflight all reports for one ExecutionMicrobatch before ordered publication."""

        if len(reports) != len(execution_microbatch.grains):
            raise CommitError("Worker reports must exactly cover the dispatch batch")
        by_grain: dict[GrainRef, WorkerReport] = {}
        for report in reports:
            if not isinstance(report, (GrainReport, GrainFailureReport)):
                raise CommitError("unsupported Worker report")
            if report.grain in by_grain:
                raise CommitError("duplicate Grain report")
            by_grain[report.grain] = report
        if set(by_grain) != set(execution_microbatch.grains):
            raise CommitError("Worker reports must exactly cover the dispatch batch")

        ordered = tuple(by_grain[grain] for grain in execution_microbatch.grains)
        for report in ordered:
            self._dispatch.validate_in_flight(report.grain, report.generation)

        # Discover every new barrier before preparing any success. Stable
        # ExecutionMicrobatch order, not report tuple order, chooses the canonical cause.
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
        execution_microbatch: ExecutionMicrobatch,
        policy: RecoveryPolicy,
        cause: object,
    ) -> int | None:
        """Partition once, decide, and apply recovery to the exact live subset.

        Return requeued Grain count, or None to abort without changing state.
        An entirely barriered batch completes without consulting the UDF policy.
        """

        live, barriered = self._partition_in_flight(execution_microbatch)
        if live is None:
            self._seal_and_publish_suppressed(barriered)
            self.advance()
            return 0
        action = policy.decide_udf(
            completed_retries=live.udf_retries,
            grain_count=len(live.grains),
        )
        if action is RecoveryAction.ABORT:
            return None

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

    def retry_infrastructure_dispatch(
        self,
        execution_microbatch: ExecutionMicrobatch,
        policy: RecoveryPolicy,
    ) -> int | None:
        """Return retried live Grains, or None when the live budget is exhausted."""

        live, barriered = self._partition_in_flight(execution_microbatch)
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
        execution_microbatch: ExecutionMicrobatch,
    ) -> tuple[ExecutionMicrobatch | None, tuple[GrainRef, ...]]:
        return self._dispatch.partition_in_flight(
            execution_microbatch,
            self._suppression_barriers.anchors_for(execution_microbatch.grains[0].call),
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
        # Summaries must see the whole committed batch before its events run.
        # Idempotent replays return above, so each member is counted only once.
        self._update_reduce_progress(item, outcome)
        self._fact_queue.append(item)

    def _publish_expansion(
        self,
        expansion: ExpansionRef,
        outcome: ExpansionOutcome,
        *,
        children: tuple[EntityRef, ...] | None = None,
        cause: object | None = None,
    ) -> None:
        """Publish one terminal Expansion and enqueue its sole fact event."""

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
        """Classify a Grain from input facts and report whether it left WAITING."""

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

    def _update_reduce_progress(self, item: ItemRef, outcome: ItemOutcome) -> None:
        origin = self._state.entity_lineage.get(item.entity)
        if origin is None:
            return
        for effect in self.plan.item_effects_by_source.get(item.port, ()):
            if isinstance(effect, ReduceEffect) and item.port == effect.members_port:
                target = ItemRef(effect.target_port, origin.parent_entity)
                progress = self._reduce_progress.get(target)
                if progress is not None:
                    progress.accept_member(origin.ordinal, outcome)

    def _reduce_values(
        self,
        effect: ReduceEffect,
        children: tuple[EntityRef, ...],
        progress: _ReduceProgress,
    ) -> Iterator[tuple[int, ItemOutcome | None]]:
        """Resume at the first unchecked value; never revisit a settled prefix.

        The transition consumes this iterator only after membership settles,
        and stops at the first missing or unsuccessful survivor value.
        """

        while progress.next_value_index < len(children):
            index = progress.next_value_index
            child = children[index]
            member = self._state.items[ItemRef(effect.members_port, child)]
            if member.outcome is ItemOutcome.PRESENT:
                value = self._state.items.get(ItemRef(effect.value_port, child))
                yield index, None if value is None else value.outcome
            progress.next_value_index += 1

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
        progress = self._reduce_progress.get(target)
        if progress is None:
            progress = _ReduceProgress(len(children))
            for index, child in enumerate(children):
                member = self._state.items.get(ItemRef(effect.members_port, child))
                if member is not None:
                    progress.accept_member(index, member.outcome)
            self._reduce_progress[target] = progress
        decision = reduce_transition(
            expansion.outcome,
            pending_members=progress.pending_members,
            first_failed_member=progress.first_failed_member,
            values=self._reduce_values(effect, children, progress),
        )
        if decision.outcome is None:
            return
        del self._reduce_progress[target]
        if decision.outcome is not ItemOutcome.PRESENT:
            cause: object = (
                expansion_ref if expansion.cause is None else expansion.cause
            )
            if decision.cause is ReduceCause.MEMBER:
                assert decision.cause_index is not None
                cause = ItemRef(effect.members_port, children[decision.cause_index])
            elif decision.cause is ReduceCause.VALUE:
                assert decision.cause_index is not None
                cause = ItemRef(effect.value_port, children[decision.cause_index])
            self._publish_item(target, decision.outcome, cause=cause)
            return

        # Build the output once; pending groups retain no per-child copy.
        survivor_items = tuple(
            ItemRef(effect.value_port, child)
            for child in children
            if self._state.items[ItemRef(effect.members_port, child)].outcome
            is ItemOutcome.PRESENT
        )

        bindings = tuple(self._state.values[item] for item in survivor_items)
        # The plan supplies the same shape contract for empty and nonempty groups.
        if effect.value_depth > 0:
            if not all(isinstance(value, NestedGroupBinding) for value in bindings):
                raise CommitError("Reduce expected grouped values from its compiled shape")
            groups = tuple(value for value in bindings if isinstance(value, NestedGroupBinding))
            group_layout = NestedGroupLayout.nest(
                tuple(group.layout for group in groups),
                child_depth=effect.value_depth,
            )
            flat_items = tuple(
                item for group in groups for item in group.flat_items
            )
        else:
            if not all(isinstance(value, RowBinding) for value in bindings):
                raise CommitError("Reduce expected row values from its compiled shape")
            group_layout = NestedGroupLayout.one_level(len(bindings))
            flat_items = survivor_items

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
        for entity in self._broadcast_descendants(effect, source_item.entity):
            self._try_broadcast_to_entity(effect, entity)

    def _broadcast_descendants(
        self,
        effect: BroadcastEffect,
        source_entity: EntityRef,
    ) -> Iterator[EntityRef]:
        """Walk existing Expansion facts with O(depth) auxiliary space.

        Keep one child iterator per level, never a collection of all descendants.
        Missing expansions need no waiter: later target Entity events read the
        already-published source. Worker commits publish their complete expansion
        and lineage facts before the event loop runs.
        """

        path = []
        domain = effect.target_domain
        while domain != effect.source_domain:
            path.append(domain)
            parent = self.plan.domain(domain).parent
            if parent is None:
                raise CommitError("broadcast target Domain has no source ancestor")
            domain = parent
        path.reverse()

        stack = [iter((source_entity,))]
        while stack:
            entity = next(stack[-1], None)
            if entity is None:
                stack.pop()
                continue
            depth = len(stack) - 1
            if depth == len(path):
                yield entity
                continue
            expansion = self._state.expansions.get(ExpansionRef(path[depth], entity))
            if expansion is not None and expansion.children:
                stack.append(iter(expansion.children))

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
        """Publish an Entity once; roots have no origin and children record lineage."""

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
        """Use Cantor pairing so child identity is independent of completion order."""

        total = left + right
        return total * (total + 1) // 2 + right


__all__ = ["InputBatchEngine", "CommitError"]
