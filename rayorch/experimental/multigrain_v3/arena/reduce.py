"""Hierarchical Reduce state and canonical nested-list shape construction.

The module has no access to Arena tables or Ray. ArenaEngine routes terminal
fanout/leaf facts into ReduceAccumulator and consumes its final GroupShape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..model import GrainId, GroupShape, InvariantError, ItemRecord, ItemRef


class FanoutTerminal(Enum):
    """Terminal classification for one parent/Expand occurrence."""

    SUCCESS = "success"
    DROPPED = "dropped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExpandInstance:
    """One terminal dynamic fanout fact."""

    grain: GrainId | None
    stage: int
    anchor: ItemRef
    terminal: FanoutTerminal
    cardinality: int | None = None
    cause: GrainId | None = None

    @classmethod
    def success(
        cls,
        grain: GrainId,
        stage: int,
        anchor: ItemRef,
        cardinality: int,
    ) -> "ExpandInstance":
        if cardinality < 0:
            raise ValueError("fanout cardinality must be non-negative")
        return cls(grain, stage, anchor, FanoutTerminal.SUCCESS, cardinality)

    @classmethod
    def dropped(
        cls,
        stage: int,
        anchor: ItemRef,
        cause: GrainId | None,
    ) -> "ExpandInstance":
        return cls(None, stage, anchor, FanoutTerminal.DROPPED, cause=cause)

    @classmethod
    def failed(
        cls,
        grain: GrainId,
        stage: int,
        anchor: ItemRef,
    ) -> "ExpandInstance":
        return cls(grain, stage, anchor, FanoutTerminal.FAILED, cause=grain)


@dataclass(slots=True)
class ReduceAccumulator:
    """One anchor's arbitrary-depth GROUP state.

    ``fanouts`` is keyed by ``(depth, parent_ordinal_path)`` and
    ``leaf_groups`` by full ordinal path. A one-level Reduce is simply a
    one-element ``scope_path``.
    """

    stage: int
    anchor: ItemRef
    scope_path: tuple[int, ...]
    fanouts: dict[tuple[int, tuple[int, ...]], ExpandInstance] = field(
        default_factory=dict
    )
    leaf_groups: dict[int, dict[tuple[int, ...], ItemRecord]] = field(
        default_factory=dict
    )
    scalar_inputs: dict[int, ItemRef] = field(default_factory=dict)
    slot_cost: int = 0

    def depth_for(self, expand_stage: int) -> int | None:
        try:
            return self.scope_path.index(expand_stage)
        except ValueError:
            return None

    def settle_fanout(
        self,
        depth: int,
        parent_path: tuple[int, ...],
        instance: ExpandInstance,
    ) -> int:
        """Store a fanout fact and return newly charged metadata slots."""

        key = (depth, parent_path)
        existing = self.fanouts.get(key)
        if existing is not None:
            if existing != instance:
                raise InvariantError("Reduce fanout terminal changed")
            return 0
        self.fanouts[key] = instance
        added = 1 + (
            instance.cardinality
            if instance.terminal is FanoutTerminal.SUCCESS
            else 0
        )
        self.slot_cost += added
        return added

    def settle_leaf(
        self,
        input_index: int,
        ordinal_path: tuple[int, ...],
        record: ItemRecord,
    ) -> int:
        """Store a GROUP leaf receipt and return newly charged slots."""

        group = self.leaf_groups.setdefault(input_index, {})
        existing = group.get(ordinal_path)
        if existing is not None:
            if existing != record:
                raise InvariantError("Reduce leaf terminal changed")
            return 0
        group[ordinal_path] = record
        self.slot_cost += 1
        return 1

    def expected_leaf_paths(self) -> set[tuple[int, ...]] | None:
        structure = self._active_structure()
        return None if structure is None else set(structure[1][-1])

    def ready(
        self,
        *,
        group_indexes: tuple[int, ...],
        scalar_indexes: tuple[int, ...],
    ) -> bool:
        if not all(index in self.scalar_inputs for index in scalar_indexes):
            return False
        expected = self.expected_leaf_paths()
        return expected is not None and all(
            expected.issubset(self.leaf_groups[index])
            for index in group_indexes
        )

    def build_shape(
        self,
        selected: set[tuple[int, ...]] | None = None,
    ) -> tuple[GroupShape, tuple[tuple[int, ...], ...]]:
        """Build canonical offsets and flat leaf path order."""

        structure = self._active_structure()
        if structure is None:
            raise InvariantError("Reduce shape requested before fanouts settle")
        offsets_by_level, paths_by_level = structure
        if selected is None:
            return GroupShape(offsets_by_level), tuple(paths_by_level[-1])

        last_depth = len(offsets_by_level) - 1
        parents = [()] if last_depth == 0 else paths_by_level[last_depth - 1]
        offsets = [0]
        selected_leaves: list[tuple[int, ...]] = []
        for parent_path in parents:
            children = [
                path
                for path in paths_by_level[-1]
                if path[:-1] == parent_path and path in selected
            ]
            selected_leaves.extend(children)
            offsets.append(len(selected_leaves))
        return (
            GroupShape((*offsets_by_level[:-1], tuple(offsets))),
            tuple(selected_leaves),
        )

    def _active_structure(
        self,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[tuple[int, ...], ...], ...],
    ] | None:
        """Return active paths, preserving N=0 and omitting dropped nodes."""

        parent_paths: list[tuple[int, ...]] = [()]
        offsets_by_level: list[tuple[int, ...]] = []
        paths_by_level: list[tuple[tuple[int, ...], ...]] = []
        last_depth = len(self.scope_path) - 1
        for depth in range(len(self.scope_path)):
            offsets = [0]
            next_paths: list[tuple[int, ...]] = []
            for parent_path in parent_paths:
                instance = self.fanouts.get((depth, parent_path))
                if instance is None:
                    return None
                if instance.terminal is FanoutTerminal.FAILED:
                    return None
                if instance.terminal is FanoutTerminal.DROPPED:
                    offsets.append(len(next_paths))
                    continue
                assert instance.cardinality is not None
                for ordinal in range(instance.cardinality):
                    child_path = (*parent_path, ordinal)
                    if depth < last_depth:
                        child = self.fanouts.get((depth + 1, child_path))
                        if child is None:
                            return None
                        if child.terminal is FanoutTerminal.FAILED:
                            return None
                        if child.terminal is FanoutTerminal.DROPPED:
                            continue
                    next_paths.append(child_path)
                offsets.append(len(next_paths))
            offsets_by_level.append(tuple(offsets))
            paths_by_level.append(tuple(next_paths))
            parent_paths = next_paths
        return tuple(offsets_by_level), tuple(paths_by_level)

