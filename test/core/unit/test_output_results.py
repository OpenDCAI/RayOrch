"""Public output values and detached, readable failure descriptions."""

from dataclasses import FrozenInstanceError
import pickle

import pytest

import rayorch as ro
from .test_runtime_plan_semantics import run_sync


@pytest.mark.parametrize("outcome", [ro.ItemOutcome.PRESENT, "FAILED", None])
def test_output_issue_rejects_invalid_outcomes(outcome):
    with pytest.raises(ValueError, match="non-PRESENT ItemOutcome"):
        ro.OutputIssue(outcome)


def test_output_issue_is_immutable_and_serializable():
    issue = ro.OutputIssue(ro.ItemOutcome.FAILED, "bad input")
    assert pickle.loads(pickle.dumps(issue)) == issue
    with pytest.raises(FrozenInstanceError):
        issue.cause = "changed"
    with pytest.raises(TypeError, match="text or None"):
        ro.OutputIssue(ro.ItemOutcome.FAILED, {"code": 7})


@pytest.mark.parametrize("cause, expected", [
    (None, None),
    ("", ""),
    ("bad input", "bad input"),
    (17, "17"),
    ({"code": 7}, "{'code': 7}"),
])
def test_failure_cause_survives_views_reduce_and_downstream_call(cause, expected):
    @ro.function
    def fail(values):
        return [ro.RecordFailure(cause) for _ in values]

    @ro.function
    def keep(values):
        return [True for _ in values]

    @ro.function
    def split(values):
        return [list(range(value)) for value in values]

    @ro.function
    def identity(values):
        return values

    class FailureViews(ro.Pipeline):
        def forward(self, values):
            failed = fail(values)
            selected = ro.F.filter(failed, keep(values))
            children = ro.F.expand(split(values))
            repeated = ro.F.broadcast(selected, like=children)
            return selected, repeated, ro.F.reduce(repeated), identity(failed)

    outputs, _, engine = run_sync(FailureViews(), [0, 2])
    failed = ro.OutputIssue(ro.ItemOutcome.FAILED, expected)
    suppressed = ro.OutputIssue(ro.ItemOutcome.SUPPRESSED, expected)
    assert outputs == (
        [failed, failed], [failed, failed], [[], suppressed], [suppressed, suppressed],
    )
    engine.release_values()
    if isinstance(cause, dict):
        cause["code"] = 8
    assert outputs[0][0].cause == expected


@pytest.mark.parametrize("cause", [None, "cannot split"])
def test_failed_expansion_retains_cause_at_reduce_output(cause):
    @ro.function
    def fail(values):
        return [ro.RecordFailure(cause) for _ in values]

    class FailedExpansion(ro.Pipeline):
        def forward(self, values):
            return ro.F.reduce(ro.F.expand(fail(values)))

    outputs, _, _ = run_sync(FailedExpansion(), [1])
    assert outputs == [ro.OutputIssue(ro.ItemOutcome.SUPPRESSED, cause)]


def test_valid_none_and_empty_values_remain_distinct_from_dropped_outputs():
    class Filtered(ro.Pipeline):
        def forward(self, values, masks):
            return values, ro.F.filter(values, masks)

    outputs, _, _ = run_sync(Filtered(), [None, []], [True, False])
    assert outputs == ([None, []], [None, ro.OutputIssue(ro.ItemOutcome.DROPPED)])


def test_unprintable_business_cause_does_not_hide_recorded_failure():
    class Unprintable:
        def __str__(self):
            raise RuntimeError("broken formatter")

    @ro.function
    def fail(values):
        return [ro.RecordFailure(Unprintable()) for _ in values]

    class Failed(ro.Pipeline):
        def forward(self, values):
            return fail(values)

    outputs, _, _ = run_sync(Failed(), [1])
    assert outputs == [ro.OutputIssue(ro.ItemOutcome.FAILED, "Unprintable")]


def test_output_issue_is_reserved_at_the_output_boundary():
    class Identity(ro.Pipeline):
        def forward(self, values):
            return values

    with pytest.raises(ro.ExecutionError, match="reserved"):
        run_sync(Identity(), [ro.OutputIssue(ro.ItemOutcome.FAILED, "user value")])
