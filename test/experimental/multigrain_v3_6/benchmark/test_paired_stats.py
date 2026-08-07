"""Tests for statistics shared by paired benchmark runners."""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3_6.benchmark.paired_stats import (
    paired_timing_summary,
    sample_summary,
)


def test_sample_summary_reports_sample_dispersion():
    sample = sample_summary([1.0, 2.0, 3.0])

    assert sample["mean"] == 2.0
    assert sample["sample_variance"] == 1.0
    assert sample["sample_stddev"] == 1.0


def test_paired_timing_summary_reports_order_bias():
    summary = paired_timing_summary(
        [10.0, 10.0, 10.0, 10.0],
        [11.0, 9.0, 12.0, 8.0],
        ["v3_first", "v36_first", "v3_first", "v36_first"],
    )

    assert summary["v36_faster_trials"] == 2
    assert summary["v3_faster_trials"] == 2
    assert summary["paired_v36_relative_change"]["mean"] == 0.0
    assert summary["relative_change_by_order"]["v3_first"]["mean"] == pytest.approx(0.15)
    assert summary["relative_change_by_order"]["v36_first"]["mean"] == pytest.approx(-0.15)
