from __future__ import annotations

import numpy as np
import pytest

from rayorch.experimental.multigrain_v2.column import concat, take, validate_column


def test_list_is_the_only_column_container() -> None:
    column = [{"tensor": np.array([1, 2])}, ("row", 2)]
    assert validate_column(column) is column
    for unsupported in ((1, 2), np.array([1, 2]), range(2), "ab"):
        with pytest.raises(TypeError):
            validate_column(unsupported)


def test_take_preserves_selector_order() -> None:
    column = ["a", "b", "c", "d"]
    assert take(column, slice(1, 4, 2)) == ["b", "d"]
    assert take(column, (3, 0, 3)) == ["d", "a", "d"]
    with pytest.raises(TypeError):
        take(column, [0, 1])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        take(column, (True,))  # type: ignore[arg-type]


def test_concat_preserves_column_and_row_order() -> None:
    left = [1, 2]
    right: list[int] = []
    final = [3, 4]
    assert concat((left, right, final)) == [1, 2, 3, 4]
    assert left == [1, 2]
