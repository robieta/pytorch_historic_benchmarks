import dataclasses
import enum
import math
import os
import pickle
import statistics
import textwrap
from typing import Iterator, List, Optional, Tuple

from yattag import Doc, indent

from v2.build import check_unbuildable
from v2.containers import BenchmarkResult, BenchmarkResults, Commit, ResultRange
from v2.gen_report.write import write_report
from v2.runner import Runner


class RowClassification(enum.Enum):
    HighSignificance = 0
    ModerateSignificance = 1
    LowSignificance = 2
    NoSignificance = 3

    @property
    def criteria(self):
        if self == RowClassification.HighSignificance:
            return ((1, 0.10),)
        if self == RowClassification.ModerateSignificance:
            return ((1, 0.05),)
        if self == RowClassification.LowSignificance:
            return ((0, 0.01),)

        return ((0, 0.0),)

    @staticmethod
    def characterize(values):
        for row_classification in RowClassification:
            for n_criteria, rel_diff_criteria in row_classification.criteria:
                if sum(abs(i) >= rel_diff_criteria for i in values) >= n_criteria:
                    return row_classification

        # Fallback, though it should not be needed.
        return RowClassification.NoSignificance


def _iter_flat(result_ranges: Tuple[ResultRange]) -> BenchmarkResult:
    for r in result_ranges:
        results = r.lower_results
        if results is not None:
            for i in results.values:
                yield r.lower_commit.sha, i

    results = r.upper_results
    if results is not None:
        for i in results.values:
            yield r.upper_commit.sha, i


def make_report(self: Runner):
    # Determine table params.
    top_level_labels = {}
    label_order = {}
    low_water_mark = {}
    all_keys = []

    result_ranges = self._group_ranges()[::-1]
    for _, r in _iter_flat(result_ranges):
        if r.label[0] not in top_level_labels:
            top_level_labels[r.label[0]] = len(top_level_labels)

        if r.label not in label_order:
            label_order[r.label] = len(label_order)

        if r.key not in all_keys:
            all_keys.append(r.key)

        low_water_mark.setdefault(r.key, r.instructions)
        low_water_mark[r.key] = min(low_water_mark[r.key], r.instructions)

    cols = sorted(
        {(label, autograd, runtime, num_threads)
        for label, _, autograd, runtime, num_threads in all_keys},
        key=lambda x: (top_level_labels[x[0][0]], label_order[x[0]], x[2], x[1], x[3])
    )
    grid_pos = {}
    for i, (label, autograd, runtime, num_threads) in enumerate(cols):
        grid_pos[(label, "Python", autograd, runtime, num_threads)] = (i, 0)
        grid_pos[(label, "C++", autograd, runtime, num_threads)] = (i, 1)

    # Process Data.
    all_tests_ref = {}
    for result_range in result_ranges:
        if result_range.upper_results is None:
            continue

        at_new = {r.key: r.instructions for r in result_range.upper_results.values}
        if len(at_new) >= len(all_tests_ref):
            all_tests_ref = at_new
    assert len(all_tests_ref) == len(all_keys), f"{len(all_tests_ref)} {len(all_keys)}"

    row_classifications = {}
    row_deltas = {}
    bisect_ranges: List[ResultRange] = []
    for result_range in result_ranges:
        if result_range.lower_results is None and result_range.upper_results is None:
            continue

        if result_range.lower_results is None or result_range.upper_results is None:
            bisect_ranges.append(result_range)
            continue

        include_in_bisect = (
            {ri.key for ri in result_range.lower_results.values} !=
            {ri.key for ri in result_range.upper_results.values}
        )

        grid = [[None, None] for _ in range(len(cols))]

        lower = {r.key: r.instructions for r in result_range.lower_results.values}
        upper = {r.key: r.instructions for r in result_range.upper_results.values}

        for key, i1 in upper.items():
            if key not in lower:
                continue

            i0 = lower[key]
            abs_delta = abs(i1 - i0) / statistics.mean([i0, i1])
            rel_delta = (i1 - i0) / low_water_mark[key]

            if abs_delta > 0.03 and result_range.lower_commit.date_str >= "09/01/2020":
                include_in_bisect = True

            i, j = grid_pos[key]
            grid[i][j] = rel_delta

        row_deltas[id(result_range)] = grid

        grid_for_criteria = [max(abs(d_py or 0), abs(d_cpp or 0)) for d_py, d_cpp in grid]
        c = RowClassification.characterize(grid_for_criteria)
        if c == RowClassification.HighSignificance:
            include_in_bisect = True

        row_classifications[id(result_range)] = c
        if include_in_bisect:
            bisect_ranges.append(result_range)

    row_counts = {}
    for sha, r in _iter_flat(result_ranges):
        grid = row_counts.get(sha, None)
        if grid is None:
            row_counts[sha] = grid = [[None, None] for _ in range(len(cols))]
        i, j = grid_pos[r.key]
        grid[i][j] = r.instructions

    bisect_count = 0
    for bisect_range in bisect_ranges:
        if not bisect_range.intermediate_commits:
            continue

        bisect_index = int(len(bisect_range.intermediate_commits) // 2)
        sha = bisect_range.intermediate_commits[bisect_index].sha
        self._state.maybe_enqueue_build(sha)
        bisect_count += 1

    print(bisect_count)

    # return

    import importlib
    import v2.gen_report.write
    importlib.reload(v2.gen_report.write)

    si_map = {
        RowClassification.HighSignificance: 0,
        RowClassification.ModerateSignificance: 1,
        RowClassification.LowSignificance: 2,
    }
    row_classification_indicies = {
        id(r): si_map[row_classifications[id(r)]]
        for r in result_ranges
        if row_classifications.get(id(r), None) in si_map
    }

    v2.gen_report.write.write_report(
        self._history, cols, top_level_labels, grid_pos, row_counts, result_ranges,
        row_deltas, row_classification_indicies
    )
