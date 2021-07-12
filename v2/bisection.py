from typing import Iterator, Tuple

from v2.containers import BenchmarkResults, ResultRange


_BASELINE_THRESHOLD = 0.05  # 5%
_POWER = 2
_CEIL_FACTOR = 4


def _iter_flat(result_ranges: Tuple[ResultRange, ...]) -> Iterator[Tuple[str, BenchmarkResults]]:
    for r in result_ranges:
        results = r.lower_results
        if results is not None:
            for i in results.values:
                yield r.lower_commit.sha, i

    results = r.upper_results
    if results is not None:
        for i in results.values:
            yield r.upper_commit.sha, i


def bisection_step(result_ranges: Tuple[ResultRange]) -> Tuple[str, ...]:
    if not result_ranges:
        return ()

    measure_shas = []
    shas_by_priority = {}

    low_water_mark = {}
    for _, r in _iter_flat(result_ranges):
        low_water_mark.setdefault(r.key, r.ct)
        low_water_mark[r.key] = min(low_water_mark[r.key], r.ct)

    for result_range in result_ranges:
        bisect_index = int(len(result_range.intermediate_commits) // 2)
        r_lower = result_range.lower_results
        r_upper = result_range.upper_results

        if not result_range.intermediate_commits:
            continue

        if r_lower is None and r_upper is None:
            continue

        intermediate_sha = result_range.intermediate_commits[bisect_index].sha
        if r_lower is None or r_upper is None:
            measure_shas.append(intermediate_sha)
            continue

        lower = {r.key: r.ct for r in r_lower.values}
        upper = {r.key: r.ct for r in r_upper.values}
        if set(lower.keys()) != set(upper.keys()):
            measure_shas.append(intermediate_sha)
            continue

        shas_by_priority[intermediate_sha] = 0
        for k, v_lower in lower.items():
            v_upper = upper[k]
            delta = (v_upper - v_lower) / low_water_mark[k]

            shas_by_priority[intermediate_sha] += min(abs(delta / _BASELINE_THRESHOLD) ** _POWER, _CEIL_FACTOR)

    measure_shas.extend([
        k for k, v in sorted(shas_by_priority.items(), key=lambda kv: kv[1], reverse=True)
        if v >= 1
    ])

    return tuple(measure_shas)
