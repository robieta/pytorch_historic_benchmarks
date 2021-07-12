import json
import math
import os
import shutil
from typing import Tuple

from v2.bisection import _iter_flat
from v2.containers import BenchmarkResults, ResultRange

REPORT_ROOT = f"/home/{os.getenv('USER')}/persistent/public-90d/public_html/{{doc_name}}"
REPORT_FILE = os.path.join(REPORT_ROOT, f"{{doc_name}}.html")
REPORT_URL_ROOT = f"https://home.fburl.com/~{os.getenv('USER')}/{{doc_name}}"


BACKGROUND_COLOR = "Black"
TABS = {
    "summary": "Summary",
    "table": "Table",
    "graphs": "Graphs",
    "static-table": "(Debug: static table)",
}

LANGUAGES = (
    "Python",
    "C++",
)


class Bookkeeping:
    def __init__(self, results: Tuple[BenchmarkResults, ...]):
        self.top_level_labels = {}
        self.label_order = {}
        self.all_keys = []

        for r in results:
            if r.label[0] not in self.top_level_labels:
                self.top_level_labels[r.label[0]] = len(self.top_level_labels)

            if r.label not in self.label_order:
                self.label_order[r.label] = len(self.label_order)

            if r.key not in self.all_keys:
                self.all_keys.append(r.key)

        self.cols = sorted(
            {(label, autograd, runtime, num_threads)
            for label, _, autograd, runtime, num_threads in self.all_keys},
            key=lambda x: (self.top_level_labels[x[0][0]], self.label_order[x[0]], x[2], x[1], x[3])
        )

        self.grid_pos = {}
        for i, (label, autograd, runtime, num_threads) in enumerate(self.cols):
            self.grid_pos[(label, "Python", autograd, runtime, num_threads)] = (i, 0)
            self.grid_pos[(label, "C++", autograd, runtime, num_threads)] = (i, 1)

        self.i_to_label_index = {
            i: self.top_level_labels[label[0]]
            for (label, _, _, _, _), (i, _) in self.grid_pos.items()
        }


def color_by_value(x):
    min_colored = 0.005
    scale_max = 0.25

    if x > 0:
        # Red
        scale = [
            BACKGROUND_COLOR,
            '#808080', '#867979', '#8c7373', '#936c6c', '#996666',
            '#9f6060', '#a65959', '#ac5353', '#b34d4d', '#b94646',
            '#bf4040', '#c63939', '#cc3333', '#d22d2d', '#d92626',
            '#df2020', '#e61919', '#ec1313', '#f20d0d', '#f90606',
            '#ff0000'
        ]
    else:
        scale = [
            BACKGROUND_COLOR,
            '#737373', '#6d786d', '#677e67', '#628462', '#5c8a5c',
            '#568f56', '#509550', '#4b9b4b', '#45a145', '#3fa63f',
            '#39ac39', '#34b234', '#2eb82e', '#28bd28', '#22c322',
            '#1dc91d', '#17cf17', '#11d411', '#0bda0b', '#06e006',
            '#00e600'
        ]

    x = abs(x)
    if x < min_colored:
        index = 0
    elif x >= scale_max:
        index = -1
    else:
        log_k = math.log2(min_colored / scale_max) / (len(scale) - 1)
        index = int(math.log2(min_colored / x) / log_k) + 1

    return scale[index]


def escape(s: str):
    return s.replace("<", r"\<").replace(">", r"\>")


month_dict = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun",
    "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}
def date_delta(date0, date1=None):
    y0, m0, d0 = date0.split("-")
    m0 = month_dict[m0]

    if date1 is None or date0 == date1:
        return f"{m0}&nbsp;{d0}&nbsp;{y0}"

    y1, m1, d1 = date1.split("-")
    m1 = month_dict[m1]
    if m0 != m1:
        return f"{m0}&nbsp;{d0}&#8209;{m1}&nbsp;{d1}&nbsp;{y1}"
    assert y0 == y1, f"{y0} {y1}"
    return f"{m0}&nbsp;{d0}&#8209;{d1}&nbsp;{y1}"


def write_page(name: str, bookkeeping: Bookkeeping):
    root = REPORT_ROOT.format(doc_name=name)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)


def generate(name: str, result_ranges: Tuple[ResultRange, ...], history):
    results: Tuple[BenchmarkResults, ...] = tuple([r for _, r in _iter_flat(result_ranges)])
    bookkeeping = Bookkeeping(results)

    row_counts = {}
    low_water_mark = {}
    for sha, r in _iter_flat(result_ranges):
        row_counts.setdefault(sha, [[None, None] for _ in range(len(bookkeeping.cols))])
        i, j = bookkeeping.grid_pos[r.key]
        ct = int(r.ct // 100)
        row_counts[sha][i][j] = ct
        low_water_mark[(i, j)] = min(ct, low_water_mark.get((i, j), ct))

    row_deltas = {}
    for result_range in result_ranges:
        grid = [[None, None] for _ in range(len(bookkeeping.cols))]

        r_lower = result_range.lower_results
        r_upper = result_range.upper_results
        lower = {ri.key: ri.ct for ri in r_lower.values}
        upper = {ri.key: ri.ct for ri in r_upper.values}
        keys = set(lower.keys()).intersection(upper.keys())
        for k in keys:
            i, j = bookkeeping.grid_pos[k]
            grid[i][j] = int((upper[k] - lower[k]) // 100) / low_water_mark[(i, j)]

        row_deltas[id(result_range)] = grid

    # write_page(name=name, bookkeeping=bookkeeping)

    import importlib
    import v2.gen_report.write
    importlib.reload(v2.gen_report.write)

    v2.gen_report.write.write_report(
        history,
        bookkeeping.cols,
        bookkeeping.top_level_labels,
        bookkeeping.grid_pos,
        row_counts,
        result_ranges,
        row_deltas,
    )
