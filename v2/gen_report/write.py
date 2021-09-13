import json
import math
import os
import shutil
import textwrap

from yattag import Doc, indent


# DOC_NAME = "test"
DOC_NAME = "wall_v4_preview"
REPORT_ROOT = f"/home/{os.getenv('USER')}/persistent/public-90d/public_html/{DOC_NAME}"
print(REPORT_ROOT)
REPORT_FILE = os.path.join(REPORT_ROOT, f"{DOC_NAME}.html")
REPORT_URL_ROOT = f"https://home.fburl.com/~{os.getenv('USER')}/{DOC_NAME}"
ARTIFACT_ROOT = os.path.dirname(os.path.abspath(__file__))

METADATA_FILE = os.path.join(REPORT_ROOT, "metadata.json")
GIT_HISTORY_FILE = os.path.join(REPORT_ROOT, "git_history.json")
GIT_HISTORY_FULL_FILE = os.path.join(REPORT_ROOT, "git_history_full.json")
GRAPH_DATA_FILE = os.path.join(REPORT_ROOT, "graph_data.json")
TABLE_DATA_FILE = os.path.join(REPORT_ROOT, "table_data.json")

BACKGROUND_COLOR = "Black"

TABS = {
    "summary": "Summary",
    "table": "Table",
    "graphs": "Graphs",
    "static-table": "(Debug: static table)",
}

SIGNIFICANCE_LEVELS = (
    "High",
    "Moderate",
    "Low",
)

LANGUAGES = (
    "Python",
    "C++",
)


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


def sha_to_url(sha):
    return f'<a href="https://github.com/pytorch/pytorch/commit/{sha}" style="color:White">{sha[:7]}</a>'


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

# =============================================================================
# =============================================================================
# =============================================================================
_COL_OFFSET = 5
def col_reprs(cols, table_format=False):
    component_map = {
        "Eager": [],
        "TorchScript": ["(TorchScript)"],
        "Forward": [],
        "Forward + Backward": ["(AutoGrad)"]
    }
    label_map = {
        "Math": "",
        "Data movement": "",
        "add": "Add",
    }
    col_labels = []
    prior_first_component = None
    for i, (label, autograd, runtime, num_threads) in enumerate(cols):
        sub_label = [label_map.get(l, l) for l in label[1:]]
        components = [("&nbsp;" * 3 if table_format else "  ").join([l for l in sub_label if l])]
        if not table_format:
            components.insert(0, f"{label[0]}: ")
        if table_format and components[0] == prior_first_component:
            components[0] = '<span style="opacity:0.5">.....</span>'
        else:
            prior_first_component = components[0]

        components.extend(component_map[runtime])
        components.extend(component_map[autograd])
        components.extend([f"({num_threads} threads)"] if num_threads > 1 else [])
        if len(components) > 1:
            if table_format:
                details = f'<span style="font-size:90%">{"&nbsp;&nbsp;".join(components[1:])}</span>'
                col_labels.append(f"{components[0]}{'&nbsp;' * 4}{details}")
            else:
                col_labels.append(" ".join(components))
        else:
            col_labels.append(components[0])
    return col_labels


def gen_controls(doc, tag, top_level_labels, cols, label_indices, row_counts):
    col_group_bounds = {i[-1] for i in label_indices.values()}
    columns = (
        ("l", LANGUAGES, 1),
        ("c", top_level_labels.keys(), 2),
    )
    with tag("table", klass="control-table"):
        with tag("tr"):
            with tag("td", colspan=3, style="min-width:700px"):
                with tag("div", klass="tab"):
                    for event, desc in TABS.items():
                        with tag("button", klass="tablinks", id=f"button-{event}", onclick=f"openTab(event, '{event}')"):
                            doc.asis(desc)

                doc.asis("<br>")

            with tag("td"):
                pass

            with tag("td", rowspan=2):
                with tag("div", klass="tabcontent", id="summary-help"):
                    pass

                with tag("div", klass="tabcontent", id="table-help"):
                    with tag("font", size=4):
                        doc.stag("br")
                        doc.text("Arrow keys can also control pagination.")

                with tag("div", klass="tabcontent", id="graphs-help"):
                    titles = col_reprs(cols, table_format=True)
                    with tag("table", klass="data-table"):
                        with tag("tr"):
                            for l, i in top_level_labels.items():
                                for j in label_indices[i]:
                                    border = "solid-bottom" if j in col_group_bounds else ""
                                    with tag("th", klass=f"c{i}"):
                                        with tag("div", klass="rotated-header-container"):
                                            with tag("div", klass=f"rotated-header-content {border}"):
                                                doc.asis(titles[j])

                        with tag("tr"):
                            for l, i in top_level_labels.items():
                                with tag("td", klass=f"top-level-label c{i}", colspan=len(label_indices[i])):
                                    doc.text(l)

                        with tag("tr"):
                            for l, i in top_level_labels.items():
                                for j in label_indices[i]:
                                    border = "solid-right" if j in col_group_bounds else ""
                                    with tag("td", klass=f"c{i} {border}"):
                                        doc.stag("br")
                                        with tag("button", id=f"graph-select-{j}", klass="graph-control-button", onclick=f"selectGraph({j})"):
                                            pass

                with tag("div", klass="tabcontent", id="static-table-help"):
                    with tag("font", size=4):
                        doc.text(
                            "No JS version of the table. Mostly for debugging, but can also be used to ctrl-f. "
                            "Be aware that it is slow. (As it contains ALL rows.)"
                        )

        with tag("tr"):
            with tag("td"):
                doc.text("Delta threshold (%)")
                doc.stag("br")
                doc.input(
                    name="table_threshold",
                    id="table_threshold",
                    type="text",
                    value="1.0",
                    onchange="page_start = 0; update()",
                )

            for prefix, entries, rowspan in columns:
                with tag("td"):
                    for i, label in enumerate(entries):
                        with tag("div"):
                            doc.input(
                                name=f"{prefix}{i}",
                                id=f"{prefix}{i}",
                                type="checkbox",
                                onclick=f"update()",
                                checked=True
                            )
                            with tag("label", **{"for": f"{prefix}{i}"}):
                                doc.asis(label)




def gen_data(doc, tag, label_indices, history, cols, row_counts, row_deltas, result_ranges, i_to_label_index):
    tab_keys = list(TABS.keys())
    right_map = {k: v for k, v in zip(tab_keys, tab_keys[1:] + tab_keys[:1])}
    metadata = {
        "column_titles": col_reprs(cols, table_format=False),
        "table_column_titles": col_reprs(cols, table_format=True),
        "label_indices": list(label_indices.values()),
        "i_to_label_index": i_to_label_index,
        "left_map": {v: k for k, v in right_map.items()},
        "right_map": right_map,
    }

    with open(METADATA_FILE, "wt") as f:
        json.dump(metadata, f)

    tested_commits = [commit for commit in history if commit.sha in row_counts]

    with open(GIT_HISTORY_FILE, "wt") as f:
        json.dump({
        commit.sha: (commit.author_name, commit.author_email, commit.date_str, commit.msg)
        for commit in tested_commits
    }, f, indent=4)

    with open(GIT_HISTORY_FULL_FILE, "wt") as f:
        json.dump({
        commit.sha: (commit.author_name, commit.author_email, commit.date_str, commit.msg)
        for commit in history
    }, f, indent=4)

    graph_data = []
    for commit in tested_commits:
        date_ints = [int(i) for i in commit.date_str.split("-")]
        date_ints[1] -= 1  # JS month is zero indexed
        graph_data.append(json.dumps([date_ints, commit.sha, row_counts[commit.sha]]))


    with open(GRAPH_DATA_FILE, "wt") as f:
        f.write("[\n" + textwrap.indent(",\n".join(graph_data), " " * 4) + "\n]")

    table_data = []
    for result_range in result_ranges:
        grid = row_deltas.get(id(result_range), None)
        if grid is None:
            continue

        c0: Commit = result_range.lower_commit
        c1: Commit = result_range.upper_commit
        is_range = bool(result_range.intermediate_commits)

        row = [None]  # Used to be si
        if bool(result_range.intermediate_commits):
            row.extend([
                f"{sha_to_url(c0.sha)} - {sha_to_url(c1.sha)}",
                date_delta(c0.date_str, c1.date_str),
                f"({len(result_range.intermediate_commits) + 1} commits)",
                "..."
            ])
        else:
            row.extend([
                sha_to_url(c1.sha),
                date_delta(c1.date_str),
                c1.author_name,
                escape(c1.msg),
            ])

        row.append(list(zip(*[
            [None if gij is None else (color_by_value(gij), gij) for gij in gi]
            for gi in grid
        ])))

        table_data.append(row)

    with open(TABLE_DATA_FILE, "wt") as f:
        json.dump(table_data, f)

    with tag("script", type="text/javascript", src="https://www.gstatic.com/charts/loader.js"):
        pass

    return table_data


def write_static_table(doc, tag, table_data, cols, top_level_labels, i_to_label_index, label_indices, result_ranges, row_deltas):
    col_group_bounds = {i[-1] for i in label_indices.values()}
    with tag("table", klass="data-table"):
        with tag("thead", id="static-thead"):
            with tag("tr"):
                with tag("td", colspan=_COL_OFFSET):
                    with tag("button", klass="table_control_button_hidden", onclick="incPage(-1000)"):
                        doc.text("<<<")

                    with tag("button", klass="table_control_button_hidden", onclick="incPage(-50)"):
                        doc.text("<<")

                    with tag("button", klass="table_control_button_hidden", onclick="incPage(-10)"):
                        doc.text("<")

                    with tag("button", klass="table_control_button_hidden", onclick="incPage(10)"):
                        doc.text(">")

                    with tag("button", klass="table_control_button_hidden", onclick="incPage(50)"):
                        doc.text(">>")

                    with tag("button", klass="table_control_button_hidden", onclick="incPage(1000)"):
                        doc.text(">>>")

                for i, l in enumerate(col_reprs(cols, table_format=True)):
                    with tag("th", klass=f"static-table-c{i_to_label_index[i]}"):
                        with tag("div", klass="rotated-header-container"):
                            border = "solid-bottom" if i in col_group_bounds else ""
                            with tag("div", klass=f"rotated-header-content {border}"):
                                doc.asis(l)

            with tag("tr"):
                with tag("td", colspan=_COL_OFFSET):
                    pass

                for l, i in top_level_labels.items():
                    with tag("td", klass=f"top-level-label static-table-c{i}", colspan=len(label_indices[i])):
                        doc.text(l)

        with tag("tbody"):
            for row_i, (_, commit_link, date_str, author, msg, (py_row, cpp_row)) in enumerate(table_data[:10]):
                with tag("tbody", id=f"static-table-row-{row_i}"):
                    def add_cells(lang: str):
                        li = {l: li for li, l in enumerate(LANGUAGES)}[lang]
                        for col_i, color_value in enumerate((py_row, cpp_row)[li]):
                            dashed = "dashed-right" if col_i in col_group_bounds else ""
                            with tag("td", id=f"static-table-cell-{row_i}-{col_i}-{li}", klass=f"col static-table-c{i_to_label_index[col_i]} static-table-l{li} {dashed}"):
                                if not color_value:
                                    continue

                                color, value = color_value
                                with tag("div", style=f"color:{color}"):
                                    doc.asis(f"{value * 100:.1f}")


                    with tag("tr", klass="data-row"):
                        with tag("td", id=f"data-col-0-{row_i}", klass=f"col-0", rowspan=2):
                            doc.asis(commit_link)

                        with tag("td", id=f"data-col-1-{row_i}", klass=f"col-1", rowspan=2):
                            doc.asis(date_str)

                        with tag("td", id=f"data-col-2-{row_i}", klass=f"col-2", rowspan=2):
                            doc.asis(author)

                        with tag("td", id=f"data-col-3-{row_i}", klass=f"col-3", rowspan=2):
                            doc.asis(msg)

                        with tag("td", klass="col-4"):
                            doc.asis("Py")

                        add_cells("Python")

                    with tag("tr", klass="data-row"):
                        with tag("td", klass="col-4"):
                            doc.asis("C++")

                        add_cells("C++")

                    with tag("tr"):
                        with tag("td", colspan=_COL_OFFSET + len(cols), klass=f"spacer"):
                            pass


def write_report(history, cols, top_level_labels, grid_pos, row_counts, result_ranges, row_deltas):
    if not os.path.exists(REPORT_ROOT):
        os.makedirs(REPORT_ROOT, exist_ok=True)

    for i in os.listdir(REPORT_ROOT):
        os.remove(os.path.join(REPORT_ROOT, i))

    i_to_label_index = {
        i: top_level_labels[label[0]]
        for (label, _, _, _, _), (i, _) in grid_pos.items()
    }

    label_indices = {i: [] for i in range(len(top_level_labels))}
    for i, label_index in i_to_label_index.items():
        label_indices[label_index].append(i)

    doc, tag, text, line = Doc().ttl()
    def add_source(source):
        shutil.copy(os.path.join(ARTIFACT_ROOT, source), os.path.join(REPORT_ROOT, source))
        if source.endswith(".css"):
            doc.stag("link", rel="stylesheet", href=source)

        elif source.endswith(".js"):
            with tag("script", type="text/javascript", src=source):
                pass

        else:
            raise ValueError(f"Invalid file: {source}")

    doc.asis("<!DOCTYPE html>")
    with tag("html"):
        with tag("head"):
            add_source("tabs.css")
            add_source("style.css")
            table_data = gen_data(doc, tag, label_indices, history, cols, row_counts, row_deltas, result_ranges, i_to_label_index)
            add_source("style.js")

        with tag("body", klass="page_style", onload="initPage();"):
            gen_controls(doc, tag, top_level_labels, cols, label_indices, row_counts)
            doc.asis("<br><br>")

            with tag("div", klass="tabcontent", id="summary"):
                body = textwrap.dedent("""
                    <h1>Note: This is still WIP. More documentation is coming.</h1>

                    <h2>Welcome to the Wall of Serotonin (TM) v2.</h2>

                    <h1>The checkboxes hide various aspects of the data:</h1>
                    &nbsp;- Levels of significance. (high, moderate, and low)
                    &nbsp;- Python or C++ (Table only)
                    &nbsp;- High level groupings of benchmarks

                    Data can be viewed in either table or graph form.

                    On the table tab deltas are shown using the formula:
                    diff = (b - a) / low_water_mark

                    On the graph tab, absolute instruction counts are shown.

                    Shift + arrow key (left / right) moves between tabs.
                """)

                doc.asis("\n<br>".join(body.splitlines()))

            with tag("div", klass="tabcontent", id="table"):
                with tag("table", klass="data-table"):
                    with tag("thead", id=f"dynamic-thead"):
                        pass
                    for i in range(10):
                        with tag("tbody", id=f"dynamic-row-{i}"):
                            pass

            with tag("div", klass="tabcontent", id="graphs"):
                with tag("table"):
                    with tag("tr"):
                        with tag("td", style="min-width:450px;max-width:450px", valign="top"):
                            with tag("font", size=6):
                                doc.text("Controls:")
                                doc.stag("br")
                            with tag("font", size=4):
                                doc.asis("&nbsp;&nbsp;Arrow keys move selection left / right.")
                                doc.stag("br")
                                doc.asis("&nbsp;&nbsp;(Or click on the buttons to select.)")
                                doc.stag("br")
                                doc.stag("br")
                                doc.asis("&nbsp;&nbsp;Drag to zoom in. Right click to reset.")
                                doc.stag("br")
                                doc.stag("br")
                                doc.asis("&nbsp;&nbsp;Click on points to get the commit.")

                            for _ in range(4):
                                doc.stag("br")

                            with tag("font", size=5):
                                with tag("div", id="graph-selection-author"):
                                    pass

                                doc.stag("br")
                                with tag("div", id="graph-selection-commit"):
                                    pass

                            with tag("font", size=4):
                                doc.stag("br")
                                with tag("div", id="graph-selection-msg"):
                                    pass

                        with tag("td", valign="top"):
                            with tag("div", id="curve_chart"):
                                pass

            with tag("div", klass="tabcontent", id="static-table"):
                write_static_table(
                    doc,
                    tag,
                    table_data,
                    cols,
                    top_level_labels,
                    i_to_label_index,
                    label_indices,
                    result_ranges,
                    row_deltas,
                )

    with open(REPORT_FILE, "wt") as f:
        f.write(indent(doc.getvalue()))

    print("Done.")
