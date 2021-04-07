var page_start = 0;
var active_page;
var active_graph;

var active_table_indices;

var n_cols;
var n_col_groups;

var col_group_active;
var lang_active = [true, true];
var sig_active = [true, true, true];

var page_max;

var metadata;
const metadata_response = fetch("metadata.json")
    .then(response => response.json());

var graph_data;
const graph_data_response = fetch("graph_data.json")
    .then(response => response.json());

var table_data;
const table_data_response = fetch("table_data.json")
    .then(response => response.json());

var git_history_data;
const git_history_data_response = fetch("git_history.json")
    .then(response => response.json());

google.charts.load('current', {'packages':['corechart']});
var chart_processed_data;


async function awaitInit() {
    await Promise.all([metadata_response, graph_data_response, table_data_response, git_history_data_response]);
}


async function initPage() {
    metadata = await metadata_response;
    graph_data = await graph_data_response;
    table_data = await table_data_response;
    git_history_data = await git_history_data_response;
    console.log(metadata);

    page_max = table_data.length - 1;

    var i, j;

    col_group_active = []
    n_col_groups = metadata["label_indices"].length;
    for (i = 0; i < n_col_groups; i++) {
        col_group_active.push(true);
    }

    n_cols = metadata["column_titles"].length;

    var chart_raw_data = [];
    for (i = 0; i < n_cols; i++) {
        chart_raw_data.push([["Commit", "Python", "C++"]]);
    }
    for (i = 0; i < graph_data.length; i++) {
        var d = new Date(graph_data[i][0][0], graph_data[i][0][1], graph_data[i][0][2]);
        for (j = 0; j < chart_raw_data.length; j++) {
            chart_raw_data[j].push([d, graph_data[i][2][j][0], graph_data[i][2][j][1]])
        }
    }

    chart_processed_data = []
    for (i = 0; i < chart_raw_data.length; i++) {
        chart_processed_data.push(google.visualization.arrayToDataTable(chart_raw_data[i]));
    }

    awaitInit();
    updateTableIndices();
    document.getElementById("button-table").click();
    document.getElementById("graph-select-0").click();
}

document.onkeydown = function(e) {
    switch (e.keyCode) {
        case 37:
            // Left key
            e.preventDefault();
            if (e.shiftKey) {
                document.getElementById("button-" + metadata["left_map"][active_page]).click();
            } else if (active_page == "table") {
                incPage(-10);
            } else if (active_page == "graphs") {
                incGraph(-1);
            }

            break;
        case 38:
            // Up key. Currently unused.
            break;

        case 39:
            // Right key
            e.preventDefault();
            if (e.shiftKey) {
                document.getElementById("button-" + metadata["right_map"][active_page]).click();
            } else if (active_page == "table") {
                incPage(10);
            } else if (active_page == "graphs") {
                incGraph(1);
            }

            break;
        case 40:
            // Down key. Currently unused.
            break;
    }
}

function openTab(evt, className) {
    active_page = className;

    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
        tabcontent[i].style.overflow = "hidden";
    }

    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    document.getElementById(className).style.display = "block";
    document.getElementById(className).style.overflow = "scroll";

    document.getElementById(className + "-help").style.display = "block";
    document.getElementById(className + "-help").style.overflow = "scroll";
    evt.currentTarget.className += " active";

    update();
}

function incPage(i) {
    page_start = Math.max(Math.min(page_start + i, active_table_indices.length - 1), 0);
    update();
}

function mod(n, m) {
    // I have no words for how dumb it is that modulo is formally ** wrong ** in
    // this language.
    return ((n % m) + m) % m;
}

function incGraph(i) {
    var j, candidate;
    for (j = 0; j < n_cols; j++) {
        candidate = mod((active_graph + i * (j + 1)), n_cols);
        if (col_group_active[metadata["i_to_label_index"][candidate]]) {
            selectGraph(candidate);
            return;
        }
    }
}

function setGraphSelection(sha) {
    var history = git_history_data[sha];
    document.getElementById("graph-selection-author").innerHTML = history[0] + " (" + history[1] + ")";
    document.getElementById("graph-selection-commit").innerHTML = (
        '<a href="https://github.com/pytorch/pytorch/commits/' + sha + '" style="color:White" target="_blank">' +
        sha.slice(0, 7) + "</a>"
    );
    document.getElementById("graph-selection-msg").innerHTML = history[3];
}

function selectGraph(benchmark_index) {
    active_graph = benchmark_index;
    document.getElementById("graph-selection-author").innerHTML = "";
    document.getElementById("graph-selection-commit").innerHTML = "";
    document.getElementById("graph-selection-msg").innerHTML = "";

    var chartbuttons;
    chartbuttons = document.getElementsByClassName("graph-control-button");
    for (i = 0; i < chartbuttons.length; i++) {
        chartbuttons[i].className = chartbuttons[i].className.replace(" active", "");
    }
    document.getElementById("graph-select-" + benchmark_index).className += " active";

    var data = chart_processed_data[benchmark_index];
    var options = {
        title: metadata["column_titles"][benchmark_index],
        titleTextStyle: { fontSize: 20, bold: true, alignment: 'center' },
        legend: { position: 'in', maxLines: 5 },
        width: 1000,
        height: 500,
        chartArea: {left:10, top:50, width:'90%', height:'90%'},
        backgroundColor: 'lightgrey',
        curveType: 'none',
        vAxis: { minValue: 0 },
        explorer: {
            actions: ['dragToZoom', 'rightClickToReset'],
            axis: 'horizontal',
            keepInBounds: true,
            maxZoomIn: 30.0
        },
    };

    var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));

    // The select handler. Call the chart's getSelection() method
    function selectHandler() {
        var selectedItem = chart.getSelection()[0];
        if (selectedItem) {
            var value = data.getValue(selectedItem.row, selectedItem.column);
            setGraphSelection(
                graph_data[selectedItem.row][1],
            );
        }
    }

    // Listen for the 'select' event, and call my function selectHandler() when
    // the user selects something on the chart.
    google.visualization.events.addListener(chart, 'select', selectHandler);

    chart.draw(data, options);
}

function updateActive() {
    var i, j, active;
    for (i = 0; i < n_col_groups; i++) {
        active = (document.getElementById('c' + i).checked == true);
        col_group_active[i] = active;
    }

    for (i = 0; i < lang_active.length; i++) {
        lang_active[i] = (document.getElementById('l' + i).checked == true);
    }

    // for (i = 0; i < sig_active.length; i++) {
    //     sig_active[i] = (document.getElementById('s' + i).checked == true);
    // }
}

async function updateTableIndices() {
    awaitInit();

    var threshold = parseFloat(document.getElementById("table_threshold").value);
    active_table_indices = [];

    var i, j, l;
    for (i = 0; i < table_data.length; i++) {
        var keep_row = false;
        var row_data = table_data[i][5];
        for (j = 0; j < n_cols; j++) {
            for (l = 0; l < lang_active.length; l++) {
                if (
                    lang_active[l] &&
                    col_group_active[metadata["i_to_label_index"][j]] &&
                    row_data[l][j] != null &&
                    Math.abs(row_data[l][j][1] * 100) >= threshold) {
                        keep_row = true;
                    }
            }
        }
        if (keep_row) { active_table_indices.push(i); }
    }
}

function updateTable() {
    var i, j, l;
    for (i = 0; i < 10; i++) {
        document.getElementById("dynamic-row-" + i).innerHTML = "";
    }

    var render_rows = []
    for (i = page_start; i < Math.min(page_start + 10, active_table_indices.length); i++) {
        render_rows.push(active_table_indices[i])
    }

    var head_html = document.getElementById("static-thead").innerHTML;
    head_html = head_html.replace(/static-table-/g, '');
    head_html = head_html.replace(/table_control_button_hidden/g, 'table_control_button');
    document.getElementById("dynamic-thead").innerHTML = head_html;

    for (i = 0; i < render_rows.length; i++) {
        row_html = document.getElementById("static-table-row-" + i).innerHTML;
        row_html = row_html.replace(/static-table-/g, '');
        document.getElementById("dynamic-row-" + i).innerHTML = row_html;
    }

    var section, cell;
    for (i = 0; i < Math.min(render_rows.length, 10); i++) {
        var row = table_data[render_rows[i]]
        for (j = 0; j < 4; j++) {
            section = document.getElementById("data-col-" + j + "-" + i);
            section.innerHTML = row[j + 1];
        }

        for (j = 0; j < n_cols; j++) {
            for (l = 0; l < lang_active.length; l++) {
                section = document.getElementById("cell-" + i + "-" + j + "-" + l);
                cell = row[5][l][j];
                if (cell == null || lang_active[l] == false) {
                    section.innerHTML = "";
                } else {
                    section.innerHTML = "<div style=color:" + cell[0] + ">" + (cell[1] * 100).toFixed(1) + "</div>";
                }
            }
        }
    }
}

async function update() {
    awaitInit();
    updateActive();
    updateTableIndices();
    updateTable();

    var i, j, cols;
    for (i = 0; i < n_col_groups; i++){
        cols = document.getElementsByClassName("c" + i);
        for (j = 0; j < cols.length; j++) {
            if (col_group_active[i]) {
                cols[j].style.display = "table-cell";
            } else {
                cols[j].style.display = "none";
            }
        }
    }

}
