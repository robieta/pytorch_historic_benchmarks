import atexit
import argparse
import collections
import itertools as it
import json
import os
import multiprocessing.dummy
import queue
import pickle
import random
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import threading
import time

from utils import build_pytorch


ROOT = os.path.split(os.path.abspath(__file__))[0]
WORKSPACE_ROOT = os.path.join(ROOT, "workspace", "sweeping_bisect")
SCRATCH_ROOT = os.path.join(WORKSPACE_ROOT, "scratch")
RESULTS_ROOT = os.path.join(WORKSPACE_ROOT, "results")
BUILT_RECORD = os.path.join(WORKSPACE_ROOT, "built.json")
SWEEP_START = "2019-09-01"
SWEEP_CADENCE = 7  # days

CALLGRIND_NUM_CORES = int(multiprocessing.cpu_count() // 6)

CALLGRIND_REPLICATES = 1
CALLGRIND_LOOP_NUMBER = 10000

WALL_TIME_REPLICATES = 10
WALL_TIME_SEC = 10

PASS = "pass"
SCALAR_X = "x = torch.ones((1,))"
SCALAR_XY = "x = torch.ones((1,));y = torch.ones((1,))"
TWO_BY_TWO_XY = "x = torch.ones((2, 2)); y = torch.ones((2, 2))"
SMALL_X = "x = torch.ones((10,))"
TEN_BY_TEN_X = "x = torch.ones((10, 10))"
SCALAR_INDEX_SETUP = "v = torch.randn(1,1,1)"
VECTOR_INDEX_SETUP = "a = torch.zeros(100, 100, 1, 1, 1); b = torch.arange(99, -1, -1).long()"
TASKS = (
    # Allocation. (With and without storage)
    (PASS, "torch.empty(())"),
    (PASS, "torch.empty((1,))"),

    # Math
    (SCALAR_X, "x += 1"),
    (SCALAR_X, "x *= 1"),
    (SCALAR_X, "x.sum()"),
    (TWO_BY_TWO_XY, "torch.mm(x, y)"),
    (TWO_BY_TWO_XY, "torch.matmul(x, y)"),
    (SCALAR_XY, "x == y"),
    (SMALL_X, "x.uniform_()"),

    # Indexing
    (SCALAR_INDEX_SETUP, "v[0] = 1"),
    (SCALAR_INDEX_SETUP, "v[0, 0] = 1"),
    (SCALAR_INDEX_SETUP, "v[0, 0, 0] = 1"),
    (SCALAR_INDEX_SETUP, "v[...] = 1"),
    (SCALAR_INDEX_SETUP, "v[:] = 1"),
    (SCALAR_INDEX_SETUP, "v[None] = 1"),
    (SCALAR_INDEX_SETUP, "v[False] = 1"),
    (SCALAR_INDEX_SETUP, "v[True] = 1"),
    (VECTOR_INDEX_SETUP, "a[b, None, ..., :, True] = 1"),

    # Data movement and assignment
    (SCALAR_X, "x.zero_()"),
    (SCALAR_X, "x.resize_(0)"),
    (SCALAR_XY, "x.copy_(y)"),
    (SCALAR_X, "x.contiguous()"),
    (SCALAR_X, "x.clone()"),
    (TEN_BY_TEN_X, "x.t().contiguous()"),

    # Metadata and views.
    (SCALAR_X, "x.size()[0]"),
    (SCALAR_X, "x.stride(0)"),
    (TEN_BY_TEN_X, "torch.as_strided(x, (4, 3), (10, 3), 6)"),
    (TEN_BY_TEN_X, "x.select(1, 4)"),
    (SMALL_X, "x.view(-1, 2)"),
    (SMALL_X, "x.unsqueeze(0)"),

    # TorchScript (measurement code place `model` in globals)
    (SCALAR_X, "model(x)  # TS: script fn `x + 1`"),
    (SCALAR_X, "model(x)  # TS: script Module `x + 1`"),

    # Autograd
    (
        "x = torch.ones((1,)) + torch.ones((1,), requires_grad=True)",
        "x.backward()"
    ),
    (
        f"{SCALAR_X}; w = torch.ones((1,), requires_grad=True)",
        "y = x * w;y.backward()"
    ),
    (
        f"{SCALAR_X}; w0 = torch.ones((1,), requires_grad=True); w1 = torch.ones((1,), requires_grad=True)",
        "y = torch.nn.functional.relu(x * w0) * w1;y.backward()"
    ),

    # Autograd + TorchScript
    (
        "x = torch.ones((1,)) + torch.ones((1,), requires_grad=True)",
        "model(x).backward()  # TS: script fn `x + 1`"
    ),
)

_TASK_GROUPS = [
    (2, "Allocation."),
    (7, "Math"),
    (9, "Indexing"),
    (6, "Data movement"),
    (6, "Metadata & views"),
    (2, "TorchScript"),
    (4, "AutoGrad (+TS)"),
]
assert sum(i[0] for i in _TASK_GROUPS) == len(TASKS)


class Runner:
    _taskset_cores = f"1-{multiprocessing.cpu_count() - 1}"
    _build_queue_len = 6
    _concurrent_builds = 2
    _build_max_jobs = int(multiprocessing.cpu_count() * 0.4)

    def __init__(self, pytorch_builder: build_pytorch.PytorchBuildHelper) -> None:
        # Reference env, and source of benchmark_utils
        timer_env_record = os.path.join(WORKSPACE_ROOT, "timer_env.txt")
        if not os.path.exists(timer_env_record):
            timer_env = pytorch_builder.build_clean(
                checkout="gh/taylorrobie/callgrind_backtest",
                show_progress=True)
            with open(timer_env_record, "wt") as f:
                f.write(timer_env)

        with open(timer_env_record, "rt") as f:
            timer_env = f.read().strip()

        self._pytorch_builder = pytorch_builder
        self._history = pytorch_builder.get_history_since(SWEEP_START)
        self._benchmark_utils_dir = os.path.join(
            self.get_torch_location(timer_env), "utils/benchmark")

        self._lock = threading.Lock()
        self._state_path = os.path.join(WORKSPACE_ROOT, "state.json")
        self._state = self.load_state()
        self._queue = collections.deque()

        # Initial sweep
        self._initial_sweep_indices = [0]
        for i, (_, date, _) in enumerate(self._history):
            if (date - self._history[self._initial_sweep_indices[-1]][1]).days >= SWEEP_CADENCE:
                self._initial_sweep_indices.append(i)
        if self._initial_sweep_indices[-1] != len(self._history) - 1:
            self._initial_sweep_indices.append(len(self._history) - 1)

    def load_state(self):
        if os.path.exists(self._state_path):
            with open(self._state_path, "rt") as f:
                return json.load(f)
        return {
            "built": {},
            "finished": {},
        }

    def save_state(self):
        with open(self._state_path, "wt") as f:
            json.dump(self._state, f)

    def maybe_enqueue_build(self, sha):
        enqueue = (
            sha not in self._state["finished"] and
            sha not in self._state["built"] and
            sha not in self._queue
        )
        if enqueue:
            self._queue.append(sha)

    def loop(self):
        self.bisect()

        while self._queue:
            num_already_built = len(self._state["built"])
            if self._build_queue_len > num_already_built:
                with multiprocessing.dummy.Pool(self._concurrent_builds) as pool:
                    pool.map(self.build, range(self._build_queue_len - num_already_built), 1)

            shas_to_test = list(self._state["built"].keys())
            scratch_dirs = {
                sha: self.make_scratch_dir_for(sha)
                for sha in shas_to_test
            }

            try:
                def count_map_fn(sha):
                    self.collect_counts(sha, scratch_dirs[sha])

                with multiprocessing.dummy.Pool(6) as pool:
                    pool.map(count_map_fn, shas_to_test)

                timing_tasks = []
                for sha in shas_to_test:
                    task_indices = list(range(len(TASKS))) * WALL_TIME_REPLICATES
                    random.shuffle(task_indices)
                    blocked_tasks = [[]]
                    for task_index in task_indices:
                        blocked_tasks[-1].append(task_index)
                        if len(blocked_tasks[-1]) > 8:
                            blocked_tasks.append([])

                    if not blocked_tasks[-1]:
                        blocked_tasks.pop()

                    for i, block in enumerate(blocked_tasks):
                        timing_tasks.append((sha, i, block))

                num_cores = multiprocessing.cpu_count()
                core_list = list(range(2, num_cores))

                def timing_map_fn(args):
                    sha, i, block = args
                    conda_env = self._state["built"][sha]
                    result_file = os.path.join(scratch_dirs[sha], f"block{i:0>2}.pkl")
                    with open(result_file, "wb") as f:
                        pickle.dump(block, f)

                    core = core_list.pop()
                    try:
                        self._pytorch_builder.subprocess_call(
                            f"taskset --cpu-list {core} "
                            f"python -u {os.path.abspath(__file__)} "
                            f"--mode measure_times --result_file {result_file}",
                            shell=True,
                            check=True,
                            conda_env=conda_env,
                        )

                    finally:
                        core_list.append(core)

                print("Begin timing:")
                start_time = time.time()
                with multiprocessing.dummy.Pool(num_cores - 2) as pool:
                    for i, _ in enumerate(pool.imap(timing_map_fn, timing_tasks, 1)):
                        print(f"\r{i + 1} / {len(timing_tasks)}  ({time.time() - start_time:.0f} sec)", end="")
                    print()

                for sha in shas_to_test:
                    conda_env = self._state["built"][sha]
                    result_file = os.path.join(scratch_dirs[sha], "result_paths.txt")
                    results_path = os.path.join(RESULTS_ROOT, f"{sha}.pkl")
                    results_path_simple = os.path.join(RESULTS_ROOT, f"{sha}_simple.pkl")

                    with open(result_file, "wb") as f:
                        pickle.dump([
                            scratch_dirs[sha],
                            results_path,
                            results_path_simple,
                        ], f)

                    self._pytorch_builder.subprocess_call(
                        f"python -u {os.path.abspath(__file__)} "
                        f"--mode merge_measurements --result_file {result_file}",
                        shell=True,
                        check=True,
                        conda_env=conda_env
                    )

                    with self._lock:
                        self._state["built"].pop(sha)
                        self._state["finished"][sha] = (results_path, results_path_simple)
                        self.save_state()
                        shutil.rmtree(conda_env)

            finally:
                for d in scratch_dirs.values():
                    shutil.rmtree(d)

            self.bisect()

    def build(self, _):
        with self._lock:
            if not self._queue:
                return
            sha = self._queue.popleft()

        print(f"Building: {sha}")
        start_time = time.time()
        conda_env = self._pytorch_builder.build_clean(
            sha,
            show_progress=False,
            taskset_cores=self._taskset_cores,
            max_jobs=self._build_max_jobs,
        )

        if conda_env is None:
            return  # unbuildable

        with self._lock:
            self._state["built"][sha] = conda_env
            self.save_state()

        print(f"Build time: {sha} {time.time() - start_time:.0f} sec")

    def collect_counts(self, sha, scratch_dir):
        print(f"Begin measure (counts): {sha}")

        start_time = time.time()
        conda_env = self._state["built"][sha]
        self.monkey_patch_benchmark_utils(conda_env)
        counts_path = os.path.join(scratch_dir, "instruction_counts.pkl")
        cpp_ext_path = os.path.join(scratch_dir, "cpp_ext")

        def per_line_fn(l):
            if "TimeoutExpired" in l:
                print(l)

        self._pytorch_builder.subprocess_call(
            f"taskset --cpu-list {self._taskset_cores} "
            f"python -u {os.path.abspath(__file__)} "
            f"--mode measure_counts --result_file {counts_path}",
            shell=True,
            check=True,
            env={"TORCH_EXTENSIONS_DIR": cpp_ext_path},
            conda_env=conda_env,
            per_line_fn=per_line_fn,
        )
        print(f"Counts time: {sha} {time.time() - start_time:.0f} sec")

    def bisect(self):
        self._queue.clear()
        for i in self._initial_sweep_indices:
            sha = self._history[i][0]
            self.maybe_enqueue_build(sha)

        segment_results = self.segment_results()
        for threshold in [0.25, 0.05, 0.01, 0.001]:
            for r in segment_results:
                max_abs_delta = max(abs(i) for i in r["Count deltas"] if i is not None)
                intermediate_shas = r["Intermediate SHAs"]
                if max_abs_delta >= threshold and intermediate_shas:
                    self.maybe_enqueue_build(intermediate_shas[int(len(intermediate_shas) // 2)])

                if len(self._queue) >= self._build_queue_len:
                    return

    def segment_results(self):
        finished_results = []
        lower_sha = None
        intermediate_shas = {lower_sha: []}
        for sha, date, msg in self._history:
            if self._pytorch_builder.unbuildable(sha):
                continue

            elif sha in self._state["finished"]:
                finished_results.append((sha, date, msg, self._state["finished"][sha][1]))
                lower_sha = sha
                intermediate_shas[sha] = []

            else:
                intermediate_shas[lower_sha].append(sha)

        if not finished_results:
            return

        results = []
        for sha, date, msg, results_path in finished_results:
            with open(results_path, "rb") as f:
                c, t = pickle.load(f)
                results.append((sha, date, msg, c, t))

        def low_water_mark(values):
            return [
                min(k for k in j if k is not None)
                for j in zip(*values)
            ]

        def deltas(x0, x1, lwm):
            return [
                (x1_i - x0_i) / lwm_i
                if (x0_i is not None and x1_i is not None)
                else None
                for x0_i, x1_i, lwm_i in zip(x0, x1, lwm)
            ]

        count_lwm = low_water_mark([i[3] for i in results])
        time_lwm = low_water_mark([i[4] for i in results])

        output = []
        for r0, r1 in zip(results[:-1], results[1:]):
            sha_0, date_0, msg_0, c0, t0 = r0
            sha_1, date_1, msg_1, c1, t1 = r1
            output.append({
                "SHAs": (sha_0, sha_1),
                "Intermediate SHAs": intermediate_shas[sha_0],
                "Dates": (date_0, date_1),
                "Counts": (c0, c1),
                "Count deltas": deltas(c0, c1, count_lwm),
                "Times": (t0, t1),
                "Time deltas": deltas(t0, t1, time_lwm),
                "Messages": (msg_0, msg_1),
            })
        return output

    def debug(self):
        for r in self.segment_results():
            max_delta = max(abs(i) for i in r["Count deltas"] if i is not None)
            if max_delta < 0.05:
                continue

            median_delta = statistics.median(i for i in r["Count deltas"] if i is not None)
            print(f"{max_delta * 100:5.1f}%,  {median_delta * 100:5.1f}%,  {len(r['Intermediate SHAs']):>4}")

    def get_torch_location(self, conda_env):
        lines = []
        self._pytorch_builder.subprocess_call(
            "python -c 'import torch;print(torch.__file__)'",
            shell=True,
            check=True,
            per_line_fn=lambda l: lines.append(l),
            conda_env=conda_env,
        )
        assert len(lines) == 1
        assert lines[0].startswith(conda_env)
        assert lines[0].endswith("torch/__init__.py")
        return os.path.split(lines[0])[0]

    def monkey_patch_benchmark_utils(self, conda_env):
        dest = os.path.join(self.get_torch_location(conda_env), "utils/benchmark")
        if os.path.exists(dest):
            shutil.rmtree(dest)

        shutil.copytree(self._benchmark_utils_dir, dest)

    def make_scratch_dir_for(self, sha):
        scratch_dir = os.path.join(SCRATCH_ROOT, sha)

        for i in range(100):
            candidate = f"{scratch_dir}_{i}"
            if not os.path.exists(candidate):
                scratch_dir = candidate
                break
        else:
            raise ValueError(f"Too many prior attempts for {sha_or_branch}")

        os.makedirs(scratch_dir)
        return scratch_dir


class SubprocessDataPipe:
    def __init__(self):
        self._file = None
        self._results = []

    def set_file(self, file: str):
        self._file = file

    def push(self, item):
        self._results.append(item)

    def read(self):
        assert self._file is not None
        with open(self._file, "rb") as f:
            return pickle.load(f)

    def write(self):
        if self._file is not None:
            with open(self._file, "wb") as f:
                pickle.dump(self._results, f)


_SUBPROCESS_PIPE = SubprocessDataPipe()
atexit.register(_SUBPROCESS_PIPE.write)


def measure_counts():
    import multiprocessing.dummy
    from torch.utils.benchmark import Timer
    from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface

    from utils.make_jit_functions import make_globals

    # JIT Callgrind bindings. (If applicable.)
    timer_interface.wrapper_singleton()

    def map_fn(args):
        task_index, (setup, stmt) = args
        timer = Timer(
            stmt,
            setup=setup,
            globals=make_globals(stmt),
        )

        for backoff, timeout in [(15, 90), (30, 120), (None, 180)]:
            try:
                stats = timer.collect_callgrind(
                    CALLGRIND_LOOP_NUMBER,
                    timeout=timeout,
                    collect_baseline=False
                )
                break
            except subprocess.TimeoutExpired as e:
                print(f"Stmt: {stmt}\n{e}")
                if backoff is not None:
                    print(f"Sleeping to limit contention.")
                    time.sleep(backoff)

            except:
                if "# TS:" in stmt:
                    stats = None
                    break
                raise
        else:
            raise ValueError(f"Failed to collect stats for stmt: {stmt}")

        _SUBPROCESS_PIPE.push((task_index, stats))
        return task_index

    tasks = list(enumerate(TASKS)) * CALLGRIND_REPLICATES

    remaining = [CALLGRIND_REPLICATES for _ in TASKS]
    with multiprocessing.dummy.Pool(CALLGRIND_NUM_CORES) as pool:
        for i, task_index in enumerate(pool.imap(map_fn, tasks), 1):
            remaining[task_index] -= 1
            print(remaining)
            sys.stdout.flush()
    print()


def measure_times(task_index=None):
    from torch.utils.benchmark import Timer
    from utils.make_jit_functions import make_globals

    if task_index is None:
        task_indices = _SUBPROCESS_PIPE.read()
    else:
        task_indices = [task_index]

    for task_index in task_indices:
        setup, stmt = TASKS[task_index]
        g = make_globals(stmt)
        try:
            m = Timer(
                stmt,
                setup=setup,
                globals=g,
            ).blocked_autorange(min_run_time=WALL_TIME_SEC)

        except:
            if task_index is not None:
                raise
            m = None

        _SUBPROCESS_PIPE.push((task_index, m))


def merge_measurements():
    from torch.utils.benchmark import Measurement
    data_dir, results_path, simple_results_path = _SUBPROCESS_PIPE.read()

    with open(os.path.join(data_dir, "instruction_counts.pkl"), "rb") as f:
        counts = pickle.load(f)
    counts.sort(key=lambda x: x[0])

    times = [[] for _ in TASKS]
    for fname in os.listdir(data_dir):
        if fname.startswith("block"):
            with open(os.path.join(data_dir, fname), "rb") as f:
                for i, m in pickle.load(f):
                    times[i].append(m)

    with open(results_path, "wb") as f:
        pickle.dump([counts, times], f)

    simple_counts = []
    for _, c in counts:
        simple_counts.append(c.counts(denoise=True))

    simple_times = []
    for t in times:
        if any(ti is None for ti in t):
            simple_times.append(None)
        else:
            m = Measurement.merge(t)
            assert len(m) == 1
            simple_times.append(m[0].median)

    with open(simple_results_path, "wb") as f:
        pickle.dump([simple_counts, simple_times], f)


def debug():
    threshold = 0.1

    builder = build_pytorch.PytorchBuildHelper(
        root_dir=WORKSPACE_ROOT,
        clean=False,
        soft_clean=False,
        main_loop=False,
    )
    runner = Runner(builder)

    newline = "\n"
    lines = [[]]
    count = 0
    partition_indices = [-1]
    for n, task_name in _TASK_GROUPS:
        lines[-1].append(f"<H3><u>{task_name}</u></H3>")
        for _ in range(n):
            lines[-1].append(f"<b>[{count}]</b>&nbsp; {TASKS[count][1]}&nbsp<br>")
            count += 1
        if len(lines[-1]) >= 10:
            lines.append([])
        partition_indices.append(partition_indices[-1] + n)
    if not lines[-1]:
        lines.pop()
    lines = [['<td style="vertical-align:top">'] + l + ["</td>\n"] for l in lines]
    lines = list(it.chain(*lines))
    partition_indices = set(partition_indices[1:])

    results = runner.segment_results()
    filtered_results = []
    for r in results:
        if any(i >= threshold for i in r["Count deltas"]):
            filtered_results.append(r)

    padding = ["<td></td>"] * 4
    result_lines = ["<tr>"] + padding
    for i in range(len(TASKS)):
        result_lines.append(f'<th style="text-align:right">[{i}]</th>')
        if i in partition_indices:
            result_lines.append("<td></td>")
    result_lines.extend(["</tr>", "<tr>"] + padding)
    for n, task_name in _TASK_GROUPS:
        result_lines.extend([
            f'<td colspan="{n}" style="text-align:center;border-bottom: 1px solid white">',
            f"&nbsp;&nbsp;<b>{task_name}</b>",
            '</td><td style="text-align:right;border-bottom: 1px solid white">&nbsp;:</td>',
        ])
    result_lines.append("</tr>")


    color_codes = [
        ("LightGreen", -threshold, True),
        ("ForestGreen", -0.25 * threshold, False),
        ("DarkGreen", 0, False),
        ("Maroon", 0.25 * threshold, False),
        ("FireBrick", threshold, False),
        ("Red", None, True)
    ]
    def color_value(x, y, max_abs_delta, border=False):
        bold = False
        if x is None or y is None:
            color = "White"
            s = "--"
        elif abs(x) < 0.05 * threshold or abs(x) < max_abs_delta * 0.01:
            color = "Black"
            s = ""
        else:
            for color, color_threshold, bold in color_codes:
                if color_threshold is None or y <= color_threshold:
                    break
            s = f"{y * 100:.1f}"
        if bold:
            s = f"<b>{s}</b>"
        border_str = ";border-bottom: 1px dashed white" if border else ""
        return f'<td style="text-align:right;color:{color}{border_str}">{s}</td>'

    def sha_to_url(sha):
        return f'<a href="https://github.com/pytorch/pytorch/commit/{sha}" style="color:White">{sha[:7]}</a>'

    for r in filtered_results[::-1]:
        max_abs_delta = max(abs(i) for i in r["Count deltas"] if i is not None)
        for key in ("Count deltas", "Time deltas"):
            string_counts = []
            if key == "Count deltas":
                n_intermediate = len(r['Intermediate SHAs']) + 1
                if n_intermediate == 1:
                    msg = r["Messages"][1].strip()
                    if len(msg) > 20:
                        skew = len(msg)
                        best_index = 0
                        words = msg.split()
                        for i in range(1, len(words)):
                            top, bottom = " ".join(words[:i]), " ".join(words[i:])
                            if abs(len(top) - len(bottom)) < skew:
                                skew = abs(len(top) - len(bottom))
                                best_index = i
                        msg = (
                            "&nbsp;" * 3 + " ".join(words[:best_index]) + "<br>" +
                            "&nbsp;" * 3 + " ".join(words[best_index:])
                        )
                    else:
                        msg = "&nbsp;" * 3 + msg
                    date_str = r["Dates"][1].strftime("%m/%d")
                else:
                    n_intermediate_str = str(n_intermediate)
                    n_intermediate_str = "&nbsp;" * (4 - len(n_intermediate_str)) + n_intermediate_str
                    msg = f"{'&nbsp;' * 3}{n_intermediate_str}&nbsp; commits{'&nbsp;' * 3}"
                    d0, d1 = [d.strftime("%m/%d") for d in r["Dates"]]
                    date_str = f"{d0} - {d1}"


                td_template = '<td rowspan="2" style="text-align:{align};border-bottom: 1px dashed white">'
                string_counts.extend([
                    td_template.format(align="right"),
                    f"{sha_to_url(r['SHAs'][0])}<br>{sha_to_url(r['SHAs'][1])}",
                    "</td>",
                    f'{td_template.format(align="left")}{msg}</td>',
                    f'{td_template.format(align="center")}{date_str}</td>',
                    f'{td_template.format(align="center")}\u0394C&nbsp; :<br>\u0394T&nbsp; :</td>',
                ])
            for i, (x, y) in enumerate(zip(r["Count deltas"], r[key])):
                string_counts.append(color_value(x, y, max_abs_delta, key == "Time deltas"))
                if i in partition_indices:
                    string_counts.append('<td style="text-align:right">&nbsp;:</td>')
            result_lines.append(
                "\n".join(["<tr>"] + string_counts + ["</tr>\n"])
            )

    body = textwrap.dedent(f"""\
    <HTML>
      <body style="background-color:Black;color:WhiteSmoke">

      <table style="width:130%">
{textwrap.indent(newline.join(lines), ' ' * 6)}
      </table>

      <br><br><br>

      <table style="width:170%">
{textwrap.indent(newline.join(result_lines), ' ' * 6)}
      </table>
      <body>
    </HTML>
    """)

    # print(body)
    with open("/mnt/shared/taylorrobie/public_html/test.html", "wt") as f:
        f.write(body)


def main():
    builder = build_pytorch.PytorchBuildHelper(
        root_dir=WORKSPACE_ROOT,
        clean=False,
        soft_clean=False,
        main_loop=False,
    )

    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    runner = Runner(builder)
    runner.loop()


_MODES = {
    "main": main,
    "measure_counts": measure_counts,
    "measure_times": measure_times,
    "merge_measurements": merge_measurements,
    "debug": debug,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="main")
    args = parser.parse_args()
    if args.result_file is not None:
        _SUBPROCESS_PIPE.set_file(args.result_file)

    _MODES[args.mode]()
