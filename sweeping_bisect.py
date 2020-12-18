import atexit
import argparse
import collections
import datetime
import itertools as it
import json
import os
import math
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
COMPLETED_BUILD_ROOT = os.path.join(WORKSPACE_ROOT, "completed_builds")
RESULTS_ROOT = os.path.join(WORKSPACE_ROOT, "results")
BUILT_RECORD = os.path.join(WORKSPACE_ROOT, "built.json")
STOP_FILE = os.path.join(WORKSPACE_ROOT, "stop")  # touch this file to stop the build/test loop
SWEEP_START = "2019-06-01"
SWEEP_CADENCE = 3  # days

# Loop is now in fine-tuning mode.
IGNORE_BEFORE = "2020-11-01"


NUM_CORES = multiprocessing.cpu_count()
assert NUM_CORES in (80,)
if NUM_CORES == 80:
    CORES_PER_WORK_UNIT = 8
    CALLGRIND_EXTRA_CORES = 4
    NUM_CONCURRENT_WORK_UNITS = 8

MAX_CALLGRIND_WORKERS = math.ceil(NUM_CONCURRENT_WORK_UNITS / 2) + 1

BISECTION_QUEUE_LEN = 5
BISECTION_THRESHOLDS = (0.25, 0.1, 0.05, 0.025, 0.01)
FILTER_THRESHOLDS = (
    (1,  5.00 / 100),
    (2,  2.50 / 100),
    (3,  1.00 / 100),
    (5,  0.50 / 100),
    (10, 0.25 / 100)
)
REPORT_THREHOLDS = FILTER_THRESHOLDS[:3]

CALLGRIND_REPLICATES = 3
CALLGRIND_LOOP_NUMBER = 10000
WALL_TIME_SEC = 10

PASS = "pass"
SCALAR_X = "x = torch.ones((1,))"
SCALAR_XY = "x = torch.ones((1,));y = torch.ones((1,))"
TWO_BY_TWO_XY = "x = torch.ones((2, 2)); y = torch.ones((2, 2))"
SMALL_X = "x = torch.ones((10,))"
TEN_BY_TEN_X = "x = torch.ones((10, 10))"
SCALAR_INDEX_SETUP = "v = torch.randn(1,1,1)"
VECTOR_INDEX_SETUP = "a = torch.zeros(100, 100, 1, 1, 1); b = torch.arange(99, -1, -1).long()"
SCALAR_WEIGHT = "w = torch.ones((1,), requires_grad=True)"
_TASKS = (
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
        (SCALAR_X, SCALAR_WEIGHT, "y = x + w"),
        "y.backward()"
    ),
    (
        (SCALAR_X, SCALAR_WEIGHT),
        ("y = x * w", "y.backward()"),
    ),
    (
        (SCALAR_X, SCALAR_WEIGHT, "w1 = torch.ones((1,), requires_grad=True)"),
        ("y = torch.nn.functional.relu(x * w) * w1", "y.backward()"),
    ),

    # Autograd + TorchScript
    (
        "x = torch.ones((1,)) + torch.ones((1,), requires_grad=True)",
        "model(x).backward()  # TS: script fn `x + 1`"
    ),
)


CPP_ANALOGS = {
    PASS: "//",  # FIXME
    SCALAR_X: "torch::Tensor x = torch::ones({1});",
    SCALAR_INDEX_SETUP: "torch::Tensor v = torch::randn({1, 1, 1});",
    SCALAR_XY: "torch::Tensor x = torch::ones({1});\ntorch::Tensor y = torch::ones({1});",
    TWO_BY_TWO_XY: "torch::Tensor x = torch::ones({2, 2});\ntorch::Tensor y = torch::ones({2, 2});",
    SMALL_X: "torch::Tensor x = torch::ones({10});",
    TEN_BY_TEN_X: "torch::Tensor x = torch::ones({10, 10});",
    "torch.empty(())": "torch::empty({0});",
    "torch.empty((1,))": "torch::empty({1});",
    "torch.mm(x, y)": "torch::mm(x, y);",
    "torch.matmul(x, y)": "torch::matmul(x, y);",
    "x.size()[0]": "x.sizes()[0];",
    "torch.as_strided(x, (4, 3), (10, 3), 6)": "torch::as_strided(x, {4, 3}, {10, 3}, 6);",
    "x.view(-1, 2)": "x.view({-1, 2});",

    # FIXME: Valid after `36919278cc82cac952a196c094324c9dcb710214`
    # "v[...] = 1": 'v.index({"..."}) = 1;',
    # "v[:] = 1": "v.index({torch::indexing::Slice()}) = 1;",

    SCALAR_WEIGHT: (
        "torch::Tensor w = torch::ones({1});\n"
        "w.requires_grad_();"
    ),
    "w1 = torch.ones((1,), requires_grad=True)": (
        "torch::Tensor w1 = torch::ones({1});\n"
        "w1.requires_grad_();"
    ),
    "y = x + w": "torch::Tensor y = x + w;",
    "y = x * w": "torch::Tensor y = x * w;",
    "y = torch.nn.functional.relu(x * w) * w1": (
        "torch::Tensor xw = x * w;\n"
        "torch::Tensor y = torch::nn::functional::relu(xw) * w1;"
    ),

}
JUST_ADD_SEMICOLON = (
    "x += 1",
    "x *= 1",
    "x.sum()",
    "x == y",
    "x.uniform_()",
    "v[0] = 1",
    "v[0, 0] = 1",
    "v[0, 0, 0] = 1",
    "x.zero_()",
    "x.resize_(0)",
    "x.copy_(y)",
    "x.stride(0)",
    "x.unsqueeze(0)",
    "x.contiguous()",
    "x.t().contiguous()",
    "x.clone()",
    "x.select(1, 4)",
    "y.backward()",
)
CPP_ANALOGS.update({i: i + ";" for i in JUST_ADD_SEMICOLON})

NO_CPP_ANALOG = (
    "v[...] = 1",
    "v[:] = 1",
    "v[None] = 1",
    "v[False] = 1",
    "v[True] = 1",
    "a[b, None, ..., :, True] = 1",
    "model(x)  # TS: script fn `x + 1`",
    "model(x)  # TS: script Module `x + 1`",
    "model(x).backward()  # TS: script fn `x + 1`",
)

KNOWN_SUSPECT = (
    # The move from TH to ATen helped a lot.
    (None, "2020-07-09", "C++", 5, "torch::mm(x, y);"),
    (None, "2020-07-09", "C++", 6, "torch::matmul(x, y);"),
    (None, "2019-10-27", "C++", 18, "x.zero_();"),

    # This comes from the C++ flakiness
    (None, "2020-07-09", "Python", 5, "torch.mm(x, y)"),
    (None, "2020-07-09", "Python", 6, "torch.matmul(x, y)"),
    (None, "2019-10-27", "Python", 18, "x.zero_()"),

    ("2019-09-03", "2019-09-13", "Python", 7, "x == y"),
    (None, None, "Python", 15, "v[False] = 1"),
    ("2019-07-23", "2019-07-24", "Python", 22, "x.clone()"),

    (None, None, "Python", 24, "x.size()[0]"),
    (None, "2019-09-13", "Python", 26, "torch.as_strided(x, (4, 3), (10, 3), 6)"),
    (None, "2019-11-14", "Python", 27, "x.select(1, 4)"),
    (None, "2019-11-14", "Python", 28, "x.view(-1, 2)"),
    (None, "2019-11-14", "Python", 29, "x.unsqueeze(0)"),
)

def _to_cpp(setup, stmt):
    def code_to_cpp(x):
        if isinstance(x, str):
            return CPP_ANALOGS.get(x)
        assert isinstance(x, tuple)
        if any(i not in CPP_ANALOGS for i in x):
            return
        return "\n".join([CPP_ANALOGS[i] for i in x])
    cpp_setup = code_to_cpp(setup)
    cpp_stmt = code_to_cpp(stmt)
    if cpp_setup is None or cpp_stmt is None:
        assert stmt in NO_CPP_ANALOG
        return None, None
    assert stmt not in NO_CPP_ANALOG
    return cpp_setup, cpp_stmt

CPP_TASKS = tuple(_to_cpp(setup, stmt) for setup, stmt in _TASKS)
TASKS = tuple(
    (
        setup if isinstance(setup, str) else "\n".join(setup),
        stmt if isinstance(stmt, str) else "\n".join(stmt),
    )
    for setup, stmt in _TASKS
)


_TASK_GROUPS = [
    (2, "Allocation"),
    (7, "Math"),
    (9, "Indexing"),
    (6, "Data movement"),
    (6, "Metadata & views"),
    (2, "TorchScript"),
    (4, "AutoGrad (+TS)"),
]
assert sum(i[0] for i in _TASK_GROUPS) == len(TASKS)


class RunnerState:
    def __init__(self):
        self._state_path = os.path.join(WORKSPACE_ROOT, "state.json")
        self._lock = threading.Lock()

        self.build_queue = collections.deque()
        self.built = {}
        self.test_queue = set()
        self.in_progress = set()
        self.finished = {}

        if os.path.exists(self._state_path):
            with open(self._state_path, "rt") as f:
                state = json.load(f)

            self.built = state["built"]
            self.finished = state["finished"]
            for sha in self.built:
                if sha not in self.finished:
                    self.test_queue.add(sha)

    def to_json(self):
        with open(self._state_path, "wt") as f:
            json.dump({
                "built": self.built,
                "finished": self.finished,
            }, f)

    def maybe_enqueue_build(self, sha):
        with self._lock:
            enqueue = (
                sha not in self.in_progress and
                sha not in self.finished and
                sha not in self.built and
                sha not in self.build_queue
            )
            if enqueue:
                self.build_queue.append(sha)

    def get_build_job(self):
        with self._lock:
            if self.build_queue:
                output = self.build_queue.popleft()
                self.in_progress.add(output)
                return output

    def report_build_finished(self, sha, env_path):
        with self._lock:
            self.built[sha] = env_path
            self.test_queue.add(sha)
            self.in_progress.remove(sha)
            self.to_json()

    def get_test_job(self):
        with self._lock:
            if self.test_queue:
                return self.test_queue.pop()

    def report_test_finished(self, sha, result):
        with self._lock:
            self.finished[sha] = result
            self.to_json()


class Runner:
    def __init__(self, pytorch_builder: build_pytorch.PytorchBuildHelper) -> None:
        # Reference env, and source of benchmark_utils
        timer_env_record = os.path.join(WORKSPACE_ROOT, "timer_env.txt")
        if not os.path.exists(timer_env_record):
            timer_env = pytorch_builder.build_clean(
                checkout="gh/taylorrobie/callgrind_backtest",
                show_progress=True,

                # Build tends to OOM. Should investigate...
                max_jobs=max(int(NUM_CORES * 0.75), 1),
            )
            assert timer_env is not None
            with open(timer_env_record, "wt") as f:
                f.write(timer_env)

        with open(timer_env_record, "rt") as f:
            self.timer_env = f.read().strip()

        self._pytorch_builder = pytorch_builder
        self._history = pytorch_builder.get_history_since(SWEEP_START)
        self._sha_to_date_str = {
            sha: date.strftime(r"%Y_%m_%d")
            for sha, date, _, _, _ in self._history
        }

        self._timer_env_torch_dir = self.get_torch_location(self.timer_env)
        self._benchmark_utils_dir = os.path.join(
            self._timer_env_torch_dir, "utils/benchmark")

        self.state = RunnerState()
        self._stop = False
        self._active_workers = 0

        # Initial sweep
        self._initial_sweep_indices = [0]
        for i, (_, date, _, _, _) in enumerate(self._history):
            if (date - self._history[self._initial_sweep_indices[-1]][1]).days >= SWEEP_CADENCE:
                self._initial_sweep_indices.append(i)
        if self._initial_sweep_indices[-1] != len(self._history) - 1:
            self._initial_sweep_indices.append(len(self._history) - 1)
        for i in self._initial_sweep_indices:
            sha = self._history[i][0]
            self.state.maybe_enqueue_build(sha)

    def loop(self):
        num_workers = 1 + NUM_CONCURRENT_WORK_UNITS
        with multiprocessing.dummy.Pool(num_workers) as pool:
            pool.map(self.loop_fn, range(num_workers), 1)

    def loop_fn(self, worker_id):
        if worker_id > 4:
            time.sleep(100 * (worker_id - 4))

        if not worker_id:
            # bisect loop
            initial_sweep_complete = False
            initial_shas = [self._history[i][0]for i in self._initial_sweep_indices]
            status = ""
            while not self._stop:
                initial_sweep_complete = initial_sweep_complete or (sum(
                    1 for sha in initial_shas
                    if not (sha in self.state.finished or self._pytorch_builder.unbuildable(sha))
                ) <= 3)

                if initial_sweep_complete:
                    self.bisect()

                    if not self._active_workers and not self.state.build_queue and not self.state.test_queue:
                        print("No work. Stopping.")
                        self._stop = True
                        break

                time.sleep(60)

                status_new = (
                    f"{len(self.state.built)} Built, "
                    f"{len(self.state.finished)} Finished, "
                    f" Queues: (build): {len(self.state.build_queue)}, "
                    f"(test): {len(self.state.test_queue)}"
                )

                if status_new != status:
                    status = status_new
                    print(status)

                if os.path.exists(STOP_FILE):
                    print("Stop registered.")
                    os.remove(STOP_FILE)
                    self._stop = True

        elif worker_id <= MAX_CALLGRIND_WORKERS:
            while not self._stop:
                self._active_workers += 1
                sha = self.state.get_test_job()
                if sha is not None:
                    scratch_dir = self.make_scratch_dir_for(sha)
                    try:
                        self.collect_counts(sha, scratch_dir)
                    finally:
                        shutil.rmtree(scratch_dir)
                else:
                    # Work stealing
                    self.build()
                self._active_workers -= 1
                time.sleep(random.random() * 30)

        else:
            while not self._stop:
                self._active_workers += 1
                supply = len(self.state.test_queue) + max(len(self.state.in_progress) - 2, 0)
                if supply < 4 * MAX_CALLGRIND_WORKERS:
                    self.build()
                self._active_workers -= 1
                time.sleep(random.random() * 20)

    def build(self):
        sha = self.state.get_build_job()
        if sha is None:
            return

        print(f"Building: {sha}")
        start_time = time.time()
        conda_env = self._pytorch_builder.build_clean(
            sha,
            show_progress=False,
            nice="19",
            max_jobs=str(CORES_PER_WORK_UNIT),
        )

        if conda_env is None:
            return  # unbuildable

        new_env_path = os.path.join(
            COMPLETED_BUILD_ROOT,
            f"{self._sha_to_date_str[sha]}_{sha}"
        )
        shutil.move(conda_env, new_env_path)
        del conda_env

        self.state.report_build_finished(sha, new_env_path)
        print(f"Build time: {sha} {time.time() - start_time:.0f} sec")

    def collect_counts(self, sha, scratch_dir):
        print(f"Begin measure (counts): {sha}")

        start_time = time.time()
        conda_env = self.state.built[sha]
        self.monkey_patch_benchmark_utils(conda_env)
        counts_path = os.path.join(scratch_dir, "instruction_counts.pkl")

        def per_line_fn(l):
            if "TimeoutExpired" in l:
                print(l)

        self._pytorch_builder.subprocess_call(
            "conda install -y -c conda-forge valgrind",
            shell=True,
            check=True,
            conda_env=conda_env,
        )

        n_retry = 2
        for i in range(n_retry):
            retcode = self._pytorch_builder.subprocess_call(
                f" python -u {os.path.abspath(__file__)} "
                f"--mode measure_counts --result_file {counts_path}",
                shell=True,
                check=(i + 1) == n_retry,
                conda_env=conda_env,
                timeout=3600,
                per_line_fn=per_line_fn,
                env={
                    "CPLUS_INCLUDE_PATH": os.path.join(conda_env, "include"),
                    "TOGGLE_CALLGRIND_PATH": self._timer_env_torch_dir,
                    "LD_LIBRARY_PATH": os.path.join(conda_env, "lib"),
                }
            )
            if not retcode:
                break

        merge_path = os.path.join(scratch_dir, "merge.pkl")
        result_path = os.path.join(RESULTS_ROOT, f"counts_{sha}.pkl")
        result_path_simple = os.path.join(RESULTS_ROOT, f"counts_{sha}_simple.pkl")
        with open(merge_path, "wb") as f:
            pickle.dump([
                scratch_dir,
                result_path,
                result_path_simple,
            ], f)

        self._pytorch_builder.subprocess_call(
            f"python -u {os.path.abspath(__file__)} "
            f"--mode post_process --result_file {merge_path}",
            shell=True,
            check=True,
            conda_env=conda_env
        )

        self.state.report_test_finished(sha, [result_path, result_path_simple])

        print(f"Counts time: {sha} {time.time() - start_time:.0f} sec")

    def bisect(self):
        segment_results, _, _ = self.segment_results(mask_suspect=True)
        if not segment_results:
            return

        for sha in self._bisect_unbuildable():
            self.state.maybe_enqueue_build(sha)

        for threshold in BISECTION_THRESHOLDS:
            for r in segment_results[::-1]:
                if r["Num unbuildable intermediates"]:
                    continue

                if IGNORE_BEFORE and r["Dates"][0] < datetime.datetime.strptime(IGNORE_BEFORE, build_pytorch.DATE_FMT):
                    continue

                max_abs_delta = max(
                    abs(i) for i in r["Count deltas (Python)"] + r["Count deltas (C++)"]
                    if i is not None
                )
                intermediate_shas = r["Intermediate SHAs"]
                if max_abs_delta >= threshold and intermediate_shas:
                    self.state.maybe_enqueue_build(
                        intermediate_shas[int(len(intermediate_shas) // 2)]
                    )
                if len(self.state.build_queue) >= BISECTION_QUEUE_LEN:
                    return

    def _bisect_unbuildable(self):
        # There are a couple regions that are unbuildable. A normal bisect
        # would wastefully try every SHA in the broken region, so we do
        # a separate higher priority bisect for unbuildable commits.
        commit_ranges = [[None, None, []]]
        def buildable(sha):
            return not self._pytorch_builder.unbuildable(sha)

        for sha, _, _, _, _ in self._history:
            if sha in self.state.finished or not buildable(sha):
                commit_ranges[-1][1] = sha
                commit_ranges.append([sha, None, []])
            else:
                commit_ranges[-1][2].append(sha)

        output = []
        for sha_0, sha_1, intermediate_shas in commit_ranges[1:-1]:
            if buildable(sha_0) != buildable(sha_1) and intermediate_shas:
                output.append(intermediate_shas[int(len(intermediate_shas) // 2)])
        return output

    def segment_results(self, assert_cpp=False, mask_suspect=False):
        finished_results = []
        lower_sha = None
        intermediate_shas = {lower_sha: []}
        n_unbuildable_intermediates = collections.Counter()
        for sha, date, _, _, msg in self._history:
            if self._pytorch_builder.unbuildable(sha):
                n_unbuildable_intermediates[lower_sha] += 1

            elif sha in self.state.finished:
                finished_results.append((sha, date, msg, self.state.finished[sha][1]))
                lower_sha = sha
                intermediate_shas[sha] = []

            else:
                intermediate_shas[lower_sha].append(sha)

        if not finished_results:
            return [], None

        results = []
        march_27 = datetime.datetime.strptime("2020-03-27", build_pytorch.DATE_FMT)

        cull_from_pairwise = collections.defaultdict(list)
        for sha, date, msg, results_path in finished_results:
            with open(results_path, "rb") as f:
                c = pickle.load(f)
                c_python = [
                    int(statistics.median(ci)) if ci else None
                    for ci in c["Python"]
                ]
                for i, ci in enumerate(c["C++"]):
                    nondeterminism_expected = (
                        "backward" in (CPP_TASKS[i][1] or "") and
                        (date - march_27).total_seconds() < 0
                    ) or not assert_cpp
                    assert not ci or len(set(ci)) == 1 or nondeterminism_expected
                c_cpp = [int(statistics.median(ci)) if ci else None for ci in c["C++"]]

                if mask_suspect:
                    for d0, d1, lang, index, stmt in KNOWN_SUSPECT:
                        assert lang in ("Python", "C++")
                        lang_tasks = TASKS if lang == "Python" else CPP_TASKS
                        c_lang = c_python if lang == "Python" else c_cpp
                        assert lang_tasks[index][1] == stmt  # Prevent drift from tasks changing.
                        d0 = datetime.datetime.strptime(d0 or "1970-01-01", build_pytorch.DATE_FMT)
                        d1 = datetime.datetime.strptime(d1 or "2050-01-01", build_pytorch.DATE_FMT)
                        if date >= d0 and date <= d1:
                            cull_from_pairwise[(sha, lang)].append(index)

                results.append((sha, date, msg, c_python, c_cpp))

        def low_water_mark(values):
            return [
                min(k for k in j if k is not None)
                if any(k for k in j if k is not None)
                else None
                for j in zip(*values)
            ]

        def deltas(x0, x1, lwm):
            return [
                (x1_i - x0_i) / lwm_i
                if (x0_i is not None and x1_i is not None)
                else None
                for x0_i, x1_i, lwm_i in zip(x0, x1, lwm)
            ]

        count_lwm_python = low_water_mark([i[3] for i in results])
        count_lwm_cpp = low_water_mark([i[4] for i in results])

        for sha, _, _, c_python, c_cpp in results:
            for i in cull_from_pairwise[(sha, "Python")]:
                c_python[i] = None

            for i in cull_from_pairwise[(sha, "C++")]:
                c_cpp[i] = None

        output = []
        for r0, r1 in zip(results[:-1], results[1:]):
            sha_0, date_0, msg_0, c0_python, c0_cpp = r0
            sha_1, date_1, msg_1, c1_python, c1_cpp = r1
            output.append({
                "SHAs": (sha_0, sha_1),
                "Intermediate SHAs": intermediate_shas[sha_0],
                "Dates": (date_0, date_1),

                "Counts (Python)": (c0_python, c1_python),
                "Count deltas (Python)": deltas(c0_python, c1_python, count_lwm_python),

                "Counts (C++)": (c0_cpp, c1_cpp),
                "Count deltas (C++)": deltas(c0_cpp, c1_cpp, count_lwm_cpp),

                "Messages": (msg_0, msg_1),
                "Num unbuildable intermediates": n_unbuildable_intermediates[sha_0],
            })
        return output, count_lwm_python, count_lwm_cpp

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

    def timing_loop(self):
        results, _, _ = self.segment_results(mask_suspect=True)

        measure_shas = set()
        for r in results:
            abs_deltas = [
                max(abs(i) for i in (i_p, i_c) if i is not None)
                for i_p, i_c in zip(r["Count deltas (Python)"], r["Count deltas (C++)"])
                if i_p is not None or i_c is not None
            ]

            measure = False
            for n, threshold in FILTER_THRESHOLDS:
                measure |= sum(1 for d in abs_deltas if d >= threshold * 1.5) >= n

            if measure:
                measure_shas.add(r["SHAs"][0])
                measure_shas.add(r["SHAs"][1])
        measure_shas = tuple(measure_shas)

        staging_area = os.path.join(SCRATCH_ROOT, "intermediate_timing")
        if os.path.exists(staging_area):
            shutil.rmtree(staging_area)
        os.makedirs(staging_area)

        time_dirs = {}
        for sha in measure_shas:
            time_dirs[sha] = os.path.join(RESULTS_ROOT, f"times_{sha}")
            os.makedirs(time_dirs[sha], exist_ok=True)

        work_queue = queue.Queue()
        block_size = 6
        reserved_cores = 4
        def map_fn(worker_id):
            if not worker_id:
                generation = -1
                for _ in range(50):
                    while work_queue.qsize() > len(measure_shas) * 2:
                        time.sleep(30)

                    generation += 1
                    generation_tasks = []
                    for sha in measure_shas:
                        task_indices = list(range(len(TASKS)))
                        random.shuffle(task_indices)
                        task_groups = [[]]
                        for task_index in task_indices:
                            task_groups[-1].append(task_index)
                            if len(task_groups[-1]) == block_size and task_index != task_indices[-1]:
                                task_groups.append([])
                        for g in task_groups:
                            generation_tasks.append((generation, sha, tuple(g)))
                    random.shuffle(generation_tasks)
                    for i in generation_tasks:
                        work_queue.put(i)
            elif worker_id == 1:
                while not self._stop:
                    self._pytorch_builder.subprocess_call(
                        "sudo /usr/local/fbprojects/dynamoserver/bin/turboDriver disable",
                        shell=True,
                        check=True,
                    )
                    time.sleep(20)
            else:
                # Reserve the first four cores for the runner.
                worker_core = worker_id + (reserved_cores - 2)
                assert worker_core < NUM_CORES

                while not self._stop:
                    try:
                        generation, sha, task_indices = work_queue.get(timeout=10)
                    except queue.Empty:
                        break

                    indices_str = ",".join([str(i) for i in task_indices])
                    output_path = os.path.join(staging_area, f"{sha}__{generation}__{indices_str}.pkl")
                    with open(output_path, "wb") as f:
                        pickle.dump(task_indices, f)

                    conda_env = self.state.built[sha]
                    self._pytorch_builder.subprocess_call(
                        f"taskset --cpu-list {worker_core} "
                        f"python -u {os.path.abspath(__file__)} "
                        f"--mode measure_times --result_file {output_path}",
                        shell=True,
                        check=True,
                        conda_env=conda_env,
                    )

        num_workers = NUM_CORES - reserved_cores
        with multiprocessing.dummy.Pool(num_workers) as pool:
            pool.map(map_fn, range(num_workers))


class SubprocessDataPipe:
    def __init__(self):
        self._file = None
        self._results = []

    def set_file(self, file: str):
        self._file = file

    def push(self, item):
        self._results.append(item)

    def read(self):
        assert self._file is not None or self._results
        if self._file:
            with open(self._file, "rb") as f:
                return pickle.load(f)
        return self._results.copy()

    def write(self):
        if self._file is not None:
            with open(self._file, "wb") as f:
                pickle.dump(self._results, f)


_SUBPROCESS_PIPE = SubprocessDataPipe()
atexit.register(_SUBPROCESS_PIPE.write)


def measure_counts():
    import multiprocessing.dummy
    from torch.utils.benchmark import Timer

    from utils.make_jit_functions import make_globals

    # JIT Callgrind bindings (if applicable) and establish baseline.
    _ = Timer().collect_callgrind()

    failures = []
    def map_fn(args):
        task_index, language, setup, stmt = args
        if stmt is None:
            assert language == "C++"
            return task_index, language, True

        try:
            timer = Timer(
                stmt,
                setup=setup,
                globals=make_globals(stmt),
                lang=language,
            )

            stats = timer.collect_callgrind(
                CALLGRIND_LOOP_NUMBER,
                timeout=300,
            )

        except subprocess.TimeoutExpired as e:
            print(f"Stmt: {stmt}\n{e}")
            failures.append(args)
            time.sleep(10)
            return task_index, language, False

        except:
            stats = None
            allowed_failure = False
            if "# TS:" in stmt:
                # Historic TorchScript can be kind of brittle.
                allowed_failure = True

            if language == "C++" and "backward" in stmt:
                # Some APIs (e.g. `requires_grad_` are not present in old versions.)
                allowed_failure = True

            if not allowed_failure:
                raise ValueError(f"Failed to collect stats for stmt: {stmt}")

        _SUBPROCESS_PIPE.push((task_index, language, stats))
        return task_index, language, True

    python_tasks = [(task_index, "Python", setup, stmt) for task_index, (setup, stmt) in enumerate(TASKS)]
    cpp_tasks = [(task_index, "C++", setup, stmt) for task_index, (setup, stmt) in enumerate(CPP_TASKS)]

    tasks = (cpp_tasks + python_tasks) * CALLGRIND_REPLICATES
    random.shuffle(tasks)

    remaining = [[CALLGRIND_REPLICATES, CALLGRIND_REPLICATES] for _ in TASKS]
    with multiprocessing.dummy.Pool(CORES_PER_WORK_UNIT + CALLGRIND_EXTRA_CORES) as pool:
        for _ in range(3):
            for i, (task_index, language, success) in enumerate(pool.imap(map_fn, tasks), 1):
                if success:
                    remaining[task_index][0 if language == "Python" else 1] -= 1
                n_remaining = sum(it.chain(*remaining))
                print(", ".join([f"{r_p} {r_c}" for r_p, r_c in remaining]), "  ", n_remaining)
                sys.stdout.flush()

            if failures:
                tasks = failures.copy()
                failures.clear()
            else:
                break
        if any(i > 1 for i in it.chain(*remaining)):
            raise ValueError("Too many failures.")

    print()


def measure_times(task_index=None):
    from torch.utils.benchmark import Timer
    from utils.make_jit_functions import make_globals

    is_manual_run = (task_index is not None)
    task_indices = (
        [task_index] if is_manual_run
        else _SUBPROCESS_PIPE.read())

    for task_index in task_indices:
        setup, stmt = TASKS[task_index]
        try:
            g = make_globals(stmt)
            m = Timer(
                stmt,
                setup=setup,
                globals=g,
            ).blocked_autorange(min_run_time=WALL_TIME_SEC)

        except:
            if is_manual_run:
                raise
            m = None

        _SUBPROCESS_PIPE.push((task_index, m))


def post_process():
    from torch.utils.benchmark import Measurement

    data_dir, results_path, simple_results_path = _SUBPROCESS_PIPE.read()
    with open(os.path.join(data_dir, "instruction_counts.pkl"), "rb") as f:
        raw_counts = pickle.load(f)

    counts, simple_counts = [{
        "Python": [[] for _ in TASKS],
        "C++": [[] for _ in TASKS],
    } for _ in range(2)]

    for task_index, language, stats in raw_counts:
        if stats is not None:
            counts[language][task_index].append(stats)
            simple_counts[language][task_index].append(stats.counts(denoise=True))

    with open(results_path, "wb") as f:
        pickle.dump(counts, f)

    with open(simple_results_path, "wb") as f:
        pickle.dump(simple_counts, f)


def compute_soft_deltas():
    result_paths = _SUBPROCESS_PIPE.read()

    def soft_delta(counts_0, counts_1):
        def group(counts, inclusive=False):
            grouped = [[] for _ in TASKS]
            for i, c in counts:
                grouped[i].append(c)

            output = []
            for counts in grouped:
                if any(i is None for i in counts):
                    output.append(None)
                    continue
                c = counts[0].as_standardized().stats(inclusive=inclusive).denoise()
                for i in counts[1:]:
                    c = c + i.as_standardized().stats(inclusive=inclusive).denoise()
                output.append(c)
            return output

        grouped_counts_0 = group(counts_0)
        grouped_counts_1 = group(counts_1)

        strict = 0.0005  # 0.05%
        lax = 0.001  # 0.1%
        very_lax = 0.005  # 0.2%
        suspect_symbols = (
            (very_lax, "obmalloc.c:_PyObject_Free"),

            (lax, "dictobject.c:_PyDict_LoadGlobal"),
            (lax, "dictobject.c:PyDict_GetItem"),
            (lax, "obmalloc.c:unicode_dealloc"),
            (lax, "obmalloc.c:PyObject_Malloc"),
            (lax, "typeobject.c:assign_version_tag"),
            (lax, "_PyType_Lookup"),
            (lax, "_PyDict_GetItem_KnownHash"),
            (lax, "PyErr_Occurred"),

            (strict, "???:malloc_consolidate"),
            (strict, "???:unlink_chunk"),
            (strict, "???:_int_free"),
            (strict, "???:_int_malloc"),
            (strict, "???:malloc")
        )

        output = []
        for i, (c0, c1) in enumerate(zip(grouped_counts_0, grouped_counts_1)):
            if c0 is None or c1 is None:
                output.append(None)
                continue
            mean_size = (c1.sum() + c0.sum()) / 2
            delta = c1 - c0

            # Efficiency finesse to limit the amout of checks.
            delta_no_torch = delta.filter(lambda l: "torch" not in l)
            for cull_threshold, s in suspect_symbols:
                n_suspect = delta_no_torch.filter(lambda l: s in l).sum()
                if n_suspect / mean_size < cull_threshold:
                    delta = delta.filter(lambda l: s not in l)

            output.append(delta.sum())

        return output

    for path_0, path_1 in result_paths:
        with open(path_0, "rb") as f:
            counts_0 = pickle.load(f)

        with open(path_1, "rb") as f:
            counts_1 = pickle.load(f)

        _SUBPROCESS_PIPE.push(soft_delta(counts_0, counts_1))


def cull():
    """Dangerous. Should not be run when the main loop is running."""
    builder = build_pytorch.PytorchBuildHelper(
        root_dir=WORKSPACE_ROOT,
        clean=False,
        soft_clean=False,
        main_loop=False,
    )

    runner = Runner(builder)
    results, _, _ = runner.segment_results(mask_suspect=True)
    shas = [r["SHAs"] for r in results]

    # Culling the initial sweep is pointless as it would just be re-run.
    protected_shas = {
        runner._history[i][0]
        for i in runner._initial_sweep_indices
    }

    for r in results:
        if r["Num unbuildable intermediates"]:
            protected_shas.add(r["SHAs"][0])
            protected_shas.add(r["SHAs"][1])

    shas_to_cull = set()
    for s0, s1 in shas:
        shas_to_cull.add(s0)
        shas_to_cull.add(s1)

    for s in protected_shas:
        if s in shas_to_cull:
            shas_to_cull.remove(s)

    for r in results:
        abs_deltas = [
            max(abs(i) for i in (i_p, i_c) if i is not None)
            for i_p, i_c in zip(r["Count deltas (Python)"], r["Count deltas (C++)"])
            if i_p is not None or i_c is not None
        ]

        keep_shas = False
        for n, threshold in FILTER_THRESHOLDS:
            keep_shas |= sum(1 for d in abs_deltas if d >= threshold * 0.7) >= n
        if keep_shas:
            for s in r["SHAs"]:
                if s in shas_to_cull:
                    shas_to_cull.remove(s)

    for sha, _, _, _, msg in runner._history:
        if sha in shas_to_cull:
            print(msg)

    print(len(shas_to_cull))

    # WARNING!!!
    #   The actual delete is left commented to avoid accidental catistrophic loss.
    #   Uncomment if you've run it and the results look reasonable.

    # import pdb
    # pdb.set_trace()

    # for i, sha in enumerate(shas_to_cull):
    #     shutil.rmtree(runner.state.built[sha])
    #     runner.state.built.pop(sha)

    #     for fpath in runner.state.finished[sha]:
    #         os.remove(fpath)
    #     runner.state.finished.pop(sha)
    #     runner.state.to_json()

    #     print(f"{i:>3}  {sha}")


def _make_report(threshold_multiplier=1, name="test"):
    """Horrible hacky mess of HTML. But it works..."""
    builder = build_pytorch.PytorchBuildHelper(
        root_dir=WORKSPACE_ROOT,
        clean=False,
        soft_clean=False,
        main_loop=False,
    )
    runner = Runner(builder)
    history_by_sha = {r[0]: r for r in runner._history}

    background_color = "Black"
    min_colored = 0.005
    scale_max = 0.25

    # Time vs LWM can't be negative, but holding the line is good and
    # should be called out as such.
    vs_lwm_factor = 3

    lang_keys = ("Python", "C++")

    def color_by_value(x):
        if x > 0:
            # Red
            scale = [
                background_color,
                '#808080', '#867979', '#8c7373', '#936c6c', '#996666',
                '#9f6060', '#a65959', '#ac5353', '#b34d4d', '#b94646',
                '#bf4040', '#c63939', '#cc3333', '#d22d2d', '#d92626',
                '#df2020', '#e61919', '#ec1313', '#f20d0d', '#f90606',
                '#ff0000'
            ]
        else:
            scale = [
                background_color,
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

    newline = "\n"
    lines = [[f"<H3>{'&nbsp;' * 110}</H3>"]]
    count = 0
    partition_indices = [-1]
    for n, task_name in _TASK_GROUPS:
        lines.append([f"<H3><u>{task_name}</u></H3>"])
        for _ in range(n):
            lines[-1].append(f"<b>[{count}]</b>&nbsp; {TASKS[count][1]}{'&nbsp' * 5}<br>")
            count += 1
        partition_indices.append(partition_indices[-1] + n)
    lines = [['<td style="vertical-align:top">'] + l + ["</td>\n"] for l in lines]
    lines = list(it.chain(*lines))
    partition_indices = set(partition_indices[1:])

    results, count_lwm_python, count_lwm_cpp = runner.segment_results(mask_suspect=True)
    filtered_results = []
    for r in results:
        abs_deltas = [
            max(abs(i) for i in (i_p, i_c) if i is not None)
            for i_p, i_c in zip(r["Count deltas (Python)"], r["Count deltas (C++)"])
            if i_p is not None or i_c is not None
        ]

        for n, th in REPORT_THREHOLDS:
            if sum(1 for i in abs_deltas if i > th * threshold_multiplier) >= n:
                if r["Num unbuildable intermediates"]:
                    print(
                        f"Skipping ambiguous range: {r['SHAs'][0]} - {r['SHAs'][1]} "
                        f"({r['Dates'][0]} - {r['Dates'][1]}), {len(r['Intermediate SHAs'])} commits")
                    break
                filtered_results.append(r)
                break

    approx_remaining = int(sum(math.log2(len(r['Intermediate SHAs']) + 1) for r in filtered_results))
    print(f"Approximate build/test runs remaining: {approx_remaining}")

    n_padding = 8
    padding = [f'<td colspan="{n_padding}"></td>']
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
            '</td><td style="text-align:right;border-bottom: 1px solid white">&nbsp;</td>',
        ])
    result_lines.append("</tr>")

    current_date_str = results[-1]["Dates"][1].strftime("%m/%d")
    for subrow_i, (lang, lwm) in enumerate(zip(lang_keys, (count_lwm_python, count_lwm_cpp))):
        key = f"Counts ({lang})"
        lower_border = ";border-bottom: 1px solid white"
        border_str = lower_border if subrow_i + 1 == len(lang_keys) else ""

        result_lines.append("<tr>")
        if not subrow_i:
            result_lines.extend([
                f'<td rowspan={len(lang_keys)} colspan="{n_padding - 3}" style="{lower_border}"></td>',
                f'<td rowspan={len(lang_keys)} style="text-align:center{lower_border}">'
                f"{'&nbsp;' * 5} Current ( {current_date_str} )<br> vs.<br> Low water mark"
            ])

        result_lines.extend([
            f'</td><td style="text-align:left{border_str}">\u0394 {key}</td>',
            f'<td style="text-align:center{border_str}">(+X %):</td>'
        ])
        for i, (x, x_lwm) in enumerate(zip(results[-1][key][1], lwm)):
            if x is None:
                color = "Grey"
            else:
                color = color_by_value(x / x_lwm - 1 - 10 * min_colored)
                color = "Grey" if color == background_color else color

            result_lines.append(
                f'<td style="text-align:right;color:{color}{border_str}">' +
                ("--" if x is None else f"{(x / x_lwm - 1) * 100:.0f}") +
                "</td>"
            )
            if i in partition_indices:
                result_lines.append(f'<td style="text-align:right{border_str}">&nbsp;</td>')
        result_lines.append("</tr>")
    result_lines.append(
        f'<tr><td colspan={n_padding + len(TASKS) + len(partition_indices)} '
        f'style="background-color:DarkGrey{lower_border}"></td></tr>'
    )

    def color_value(x, border=False):
        if x is None:
            color = "Grey"
            s = "--"
        else:
            color = color_by_value(x)
            s = f"{x * 100:.1f}"
        border_str = ";border-bottom: 1px dashed grey" if border else ""
        return f'<td style="text-align:right;color:{color}{border_str}">{s}</td>'

    def sha_to_url(sha):
        return f'<a href="https://github.com/pytorch/pytorch/commit/{sha}" style="color:White">{sha[:7]}</a>'

    arrows = [
        "\u21e7",  # ⇧
        "\u2191",  # ↑
        "\u21e1",  # ⇡
        "&nbsp;",
        "&nbsp;",
        "&nbsp;",
        "&nbsp;",
        "\u21e3",  # ⇣
        "\u2193",  # ↓
        "\u21e9",  # ⇩
    ]

    row_keys = [f"Count deltas ({lang})" for lang in lang_keys]
    for r in filtered_results[::-1]:
        all_printed_values = list(it.chain(*[
            [i for i in r[key] if i is not None and abs(i) >= min_colored]
            for key in row_keys
        ]))

        if not all_printed_values:
            continue

        for subrow_i, (lang, key) in enumerate(zip(lang_keys, row_keys)):
            border = (subrow_i + 1 == len(row_keys))
            border_str = "border-bottom: 1px dashed grey;" if border else ""

            string_counts = []
            n_intermediate = len(r['Intermediate SHAs']) + 1

            printed_values = [i for i in r[key] if i is not None and abs(i) >= min_colored]
            color = (
                color_by_value(statistics.median(printed_values))
                if printed_values else background_color
            )
            if color == background_color:
                color = "Grey"

            if not printed_values:
                direction_text = ""
            elif min(printed_values) > 0:
                direction_text = "Increase"
            elif max(printed_values) < 0:
                direction_text = "Decrease"
            else:
                direction_text = "Mixed"

            direction_template = f'<td style="{border_str}text-align:{{align}};color:{color}">{{text}}</td>'
            direction = direction_template.format(align="center", text=direction_text)

            if subrow_i:
                msg, date_str, author_name = "", "", ""

            else:
                if n_intermediate == 1:
                    msg = r["Messages"][1].strip()
                    date_str = r["Dates"][1].strftime("%m/%d/%Y")
                    _, _, author_name, _, _ = history_by_sha[r["SHAs"][1]]
                else:
                    msg = f"{'&nbsp;' * 4}{n_intermediate} &nbsp; commits"
                    d0, d1 = [d.strftime("%m/%d/%Y") for d in r["Dates"]]
                    date_str = f"{d0} - {d1}"
                    author_name = ""

                if len(msg) > 80:
                    msg = msg[:77] + "..."

            td_template = f'<td style="{border_str}text-align:{{align}}">'
            string_counts.extend([
                td_template.format(align="right"),
                ("" if n_intermediate == 1 or subrow_i else f"{sha_to_url(r['SHAs'][0])} - ") +
                ("" if subrow_i else f"{sha_to_url(r['SHAs'][1])}{'&nbsp;' * 8}</td>"),
                f'{td_template.format(align="left")}{author_name}</td>',
                direction,
                direction_template.format(
                    align="right",
                    text=f"{min(printed_values) * 100:.1f}" if printed_values else ""
                ),
                direction_template.format(
                    align="left",
                    text=f", {'&nbsp;' * 3}{max(printed_values) * 100:.1f}" if len(printed_values) > 1 else ""
                ),
                f'{td_template.format(align="left")}{msg}</td>',
                f'{td_template.format(align="left")}{date_str}</td>',
                f'<td style="{border_str}border-right: 1px dashed white;text-align:left">{lang}</td>'
            ])

            for i, x in enumerate(r[key]):
                string_counts.append(color_value(x, border=(subrow_i + 1 == len(row_keys))))
                if i in partition_indices:
                    string_counts.append(
                        f'<td style="{border_str}border-right: 1px dashed white">&nbsp;</td>')

            # string_counts.extend([
            #     f'<td style="center">{random.choice(arrows)}</td>'
            #     for _ in range(len(TASKS))
            # ])

            result_lines.append(
                "\n".join(["<tr>"] + string_counts + ["</tr>\n"])
            )

    table_width = f"{7 * len(TASKS) + 120}ch"
    body = textwrap.dedent(f"""\
    <HTML>
      <body style="background-color:{background_color};color:WhiteSmoke">

      <table style="width:{table_width}">
{textwrap.indent(newline.join(lines), ' ' * 6)}
      </table>

      <br><br><br>

      <table style="width:{table_width}">
{textwrap.indent(newline.join(result_lines), ' ' * 6)}
      </table>
      <body>
    </HTML>
    """)

    fpath = f"/mnt/shared/{os.getenv('USER')}/public_html/{name}.html"
    print(f"Path: {fpath}")
    with open(fpath, "wt") as f:
        f.write(body)


def make_report():
    # _make_report(threshold_multiplier=1, name="bisect_report")
    # _make_report(threshold_multiplier=1.5, name="bisect_report_strict")
    _make_report(threshold_multiplier=1, name="bisect_report_debug")


def debug():
    builder = build_pytorch.PytorchBuildHelper(
        root_dir=WORKSPACE_ROOT,
        clean=False,
        soft_clean=False,
        main_loop=False,
    )

    runner = Runner(builder)
    print(len(runner.state.built), len(runner.state.finished))

    first_sha = None
    history = builder.get_history_since("2020-07-01")
    for i in history:
        sha = i[0]
        if sha in runner.state.finished:
            first_sha = first_sha or sha
            last_sha = sha

    for i in history:
        if i[0] in (first_sha, last_sha):
            print(i)

    with open(runner.state.finished[first_sha][1], "rb") as f:
        old_results = pickle.load(f)

    with open(runner.state.finished[last_sha][1], "rb") as f:
        new_results = pickle.load(f)

    import json
    import statistics
    results = []
    for (_, stmt), old_py_cts, new_py_cts, old_cpp_cts, new_cpp_cts in zip(_TASKS, old_results["Python"], new_results["Python"], old_results["C++"], new_results["C++"]):
        results.append(["Python", stmt, [statistics.median(old_py_cts), statistics.median(new_py_cts)]])

        cpp_stmt = CPP_ANALOGS.get(stmt)
        if cpp_stmt is None:
            results.append(["C++", "No analog tested", [None, None]])
        else:
            results.append(["C++", cpp_stmt, [statistics.median(old_cpp_cts), statistics.median(new_cpp_cts)]])

    rel_diffs = []
    for lang, stmt, (c_old, c_new) in results:
        if not c_old:
            continue
        stmt = stmt if isinstance(stmt, str) else " \\n ".join(stmt)
        print(f"{lang:<8} {(c_new - c_old) / c_new * 100:>8.1f}%   {stmt}")
        rel_diffs.append((c_new - c_old) / c_new)
    print(statistics.mean(rel_diffs))

    # results = {"Old SHA": first_sha, "New SHA": last_sha, "Counts": results}
    # with open("/mnt/shared/taylorrobie/public_html/that_which_is_measured_improves.json", "wt") as f:
    #     json.dump(results, f, indent=4)


    # runner.timing_loop()


def main():
    builder = build_pytorch.PytorchBuildHelper(
        root_dir=WORKSPACE_ROOT,
        clean=False,
        soft_clean=False,
        main_loop=True,
    )

    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    os.makedirs(COMPLETED_BUILD_ROOT, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    runner = Runner(builder)
    runner.loop()


_MODES = {
    "main": main,
    "measure_counts": measure_counts,
    "measure_times": measure_times,
    "post_process": post_process,
    "compute_soft_deltas": compute_soft_deltas,
    "make_report": make_report,
    "cull": cull,  # Warning: dangerous!

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
