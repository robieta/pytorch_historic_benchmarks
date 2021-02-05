import collections
import dataclasses
import enum
import json
import multiprocessing.dummy
import os
import pickle
import shutil
import threading
import time
import traceback
from typing import Deque, Dict, Generic, Iterable, List, Optional, Set, TypeVar

from torch.utils.benchmark import CallgrindStats, Language, Measurement

from v2.build import build_clean, check_unbuildable
from v2.containers import BenchmarkResult, BenchmarkResults
from v2 import init_pytorch
from v2.runner_interface import backport, clean_backport, get_benchmarks, AutogradMode, BenchmarkRunner, InstrumentedCorePool, RuntimeMode, WorkOrder
from v2.logging_subprocess import call
from v2.runner_policy import get_exclude_sets, BASELINE_BUILD_CORE_COUNT, NUM_CORES, OPPORTUNISTIC_BUILD_CORE_COUNTS, POOL_SLACK
from v2.workspace import (
    BUILD_COMPLETED_ROOT, DATE_FMT, MUTATION_LOCK, PDB_FILE, RUN_LOG_ROOT,
    RUN_COMPLETED_ROOT, RUNNER_STATE_ROOT, STOP_FILE, SWEEP_CADENCE, SWEEP_START)


_STATE_PATH = os.path.join(RUNNER_STATE_ROOT, "state.json")


T = TypeVar("T")
class UniqueDeque(Generic[T]):
    def __init__(self):
        self._queue: Deque[T] = collections.deque()
        self._contents: Set[T] = set()

    def append(self, item: T) -> None:
        if item not in self._contents:
            self._contents.add(item)
            self._queue.append(item)

    def popleft(self) -> T:
        # NB: We do not remove from `_contents`
        result = self._queue.popleft()
        return result

    def extend_contents(self, new_contents: Iterable[T]):
        for i in new_contents:
            self._contents.add(i)

    def __contains__(self, item: T) -> bool:
        return item in self._contents

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        return iter(self._queue)


class TaskType(enum.Enum):
    BUILD = 0
    TEST = 1
    ARCHIVE = 2


@dataclasses.dataclass()
class RunnerState:
    # Contents are SHAs
    _build_queue: UniqueDeque[str]
    _test_queue: UniqueDeque[str]
    _archive_queue: UniqueDeque[str]
    _in_progress: Dict[str, TaskType]

    _built: Dict[str, str]
    _tested: Dict[str, str]
    _archived: Dict[str, str]

    _lock: threading.Lock
    _frozen: bool

    @staticmethod
    def from_file(frozen: bool = False) -> "RunnerState":
        state = {}
        if os.path.exists(_STATE_PATH):
            with open(_STATE_PATH, "rt") as f:
                state = json.load(f)

        result = RunnerState(
            UniqueDeque(),
            UniqueDeque(),
            UniqueDeque(),
            {},
            state.get("built", {}),
            state.get("tested", {}),
            state.get("archived", {}),
            threading.Lock(),
            frozen,
        )

        result._build_queue.extend_contents(result._built.keys())
        result._test_queue.extend_contents(result._tested.keys())
        result._archive_queue.extend_contents(result._archived.keys())

        for i in result._built.keys():
            result._test_queue.append(i)

        for i in result._tested.keys():
            result._archive_queue.append(i)

        for sha in result._built:
            result._build_queue._contents.add(sha)

        return result

    @staticmethod
    def clean(frozen: bool = False) -> "RunnerState":
        if os.path.exists(_STATE_PATH):
            os.remove(_STATE_PATH)
        return RunnerState.from_file(frozen)

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def queue_dict(self) -> Dict[TaskType, UniqueDeque]:
        return {
            TaskType.BUILD: self._build_queue,
            TaskType.TEST: self._test_queue,
            TaskType.ARCHIVE: self._archive_queue,
        }

    def to_file(self) -> None:
        assert not self.frozen
        MUTATION_LOCK.get()
        if os.path.exists(_STATE_PATH):
            old_path = f"{_STATE_PATH}.old"
            if os.path.exists(old_path):
                os.remove(old_path)
            shutil.copyfile(_STATE_PATH, old_path)

        with open(_STATE_PATH, "wt") as f:
            json.dump({
                "built": self._built,
                "tested": self._tested,
                "archived": self._archived,
            }, f, indent=4)

    def maybe_enqueue_build(self, sha: str) -> None:
        assert not self.frozen
        with self._lock:
            if sha in self._built or sha in self._in_progress:
                return

            self._build_queue.append(sha)

    def get_job(self, task_type: TaskType) -> Optional[str]:
        assert not self.frozen
        queue = self.queue_dict[task_type]

        with self._lock:
            if queue:
                output = queue.popleft()
                assert output not in self._in_progress
                self._in_progress[output] = task_type
                return output

    def report_finished(self, task_type: TaskType, sha: str, result: str) -> None:
        assert not self.frozen
        results, next_queue = {
            TaskType.BUILD: (self._built, self._test_queue),
            TaskType.TEST: (self._tested, self._archive_queue),
            TaskType.ARCHIVE: (self._archived, None),
        }[task_type]

        with self._lock:
            results[sha] = result
            assert self._in_progress[sha] == task_type
            self._in_progress.pop(sha)
            if next_queue:
                next_queue.append(sha)
            self.to_file()

    def queue_len(self, task_type: TaskType):
        return len(self.queue_dict[task_type])

    def get_env(self, sha: str) -> str:
        assert sha in self._built
        return self._built[sha]


class Runner:
    def __init__(self):
        MUTATION_LOCK.get()
        # init_pytorch.fetch_fbcode_warm()
        self._state = RunnerState.from_file(frozen=False)
        self._core_pool = InstrumentedCorePool(POOL_SLACK)
        self._lock = threading.RLock()
        self._history = init_pytorch.get_history().since(SWEEP_START)
        self._history_dict = {commit.sha: commit for commit in self._history}

        # Initial sweep
        self._initial_sweep_indices: List[int] = [0]
        for i, commit in enumerate(self._history[:-1]):
            last_commit = self._history[self._initial_sweep_indices[-1]]
            if (commit.date - last_commit.date).days >= SWEEP_CADENCE:
                self._initial_sweep_indices.append(i)
        self._initial_sweep_indices.append(i + 1)

        for i in self._initial_sweep_indices:
            if not check_unbuildable(self._history[i].sha):
                self._state.maybe_enqueue_build(self._history[i].sha)

        self._stop = False
        self._benchmarks = None

    def run(self):
        assert self._benchmarks is None
        self._benchmarks = get_benchmarks()

        # Does not include control loop.
        num_workers = 6  # TODO: tune.
        with multiprocessing.dummy.Pool(num_workers + 1) as pool:
            pool.map(self.worker_fn, range(num_workers + 1), 1)

    def worker_fn(self, worker_id: int):
        # Worker 0 is the control loop.
        if not worker_id:

            while not self._stop:
                if os.path.exists(PDB_FILE):
                    os.remove(PDB_FILE)
                    import pdb
                    pdb.set_trace()

                    # print(sum([n.num_available for n in self._core_pool._nodes]))

                    # print("\n".join([" ".join([str(j) for j in i]) for i in self._core_pool._log]))

                if os.path.exists(STOP_FILE):
                    os.remove(STOP_FILE)
                    self._stop = True
                    print("Stop registered.")
                    return

                if not self.scheduled_work:
                    self._stop = True  # Debug

                # TODO: Bisection.

                time.sleep(1)
            return

        while not self._stop:
            if worker_id <= 3:
                # Reserve four workers for building.
                self.build()

            else:
                self.measure()
                # self.build()
            time.sleep(10)

    @property
    def scheduled_work(self):
        # Approximate number of cores worth of concurrent outstanding work.
        return sum([
            self._state.queue_len(TaskType.BUILD) * BASELINE_BUILD_CORE_COUNT,
            self._state.queue_len(TaskType.TEST) * int(NUM_CORES / 2),
            self._state.queue_len(TaskType.ARCHIVE),
        ])

    def build(self):
        priority = (BASELINE_BUILD_CORE_COUNT,)
        if self.scheduled_work + BASELINE_BUILD_CORE_COUNT < NUM_CORES:
            priority = OPPORTUNISTIC_BUILD_CORE_COUNTS + priority

        for core_request in priority:
            allocation = self._core_pool.reserve(core_request)

        if allocation is None:
            return

        sha = self._state.get_job(TaskType.BUILD)
        if sha is None:
            return

        try:
            commit = self._history_dict[sha]
            print(f"Building: {sha} ({commit.date_str})  {core_request} workers")
            start_time = time.time()
            conda_env = build_clean(
                sha,
                commit.build_cfg,
                show_progress=False,
                taskset_cores=allocation,
                nice="15",
                max_jobs=core_request,
            )

            if conda_env is None:
                return  # unbuildable

            new_env_path = os.path.join(BUILD_COMPLETED_ROOT, f"{commit.date_str}_{sha}")
            shutil.copytree(conda_env, new_env_path, symlinks=True)
            shutil.rmtree(conda_env)

            self._state.report_finished(TaskType.BUILD, sha, new_env_path)
            print(f"Build time: {sha} ({commit.date_str}) {time.time() - start_time:.0f} sec")

        finally:
            self._core_pool.release(allocation)

    def measure(self):
        # sha = self._state.get_job(TaskType.TEST)

        with self._lock:
            sha = self._state.get_job(TaskType.TEST)
            for _ in range(5):
                self._state.get_job(TaskType.TEST)

        if sha is None:
            return

        commit = self._history_dict[sha]
        conda_env = self._state.get_env(sha)
        torch_path = os.path.join(
            conda_env,
            "lib",
            f"python{commit.build_cfg.python_version}",
            "site-packages",
            "torch",
        )

        print(f"Test begin: {sha} {commit.date_str}")

        if not os.path.exists(torch_path):
            print(f"TODO: {sha} {commit.date_str}")
            return

        assert os.path.exists(torch_path), torch_path
        clean_backport(torch_path)

        try:
            # Valgrind will sometimes fail with:
            #   `valgrind: failed to start tool 'callgrind' for platform 'amd64-linux': No such file or directory`
            # but a re-install fixes it.
            check_valgrind_retcode = call(
                "valgrind --version",
                shell=True,
                conda_env=conda_env,
                timeout=60,
                task_name="Check Valgrind",
                log_dir=RUN_LOG_ROOT,
            )

            if check_valgrind_retcode:
                call(
                    """
                    conda remove -y valgrind
                    conda install -y valgrind -c conda-forge
                    valgrind --version
                    """,
                    shell=True,
                    check=True,
                    conda_env=conda_env,
                    timeout=60,
                    task_name="Reinstall Valgrind",
                    log_dir=RUN_LOG_ROOT,
                )

            backport(torch_path)

            work_orders: List[WorkOrder] = []
            may_fail, will_fail = get_exclude_sets(commit.date_str)
            for label, auto_labels, timer_args in self._benchmarks:
                # fail_key = (
                #     timer_args.language == Language.CPP,
                #     auto_labels.autograd == AutogradMode.FORWARD_BACKWARD,
                #     auto_labels.runtime == RuntimeMode.JIT,
                #     label,
                # )
                # if fail_key in will_fail:
                #     continue

                work_orders.append(WorkOrder(
                    label=label,
                    auto_labels=auto_labels,
                    timer_args=timer_args,
                    source_cmd=f"source activate {conda_env}",
                    timeout=400.0,
                    retries=3,
                    allow_failure=True, # (fail_key in may_fail),
                ))

            benchmark_runner = BenchmarkRunner(
                work_items=tuple(work_orders),
                core_pool=self._core_pool,
                display_progress=False,
            )

            try:
                self._core_pool.register_benchmark_runner(benchmark_runner)
                results = benchmark_runner.run()
                simple_results = []
                raw_results = []
                lang_to_str = {Language.PYTHON: "Python", Language.CPP: "C++"}
                num_failures = 0
                for w in work_orders:
                    if w not in results:
                        num_failures += 1
                        continue

                    r = results[w]
                    simple_results.append(BenchmarkResult(
                        w.label,
                        lang_to_str[w.auto_labels.language],
                        w.auto_labels.autograd.value,
                        w.auto_labels.runtime.value,
                        r.wall_time.median,
                        r.instructions.counts(denoise=True),
                    ))
                    raw_results.append((w.label, w.auto_labels, r))

                simple_path = os.path.join(RUN_COMPLETED_ROOT, f"{commit.date_str}__{sha}.pkl")
                raw_path = os.path.join(RUN_COMPLETED_ROOT, f"{commit.date_str}__{sha}_raw.pkl")

                with open(simple_path, "wb") as f:
                    pickle.dump(BenchmarkResults(
                        sha=sha,
                        conda_env=conda_env,
                        values=tuple(simple_results)
                    ), f)

                with open(raw_path, "wb") as f:
                    pickle.dump([sha, conda_env, raw_results], f)

                self._state.report_finished(TaskType.TEST, sha=sha, result=simple_path)
                print(f"Test finished: {sha} {commit.date_str}  {num_failures} / {len(work_orders)} failures")


            except BaseException as e:
                print(f"Failed: {sha} {e}")

            finally:
                try:
                    self._core_pool.unregister_benchmark_runner()
                except BaseException:
                    pass

        finally:
            clean_backport(torch_path)
