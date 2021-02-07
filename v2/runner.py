import collections
import dataclasses
import enum
import json
import multiprocessing.dummy
import os
import pickle
import shutil
import tarfile
import tempfile
import threading
import time
import traceback
from typing import Dict, Generic, List, Optional

from torch.utils.benchmark import CallgrindStats, Language, Measurement

from v2.build import build_clean, check_unbuildable, mark_unbuildable
from v2.containers import BenchmarkResult, BenchmarkResults, UniqueDeque
from v2 import init_pytorch
from v2.runner_interface import backport, clean_backport, get_benchmarks, AutogradMode, BenchmarkRunner, RuntimeMode, WorkOrder
from v2.logging_subprocess import call
from v2.runner_policy import BUILD_WORKERS, TEST_WORKERS, NUM_CORES, SharedCorePool, TaskType
from v2.workspace import (
    BUILD_COMPLETED_ROOT, BUILD_LOG_ROOT, BUILD_IN_PROGRESS_ROOT, DATE_FMT,
    MUTATION_LOCK, PDB_FILE, RUN_LOG_ROOT,
    RUN_IN_PROGRESS_ROOT, RUN_COMPLETED_ROOT, RUNNER_STATE_ROOT, STOP_FILE,
    SWEEP_CADENCE, SWEEP_START)


_STATE_PATH = os.path.join(RUNNER_STATE_ROOT, "state.json")

class WorkerType(enum.Enum):
    CONTROL = 0
    BUILD = 1
    TEST = 2


@dataclasses.dataclass()
class RunnerState:
    # Contents are SHAs
    _build_queue: UniqueDeque[str]
    _test_queue: UniqueDeque[str]
    _in_progress: Dict[str, TaskType]

    _built: Dict[str, str]
    _tested: Dict[str, str]

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
            {},
            state.get("built", {}),
            state.get("tested", {}),
            threading.Lock(),
            frozen,
        )

        result._build_queue.extend_contents(result._built.keys())
        result._test_queue.extend_contents(result._tested.keys())

        for i in result._built.keys():
            result._test_queue.append(i)

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
            TaskType.MEASURE: self._test_queue,
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
            TaskType.MEASURE: (self._tested, None),
        }[task_type]

        with self._lock:
            results[sha] = result
            assert self._in_progress[sha] == task_type
            self._in_progress.pop(sha)
            if next_queue is not None:
                next_queue.append(sha)
            self.to_file()

    def queue_len(self, task_type: TaskType):
        return len(self.queue_dict[task_type])

    def get_build(self, sha: str) -> str:
        assert sha in self._built
        return self._built[sha]


class Runner:
    def __init__(self):
        MUTATION_LOCK.get()
        # init_pytorch.fetch_fbcode_warm()

        self._state = RunnerState.from_file(frozen=False)
        self._core_pool = SharedCorePool()
        self._lock = threading.RLock()
        self._history = init_pytorch.get_history().since(SWEEP_START)
        self._history_dict = {commit.sha: commit for commit in self._history}
        self._history_indices = {commit.sha: i for i, commit in enumerate(self._history)}
        self._success_by_sha = {}
        self._main_thread = threading.get_ident()

        # Initial sweep
        self._initial_sweep_indices: List[int] = [0]
        for i, commit in enumerate(self._history[:-1]):
            last_commit = self._history[self._initial_sweep_indices[-1]]
            if (commit.date - last_commit.date).days >= SWEEP_CADENCE:
                self._initial_sweep_indices.append(i)
        self._initial_sweep_indices.append(i + 1)

        # First do a strided passes to establish which benchmarks will fail
        # on which commit ranges.
        sweep_order = (
            self._initial_sweep_indices[::10] +
            self._initial_sweep_indices[::5] +
            self._initial_sweep_indices
        )

        for i in sweep_order:
            if not check_unbuildable(self._history[i].sha):
                self._state.maybe_enqueue_build(self._history[i].sha)

        self._stop = False
        self._active_workers = 0
        self._benchmarks = None

    def run(self):
        assert self._benchmarks is None
        self._benchmarks = get_benchmarks()
        self._update_success_dict()

        worker_specs = (
            [(WorkerType.CONTROL, 0)] +
            [(WorkerType.BUILD, i * 10) for i in range(BUILD_WORKERS)] +
            [(WorkerType.TEST, i * 120) for i in range(TEST_WORKERS)]
        )
        with multiprocessing.dummy.Pool(len(worker_specs)) as pool:
            pool.map(self.worker_fn, enumerate(worker_specs), 1)

    def worker_fn(self, args: int):
        worker_id, (worker_type, startup_sleep) = args
        self._active_workers += 1
        time.sleep(startup_sleep)

        try:
            while True:
                if worker_type == WorkerType.CONTROL:
                    if os.path.exists(PDB_FILE):
                        os.remove(PDB_FILE)
                        import pdb
                        pdb.set_trace()

                    if os.path.exists(STOP_FILE):
                        os.remove(STOP_FILE)
                        self._stop = True
                        print("Stop registered.")
                        return

                    # TODO: Bisection.

                    time.sleep(1)
                    if self._stop and self._active_workers <= 1:
                        break

                if worker_type == WorkerType.BUILD:
                    self.build()

                    time.sleep(10)
                    if self._stop:
                        break

                if worker_type == WorkerType.TEST:
                    self.measure()

                    time.sleep(10)
                    if self._stop:
                        break

        except KeyboardInterrupt:
            return

        except BaseException:
            print(f"\n{'=' * 80}")
            print(f"== Worker {worker_id} ({worker_type}) failed ".ljust(80, "="))
            print(f"{'=' * 80}\n")
            traceback.print_exc()

        finally:
            self._active_workers -= 1

    def _update_success_dict(self, sha=None):
        if sha is None:
            shas = list(self._state._tested.keys())
        else:
            shas = [sha]

        for sha in shas:
            if sha in self._success_by_sha:
                continue

            with open(self._state._tested[sha], "rb") as f:
                result: BenchmarkResults = pickle.load(f)
            assert result.sha == sha
            self._success_by_sha[sha] = tuple(
                (r.label, r.language, r.autograd, r.runtime)
                for r in result.values
            )

    def _filter_work_orders(self, sha: str, work_orders: List[WorkOrder]):
        sha_ind = self._history_indices[sha]

        lower_sha = None
        for i in range(sha_ind, -1, -1):
            ls = self._history[i].sha
            if ls in self._success_by_sha:
                lower_sha = ls
                break

        upper_sha = None
        for i in range(sha_ind, len(self._history)):
            us = self._history[i].sha
            if us in self._success_by_sha:
                upper_sha = us
                break

        if lower_sha is None or upper_sha is None:
            return work_orders

        success = (
            set(self._success_by_sha[lower_sha]) |
            set(self._success_by_sha[upper_sha])
        )

        output = []
        lang_to_str = {Language.PYTHON: "Python", Language.CPP: "C++"}
        for w in work_orders:
            key = (
                w.label,
                lang_to_str[w.auto_labels.language],
                w.auto_labels.autograd.value,
                w.auto_labels.runtime.value,
            )
            if key in success:
                output.append(w)
        return output

    @staticmethod
    def check_env(conda_env: str):
        def check_call(cmd):
            retcode = call(
                cmd,
                cwd=tempfile.gettempdir(),
                shell=True,
                conda_env=conda_env,
                timeout=60,
                log_dir=RUN_LOG_ROOT,
            )
            if retcode:
                print(f"Failed: {cmd}")
            return retcode

        test_retcode = check_call('python -c "import torch"')
        check_call('conda env config vars set VALGRIND_LIB="${CONDA_PREFIX}/lib/valgrind"')
        valgrind_retcode = check_call('valgrind --version')

        return test_retcode or valgrind_retcode

    def build(self):
        sha = None
        success = False
        conda_env = None
        self._core_pool.begin_task(TaskType.BUILD)
        try:
            with self._lock:
                n_builds_queued = self._state.queue_len(TaskType.BUILD)
                allocation = None
                for limit, core_request in ((1, 32), (2, 16), (n_builds_queued, 8)):
                    if n_builds_queued <= limit:
                        allocation = allocation or self._core_pool.reserve(core_request)
                        if allocation:
                            break

                if allocation is None:
                    success = True
                    return

                sha = self._state.get_job(TaskType.BUILD)

            if sha is None:
                success = True
                return

            start_time = time.time()
            commit = self._history_dict[sha]

            def log_build_state(msg: str):
                print(f"{time.time() - start_time:>8.0f} Building: {sha} ({commit.date_str}) {msg}")

            log_build_state(f"{core_request} workers")
            conda_env = build_clean(
                sha,
                commit.build_cfg,
                show_progress=(threading.get_ident() == self._main_thread),
                taskset_cores=allocation,
                nice="15",
                max_jobs=core_request,
            )
            self._core_pool.release(allocation)
            log_build_state(f"Conda env: {conda_env}")

            if conda_env is None:
                return  # unbuildable

            symlinks = {}
            for root, _, fnames in os.walk(conda_env):
                for fname in fnames:
                    fpath = os.path.join(root, fname)
                    if os.path.islink(fpath):
                        link_target = os.readlink(fpath)
                        if not os.path.isabs(link_target):
                            link_target = os.path.abspath(os.path.join(root, link_target))
                            assert os.path.exists(link_target)
                        symlinks[fpath[len(conda_env) + 1:]] = link_target[len(conda_env) + 1:]
            with open(os.path.join(conda_env, "SYMLINKS.json"), "wt") as f:
                json.dump(symlinks, f, indent=4)

            if self.check_env(conda_env):
                log_build_state("Check env failed.")

            self._core_pool.finish_task()
            self._core_pool.begin_task(TaskType.ARCHIVE)
            allocation = self._core_pool.reserve(1)
            while allocation is None:
                time.sleep(1)
                allocation = self._core_pool.reserve(1)

            final_name = f"{commit.date_str}_{sha}"

            archive_staging_path = os.path.join(BUILD_IN_PROGRESS_ROOT, f"{final_name}.tar.gz")
            archived_path = os.path.join(BUILD_COMPLETED_ROOT, f"{final_name}.tar.gz")

            with tarfile.open(archive_staging_path, "w:gz") as tar:
              tar.add(conda_env, arcname=final_name)
            shutil.move(archive_staging_path, archived_path)

            log_build_state(f"Archive: {archived_path}")
            self._state.report_finished(TaskType.BUILD, sha, archived_path)
            self._core_pool.release(allocation)

            shutil.rmtree(conda_env)
            del conda_env

            success = True

        except BaseException:
            traceback.print_exc()

        finally:
            self._core_pool.finish_task()
            if not success:
                print(f"Build {sha} failed. ({conda_env})")
                if sha is not None:
                    mark_unbuildable(sha)

    def measure(self):
        sha = self._state.get_job(TaskType.MEASURE)
        if sha is None:
            return

        start_time = time.time()
        commit = self._history_dict[sha]
        archive = self._state.get_build(sha)

        def log_measure_state(msg: str):
            print(f"{time.time() - start_time:>8.0f} Testing: {sha} ({commit.date_str}) {msg}")

        suffix = ".tar.gz"
        assert archive.endswith(suffix)
        name = os.path.split(archive)[1][:-len(suffix)]

        conda_env = os.path.join(RUN_IN_PROGRESS_ROOT, name)
        conda_lib_path = os.path.join(conda_env, "lib")
        log_measure_state("Testing,")

        if os.path.exists(conda_env):
            os.chmod(conda_lib_path, 0o755)
            shutil.rmtree(conda_env)
        shutil.unpack_archive(archive, RUN_IN_PROGRESS_ROOT)

        # Lock `lib` to prevent mkl symlinks from being removed.
        os.chmod(conda_lib_path, 0o555)

        valid_env = not self.check_env(conda_env)
        if not valid_env:
            log_measure_state("Failed. Exiting.")
            return

        torch_path = os.path.join(
            conda_env,
            "lib",
            f"python{commit.build_cfg.python_version}",
            "site-packages",
            "torch",
        )
        if not os.path.exists(torch_path):
            log_measure_state(f"torch path {torch_path} does not exist.")

        try:
            self._core_pool.begin_task(TaskType.MEASURE)
            backport(torch_path)

            work_orders: List[WorkOrder] = []
            for label, auto_labels, timer_args in self._benchmarks:
                work_orders.append(WorkOrder(
                    label=label,
                    auto_labels=auto_labels,
                    timer_args=timer_args,
                    source_cmd=f"source activate {conda_env}",
                    timeout=400.0,
                    retries=3,
                    allow_failure=True,
                ))
            work_orders = self._filter_work_orders(sha, work_orders)

            benchmark_runner = BenchmarkRunner(
                work_items=tuple(work_orders),
                core_pool=self._core_pool,
                display_progress=(threading.get_ident() == self._main_thread),
            )
            results = benchmark_runner.run()
            log_measure_state("Benchmark run complete.")

            simple_results = []
            raw_results = []
            lang_to_str = {Language.PYTHON: "Python", Language.CPP: "C++"}
            for w in work_orders:
                if w not in results:
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

            simple_path = os.path.join(RUN_COMPLETED_ROOT, f"{name}.pkl")
            raw_path = os.path.join(RUN_COMPLETED_ROOT, f"{name}_raw.pkl")

            with open(simple_path, "wb") as f:
                pickle.dump(BenchmarkResults(
                    sha=sha,
                    conda_env=conda_env,
                    values=tuple(simple_results)
                ), f)

            with open(raw_path, "wb") as f:
                pickle.dump([sha, conda_env, raw_results], f)

            self._state.report_finished(TaskType.MEASURE, sha=sha, result=simple_path)
            os.chmod(conda_lib_path, 0o755)
            shutil.rmtree(conda_env)
            log_measure_state(f"{len(results)} / {len(work_orders)} succeeded.")
            self._update_success_dict(sha)

        except BaseException:
            log_measure_state("Failed.")
            traceback.print_exc()

        finally:
            self._core_pool.finish_task()
            if os.path.exists(conda_lib_path):
                os.chmod(conda_lib_path, 0o755)
            clean_backport(torch_path)
