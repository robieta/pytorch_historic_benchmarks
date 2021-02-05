import os
import sys
import threading
import time
from typing import cast, Optional, Protocol

from v2.workspace import BENCHMARK_BRANCH_ROOT, BENCHMARK_ENV

assert os.getenv("CONDA_PREFIX") == BENCHMARK_ENV
CI_ROOT_PATH = os.path.join(BENCHMARK_BRANCH_ROOT, "benchmarks", "instruction_counts")


sys.path.insert(0, CI_ROOT_PATH)

# Note:
#   We have to use normal imports rather than importlib, because the latter
#   can result in multiple copies of enums.
from core.api import AutogradMode, RuntimeMode
from core.utils import unpack
from definitions.standard import BENCHMARKS
from execution.cores import CorePool
from execution.runner import Runner as BenchmarkRunner
from execution.work import WorkOrder

assert sys.path.pop(0) == CI_ROOT_PATH

sys.path.insert(0, BENCHMARK_BRANCH_ROOT)
from torch.utils.benchmark.utils.historic.patch import backport, clean_backport
assert sys.path.pop(0) == BENCHMARK_BRANCH_ROOT


class InstrumentedCorePool(CorePool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allocating_thread = {}
        self._active_runners = {}
        self._log = []
        self._thread_type = {}

        self._lock = threading.Lock()
        self._requests = 0
        self._max_cores = 80
        self._max_build_cores = 40
        self._max_small_jobs = 60
        self._job_slots = (
            [f"0-74,{i}" for i in range(74)] +
            [f"0-74,{i},{i}" for i in range(74)] +
            [f"0-74,{i}-{i}" for i in range(74)]
        )
        self._cost = {}

    @property
    def allocated(self):
        return sum(self._cost.values())

    def reserve(self, n: int) -> Optional[str]:
        self._requests += 1
        with self._lock:
            if self.allocated + n > self._max_cores:
                return

            if n <= 2 and len([v for v in self._cost.values() if v <= 2]) >= self._max_small_jobs:
                return

            if n >= 8 and (n + sum([v for v in self._cost.values() if v >= 8])) > self._max_build_cores:
                return

            allocation = self._job_slots.pop()
            self._cost[allocation] = n

        thread_id = threading.get_ident()
        self._log.append(("reserve", n, allocation, thread_id))

        time.sleep(2)

        self._allocating_thread[allocation] = thread_id
        return allocation

    def release(self, key: str) -> None:
        thread_id = self._allocating_thread.pop(key)
        self._log.append(("release", key, thread_id, threading.get_ident()))
        self._job_slots.append(key)
        self._cost.pop(key)

    def register_benchmark_runner(self, benchmark_runner: BenchmarkRunner):
        self._active_runners[threading.get_ident()] = benchmark_runner

    def unregister_benchmark_runner(self):
        benchmark_runner = self._active_runners.pop(threading.get_ident(), None)
        if benchmark_runner is None:
            print(f"Warning: thread {threading.get_ident()} failed to register runner.")
            return

        for allocation in benchmark_runner._active_jobs:
            try:
                self.release(allocation)
            except BaseException:
                pass


_BENCHMARKS = None
def get_benchmarks():
    """This will write artifacts, so we guard it in a function."""
    global _BENCHMARKS
    if _BENCHMARKS is None:
        _BENCHMARKS = unpack(BENCHMARKS)
    return _BENCHMARKS
