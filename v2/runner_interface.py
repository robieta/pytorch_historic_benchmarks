import os
import sys
import threading
from typing import cast, Optional, Protocol

from v2.workspace import BENCHMARK_BRANCH_ROOT, BENCHMARK_ENV

assert os.getenv("CONDA_PREFIX") == BENCHMARK_ENV
CI_ROOT_PATH = os.path.join(BENCHMARK_BRANCH_ROOT, "benchmarks", "instruction_counts")


sys.path.insert(0, CI_ROOT_PATH)

# Note:
#   We have to use normal imports rather than importlib, because the latter
#   can result in multiple copies of enums.
from core.api import RuntimeMode
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

    def reserve(self, n: int) -> Optional[str]:
        allocation = super().reserve(n)
        if allocation is not None:
            thread_id = threading.get_ident()
            self._allocating_thread[allocation] = thread_id
            self._log.append(("reserve", n, allocation, thread_id))
        return allocation

    def release(self, key: str) -> None:
        super().release(key)
        thread_id = self._allocating_thread.pop(key)
        self._log.append(("release", key, thread_id, threading.get_ident()))

    def register_benchmark_runner(self, benchmark_runner: BenchmarkRunner):
        self._active_runners[threading.get_ident()] = benchmark_runner

    def unregister_benchmark_runner(self):
        benchmark_runner = self._active_runners.pop(threading.get_ident(), None)
        if runner is None:
            print(f"Warning: thread {threading.get_ident()} failed to register runner.")
            return

        for allocation in benchmark_runner._active_jobs:
            try:
                self.release(allocation)
            except KeyError:
                pass


_BENCHMARKS = None
def get_benchmarks():
    """This will write artifacts, so we guard it in a function."""
    global _BENCHMARKS
    if _BENCHMARKS is None:
        _BENCHMARKS = unpack(BENCHMARKS)
    return _BENCHMARKS
