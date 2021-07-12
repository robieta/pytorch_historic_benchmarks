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
from core.expand import materialize
from definitions.standard import BENCHMARKS
from execution.runner import CorePool
from execution.runner import Runner as BenchmarkRunner
from execution.work import WorkOrder

assert sys.path.pop(0) == CI_ROOT_PATH

sys.path.insert(0, BENCHMARK_BRANCH_ROOT)
from torch.utils.benchmark.utils.historic.patch import backport
assert sys.path.pop(0) == BENCHMARK_BRANCH_ROOT


_BENCHMARKS = None
def get_benchmarks():
    """This will write artifacts, so we guard it in a function."""
    global _BENCHMARKS
    if _BENCHMARKS is None:
        _BENCHMARKS = materialize(BENCHMARKS)
    return _BENCHMARKS
