import importlib.abc
import importlib.util
import os
import sys
from typing import cast, Optional, Protocol

from v2.workspace import BENCHMARK_BRANCH_ROOT, BENCHMARK_ENV

assert os.getenv("CONDA_PREFIX") == BENCHMARK_ENV
CI_ROOT_PATH = os.path.join(BENCHMARK_BRANCH_ROOT, "benchmarks", "instruction_counts")



def import_module(*path: str):
    module_path = os.path.join(*path)
    assert os.path.exists(module_path), f"{module_path} does not exist."
    assert module_path.endswith(".py")
    name = "_".join(path)[:-3]

    module_spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(module_spec)
    loader = module_spec.loader
    assert loader is not None
    cast(importlib.abc.Loader, loader).exec_module(module)
    return module

sys.path.insert(0, CI_ROOT_PATH)
_core_utils = import_module(CI_ROOT_PATH, "core", "utils.py")
_definitions_standard = import_module(CI_ROOT_PATH, "definitions", "standard.py")
_execution_cores = import_module(CI_ROOT_PATH, "execution", "cores.py")
_execution_runner = import_module(CI_ROOT_PATH, "execution", "runner.py")
_execution_work = import_module(CI_ROOT_PATH, "execution", "work.py")
assert sys.path.pop(0) == CI_ROOT_PATH

_historic_patch = import_module(BENCHMARK_BRANCH_ROOT, "torch", "utils", "benchmark", "utils", "historic", "patch.py")
clean_backport = _historic_patch.clean_backport
backport = _historic_patch.backport


class CorePoolProtocol(Protocol):
    def __init__(self, slack: int) -> None:
        ...

    def reserve(self, n: int) -> Optional[str]:
        ...

    def release(self, key: str) -> None:
        ...


CorePool: CorePoolProtocol = _execution_cores.CorePool
WorkOrder = _execution_work.WorkOrder
BenchmarkRunner = _execution_runner.Runner

_BENCHMARKS = None
def get_benchmarks():
    """This will write artifacts, so we guard it in a function."""
    global _BENCHMARKS
    if _BENCHMARKS is None:
        _BENCHMARKS = _core_utils.unpack(_definitions_standard.BENCHMARKS)
    return _BENCHMARKS
