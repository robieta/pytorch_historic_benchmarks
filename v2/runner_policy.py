import multiprocessing
from typing import Tuple


NUM_CORES: int = multiprocessing.cpu_count()

# Currently only support DevBig
assert NUM_CORES in (80,)

POOL_SLACK = 6
OPPORTUNISTIC_BUILD_CORE_COUNTS = (20, 16)
BASELINE_BUILD_CORE_COUNT = 12
MAX_BUILD_CORES = int(0.75 * (NUM_CORES - POOL_SLACK))

_KNOWN_HISTORIC_FAILURES = (
    # Any
    (
        # Type promotion rules were added after SWEEP_START.
        ('Pointwise', 'Math', 'add', 'Tensor-Tensor (type promotion)'),

        # Early versions of MLK have illegal accesses
        ('nn Modules', 'Conv1d'),
        ('nn Modules', 'Conv2d'),

        # GeLU was added after SWEEP_START.
        ('training', 'ensemble'),
    ),

    # C++ only
    (
        # Much of the torch::nn API was added after SWEEP_START.
        ('Mesoscale', 'MatMul-Bias-ReLU'),
        ('nn Modules', 'BatchNorm2d'),
        ('nn Modules', 'GroupNorm'),
        ('nn Modules', 'LayerNorm'),
        ('nn Modules', 'MaxPool2d'),
        ('nn Modules', 'ReLU'),
        ('nn Modules', 'Sigmoid'),
        ('training', 'simple'),
    ),
)

def benchmark_may_fail(is_cpp: bool, is_jit: bool, label: Tuple[str, ...]) -> bool:
    return any([
        # JIT format has changed, and old versions can't read new artifacts.
        is_jit,

        # C++ advanced indexing was added after SWEEP_START.
        ('Indexing' in label and is_cpp),

        # Language agnostic failures
        label in _KNOWN_HISTORIC_FAILURES[0],

        # C++ specific failures
        (label in _KNOWN_HISTORIC_FAILURES[1] and is_cpp)
    ])
