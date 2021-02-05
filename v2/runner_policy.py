import datetime
import multiprocessing
import threading
from typing import Tuple

from v2.workspace import DATE_FMT


NUM_CORES: int = multiprocessing.cpu_count()

# Currently only support DevBig
assert NUM_CORES in (80,)

POOL_SLACK = 8
OPPORTUNISTIC_BUILD_CORE_COUNTS = (20, 16)
BASELINE_BUILD_CORE_COUNT = 8


_KNOWN_FAILURES = {
    # is_cpp, is_fwd+bwd, is_jit, label

    # Unknown failure date.
    "0000-00-00": (
        (False, False, False,  ('training', 'ensemble')),
        (False, True,  False,  ('training', 'ensemble')),
    ),

    "2019-06-19": (
        (True,  False, False, ('Mesoscale', 'MatMul-Bias-ReLU')),
        (False, False, False, ('Pointwise', 'Math', 'add', 'Tensor-Tensor (type promotion)')),
        (True,  False, False, ('Pointwise', 'Math', 'add', 'Tensor-Tensor (type promotion)')),
        (True,  False, False, ('nn Modules', 'BatchNorm2d')),
        (False, False, False, ('nn Modules', 'Conv1d')),
        (True,  False, False, ('nn Modules', 'Conv1d')),
        (False, False, False, ('nn Modules', 'Conv2d')),
        (True,  False, False, ('nn Modules', 'Conv2d')),
        (True,  False, False, ('nn Modules', 'LayerNorm')),
        (True,  False, False, ('nn Modules', 'MaxPool2d')),
        (True,  False, False, ('nn Modules', 'ReLU')),
        (True,  False, False, ('nn Modules', 'Sigmoid')),
    ),

    "2019-11-02": (
        (True,  False, False, ('Indexing', 'Tensor index')),
        (False, False, True,  ('Indexing', 'Tensor index')),
        (True,  False, True,  ('Indexing', 'Tensor index')),
        (True,  False, False, ('Indexing', '[...]')),
        (True,  False, False, ('Indexing', '[0, 0, 0]')),
        (True,  False, False, ('Indexing', '[0, 0]')),
        (True,  False, False, ('Indexing', '[0]')),
        (True,  False, False, ('Indexing', '[:]')),
        (True,  False, False, ('Indexing', '[False]')),
        (True,  False, False, ('Indexing', '[None]')),
        (True,  False, False, ('Indexing', '[True]')),
        (False, False, True,  ('Mesoscale', 'MatMul-Bias-ReLU')),
        (True,  False, True,  ('Mesoscale', 'MatMul-Bias-ReLU')),
        (False, False, True,  ('Mesoscale', 'Off diagonal indices')),
        (True,  False, True,  ('Mesoscale', 'Off diagonal indices')),
        (False, False, True,  ('Pointwise', 'Math', 'add', 'Tensor-Tensor')),
        (True,  False, True,  ('Pointwise', 'Math', 'add', 'Tensor-Tensor')),
        (False, False, True,  ('nn Modules', 'BatchNorm2d')),
        (True,  False, True,  ('nn Modules', 'BatchNorm2d')),
        (False, False, True,  ('nn Modules', 'Conv1d')),
        (True,  False, True,  ('nn Modules', 'Conv1d')),
        (False, False, True,  ('nn Modules', 'Conv2d')),
        (True,  False, True,  ('nn Modules', 'Conv2d')),
        (True,  False, False, ('nn Modules', 'GroupNorm')),
        (False, False, True,  ('nn Modules', 'GroupNorm')),
        (True,  False, True,  ('nn Modules', 'GroupNorm')),
        (False, False, True,  ('nn Modules', 'LayerNorm')),
        (True,  False, True,  ('nn Modules', 'LayerNorm')),
        (False, False, True,  ('nn Modules', 'Linear')),
        (True,  False, True,  ('nn Modules', 'Linear')),
        (False, False, True,  ('nn Modules', 'MaxPool2d')),
        (True,  False, True,  ('nn Modules', 'MaxPool2d')),
        (False, False, True,  ('nn Modules', 'ReLU')),
        (True,  False, True,  ('nn Modules', 'ReLU')),
        (False, False, True,  ('nn Modules', 'Sigmoid')),
        (True,  False, True,  ('nn Modules', 'Sigmoid')),
        (True,  False, False, ('training', 'ensemble')),
        (True,  True,  False, ('training', 'ensemble')),
        (False, False, True,  ('training', 'ensemble')),
        (True,  False, True,  ('training', 'ensemble')),
        (False, True,  True,  ('training', 'ensemble')),
        (True,  True,  True,  ('training', 'ensemble')),
        (True,  False, False, ('training', 'simple')),
        (True,  True,  False, ('training', 'simple')),
        (False, False, True,  ('training', 'simple')),
        (True,  False, True,  ('training', 'simple')),
        (False, True,  True,  ('training', 'simple')),
        (True,  True,  True,  ('training', 'simple')),
    )
}


def get_exclude_sets(date: str):
    datetime.datetime.strptime(date, DATE_FMT)
    may_fail = set()
    will_fail = set()
    for d, known_failures in _KNOWN_FAILURES.items():
        for i in known_failures:
            may_fail.add(i)
            if date <= d:
                will_fail.add(i)
    return may_fail, will_fail
