import datetime
import enum
import multiprocessing
import os
import threading
import time
from typing import Optional, Tuple

from v2.workspace import DATE_FMT, THROTTLE_FILE


NUM_CORES: int = multiprocessing.cpu_count()
POOL_SLACK: int = 8
BUILD_WORKERS = 12  # tar takes a long time, but only takes a single core.
TEST_WORKERS = 5


# Currently only support DevBig
assert NUM_CORES in (56, 80)


class TaskType(enum.Enum):
    BUILD = 0
    MEASURE = 1
    ARCHIVE = 2


class SharedCorePool:
    def __init__(self):
        # For ETA estimation.
        self._num_cores = NUM_CORES

        self._log = []
        self._thread_type = {}
        self._lock = threading.Lock()

        self._requests = 0
        self._last_allocation_by_type = {
            TaskType.BUILD: -1,
            TaskType.MEASURE: -1,
            TaskType.ARCHIVE: -1,
        }
        self._cost = {}
        self._max_total_cores = NUM_CORES - POOL_SLACK
        self._max_cores_by_type = {
            TaskType.BUILD: int(0.6 * NUM_CORES),
            TaskType.MEASURE: int(0.6 * NUM_CORES),
            TaskType.ARCHIVE: NUM_CORES,
        }
        self._allocations_by_type = {
            TaskType.BUILD: 0,
            TaskType.MEASURE: 0,
            TaskType.ARCHIVE: 0,
        }
        self._allocation_stagger = {
            TaskType.BUILD: 180,
            TaskType.MEASURE: 1,
            TaskType.ARCHIVE: 0,
        }
        self._job_slots = (
            [f"0-{self._max_total_cores},{i}" for i in range(self._max_total_cores)] +
            [f"0-{self._max_total_cores},{i},{i}" for i in range(self._max_total_cores)] +
            [f"0-{self._max_total_cores},{i}-{i}" for i in range(self._max_total_cores)]
        )
        self._allocating_thread = {}

    @property
    def allocated(self):
        return sum(self._cost.values())

    def begin_task(self, t: TaskType):
        self._thread_type[threading.get_ident()] = t

    def finish_task(self):
        thread_id = threading.get_ident()
        for allocation, allocation_thread_id in list(self._allocating_thread.items()):
            if allocation_thread_id == thread_id:
                self.release(allocation)
        t: TaskType = self._thread_type.pop(thread_id)

    def reserve(self, n: int) -> Optional[str]:
        self._requests += 1
        thread_id = threading.get_ident()
        t: TaskType = self._thread_type[thread_id]

        if time.time() - self._last_allocation_by_type[t] < self._allocation_stagger[t]:
            return

        if self.allocated + n > self._max_total_cores:
            return

        if self._allocations_by_type[t] + n > self._max_cores_by_type[t]:
            return

        if os.path.exists(THROTTLE_FILE):
            return

        with self._lock:
            allocation = self._job_slots.pop()

            self._cost[allocation] = n
            self._allocations_by_type[t] += n
            self._last_allocation_by_type[t] = time.time()
            self._allocating_thread[allocation] = thread_id
            self._log.append(("reserve", n, allocation, thread_id, t))

            return allocation

    def release(self, key: str) -> None:
        thread_id = self._allocating_thread.pop(key)
        t = self._thread_type[thread_id]
        self._log.append(("release", key, thread_id, threading.get_ident(), t))

        self._job_slots.append(key)
        n = self._cost.pop(key)
        self._allocations_by_type[t] -= n
