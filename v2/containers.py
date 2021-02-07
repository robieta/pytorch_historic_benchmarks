import collections
import dataclasses
import datetime
from typing import Deque, Generic, Iterable, Set, Tuple, TypeVar

from v2.workspace import DATE_FMT


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


@dataclasses.dataclass(frozen=True)
class BuildCfg:
    python_version: str = "3.8"
    build_tests: str = "0"
    mkl_version: str = ""


@dataclasses.dataclass(frozen=True)
class Commit:
    sha: str
    date: datetime.datetime
    date_str: str
    author_name: str
    author_email: str
    msg: str
    build_cfg: BuildCfg


@dataclasses.dataclass(frozen=True)
class History:
    commits: Tuple[Commit, ...]

    def since(self, start_date: str) -> Tuple[Commit, ...]:
        t0 = datetime.datetime.strptime(start_date, DATE_FMT)
        return tuple(
            c for c in self.commits
            if (c.date - t0).total_seconds() >= 0
        )


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    label: Tuple[str, ...]
    language: str
    autograd: str
    runtime: str
    wall_time: float
    instructions: int


@dataclasses.dataclass(frozen=True)
class BenchmarkResults:
    sha: str
    conda_env: str
    values: Tuple[BenchmarkResult, ...]
