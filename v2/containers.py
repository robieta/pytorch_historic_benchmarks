import dataclasses
import datetime
from typing import Tuple

from v2.workspace import DATE_FMT


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
