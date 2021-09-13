import datetime
import os
import re
import shutil
from typing import List

from v2.build import build_benchmark_env
from v2.containers import BuildCfg, Commit, History
from v2.logging_subprocess import call
from v2.workspace import (
    make_dirs, BUILD_LOG_ROOT, DATE_FMT, MUTATION_LOCK, REF_REPO_ROOT,
    BENCHMARK_BRANCH_NAME, BENCHMARK_BRANCH_ROOT, BENCHMARK_ENV, BENCHMARK_ENV_BUILT)


_MKL_CONDA_RELEASES = (
    ("", datetime.datetime.strptime("2019-09-16", DATE_FMT)),
    ("2019.5", datetime.datetime.strptime("2019-09-15", DATE_FMT)),
    ("2019.4", datetime.datetime.strptime("2019-05-15", DATE_FMT)),
    ("2019.3", datetime.datetime.strptime("2019-03-15", DATE_FMT)),
    # 2020.2 missing from release notes.
    ("2019.1", datetime.datetime.strptime("2018-11-15", DATE_FMT)),
    ("2019.0", datetime.datetime.strptime("2018-09-15", DATE_FMT)),
)



def _fetch_branch(branch_name: str, branch_path: str) -> None:
    MUTATION_LOCK.get()
    if not os.path.exists(branch_path):
        print(f"Cloning into {branch_path}")
        call(
            f"git clone git@github.com:pytorch/pytorch.git {branch_path}",
            check=True,
            task_name="Checkout PyTorch",
            log_dir=BUILD_LOG_ROOT,
        )

    print(f"Checking out {branch_name}")
    call(
        f"""
        git remote prune origin
        git checkout {branch_name}
        git pull
        git clean -fd
        git submodule sync
        git submodule update --init --recursive
        """,
        shell=True,
        cwd=branch_path,
        check=True,
        task_name="Update PyTorch",
        log_dir=BUILD_LOG_ROOT,
    )


def get_history() -> History:
    lines = []
    sep = "_____PARTITION_____"
    call(
        f"git log --pretty='format:%H %ai {sep} %aN {sep} %aE {sep} %s' | cat",
        shell=True,
        cwd=REF_REPO_ROOT,
        check=True,
        per_line_fn=lambda l: lines.append(l),
        task_name="Git log",
        log_dir=BUILD_LOG_ROOT,
    )

    pattern = re.compile(
        r"^([a-z0-9]{40}) ([0-9\-]{10}) [0-9:]+ [\-0-9]+ " +
        sep + r" (.+) " + sep + r" (.+) " + sep + " (.+)$"
    )

    commits: List[Commit] = []

    python_version = "3.7"
    build_tests = "0"

    for l in lines[::-1]:
        match = pattern.match(l)
        if match:
            sha, date_str, author_name, author_email, msg = match.groups()
            # We're only interested in date for the initial sweep and some
            # version decions, so we don't need to worry about HH:MM:SS.
            date = datetime.datetime.strptime(date_str, DATE_FMT)

            # This is the commit that added Python 3.8 support
            if sha == "86c64440c9169d94bffb58b523da1db00c896703":
                python_version = "3.8"

            # There was a breakage where PyTorch will not build correctly
            # unless tests are also built.
            if sha == "ddff4efa26d527c99cd9892278a32529ddc77e66":
                build_tests = "1"
            elif sha == "61e509b9922f632a9bb89ed06406df93f8bd2da8":
                build_tests = "0"


            for mkl_version, mkl_release_date in _MKL_CONDA_RELEASES:
                if (date - mkl_release_date).total_seconds() >= 0:
                    break

            commits.append(
                Commit(
                    sha,
                    date,
                    date_str,
                    author_name,
                    author_email,
                    msg,
                    BuildCfg(python_version, build_tests, mkl_version),
                )
            )
    return History(tuple(commits))


def fetch_fbcode_warm():
    _fetch_branch("fbcode/warm", REF_REPO_ROOT)


def run() -> None:
    # make_dirs is itempotent, and we need it before we can grab the lock.
    make_dirs()
    fetch_fbcode_warm()

    if os.getenv("CONDA_PREFIX") is None:
        if not os.path.exists(BENCHMARK_ENV_BUILT):
            if os.path.exists(BENCHMARK_BRANCH_ROOT):
                shutil.rmtree(BENCHMARK_BRANCH_ROOT)
            _fetch_branch(BENCHMARK_BRANCH_NAME, BENCHMARK_BRANCH_ROOT)
            build_benchmark_env()
        print(f"source activate {BENCHMARK_ENV}")
