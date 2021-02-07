import multiprocessing
import os
import re
import threading
import shutil
import sys
from typing import Optional
import uuid

from v2.containers import BuildCfg
from v2.logging_subprocess import call
from v2.workspace import (
    CANNOT_BUILD,
    BENCHMARK_BRANCH_NAME, BENCHMARK_BRANCH_ROOT, BENCHMARK_ENV, BENCHMARK_ENV_BUILT,
    BUILD_LOG_ROOT, BUILD_IN_PROGRESS_ROOT,
    MUTATION_LOCK, REF_REPO_ROOT)

_NAMESPACE_LOCK = threading.Lock()
_CONDA_ENV_TEMPLATE = "env_{n:0>2}"
_MAX_ACTIVE_ENVS = 50


class _Unbuildable:
    def __init__(self):
        self._known_unbuildable = None

    def _lazy_init(self):
        if self._known_unbuildable is None:
            with open(CANNOT_BUILD, "at") as f:
                pass

            with open(CANNOT_BUILD, "rt") as f:
                self._known_unbuildable = set(f.read().splitlines(keepends=False))

    def check(self, checkout: str) -> bool:
        self._lazy_init()
        return checkout in self._known_unbuildable

    def update(self, checkout: str) -> None:
        self._lazy_init()
        MUTATION_LOCK.get()
        if checkout not in self._known_unbuildable:
            with open(CANNOT_BUILD, "at") as f:
                f.write(f"{checkout}\n")
        self._known_unbuildable.add(checkout)

_UnbuildableSingleton = _Unbuildable()
check_unbuildable = _UnbuildableSingleton.check
mark_unbuildable = _UnbuildableSingleton.update


def make_conda_env(
    env_path: Optional[str] = None,
    build_cfg: BuildCfg = BuildCfg(),
):
    MUTATION_LOCK.get()
    with _NAMESPACE_LOCK:
        if env_path is None:
            active_envs = set(os.listdir(BUILD_IN_PROGRESS_ROOT))
            for i in range(_MAX_ACTIVE_ENVS):
                env_name = _CONDA_ENV_TEMPLATE.format(n=i)
                if env_name not in active_envs:
                    break
            else:
                raise ValueError("Failed to create env. Too many already exist.")

            env_path = os.path.join(BUILD_IN_PROGRESS_ROOT, env_name)
        else:
            env_name = "custom"

        mkl_spec = f"=={build_cfg.mkl_version}" if build_cfg.mkl_version else ""
        call(
            f"conda create --no-default-packages -y --prefix {env_path} python={build_cfg.python_version}",
            shell=True,
            check=True,
            task_name=f"Conda env creation: {env_name}",
            log_dir=BUILD_LOG_ROOT,
        )

    call(
        f"""
        echo ADD_INTEL
        conda config --env --add channels intel

        echo MAIN_INSTALL
        conda install -y numpy ninja pyyaml mkl{mkl_spec} mkl-include setuptools cmake cffi hypothesis typing_extensions pybind11 ipython

        echo GLOG_INSTALL
        conda install -y -c conda-forge glog

        echo INSTALL_VALGRIND
        conda install -y -c conda-forge valgrind
        """,
        shell=True,
        check=True,
        task_name=f"Conda env install: {env_name}",
        conda_env=env_path,
        log_dir=BUILD_LOG_ROOT,
    )

    return env_path


def _build(
    repo_path: str,
    checkout: Optional[str],
    setup_mode: str,
    conda_env: str,
    build_cfg: BuildCfg,
    show_progress: bool,
    taskset_cores: Optional[str],
    nice: Optional[str],
    max_jobs: Optional[int],
) -> int:
    assert setup_mode in ("develop", "install")

    no_xnnpack = '-c submodule."third_party/XNNPACK".update=none'
    no_nervanagpu = '-c submodule."third_party/nervanagpu".update=none'
    call(
        f"""
        retry () {{ $* || (sleep 1 && $*) || (sleep 2 && $*); }}

        git checkout .
        git clean -fd
        git checkout .
        git checkout {checkout}
        git clean -fd

        # `git submodule sync` doesn't sync submodule submodules, which can
        # cause build failures. So instead we just start over.
        rm -rf third_party/*
        git checkout third_party
        retry git submodule sync

        # History for XNNPack has changed, so this will fail in February/March
        retry git {no_xnnpack} {no_nervanagpu} submodule update --init --recursive
        """,
        shell=True,
        cwd=repo_path,
        check=True,
        conda_env=conda_env,
        task_name="(pre) Build PyTorch",
        log_dir=BUILD_LOG_ROOT,
    )

    call(
        f"""
        retry () {{ $* || (sleep 1 && $*) || (sleep 2 && $*); }}
        retry git submodule update --init --recursive
        """,
        shell=True,
        cwd=repo_path,
        check=False,
        conda_env=conda_env,
        task_name="(pre) Build PyTorch",
        log_dir=BUILD_LOG_ROOT,
    )

    progress_pattern = re.compile(r"^\[[0-9]+/[0-9]+\]\s.+$")
    def per_line_fn(l):
        if progress_pattern.search(l):
            print(f"\r{l.strip()[:120]:<120}", end="")
            sys.stdout.flush()

        if "BUILD_DONE" in l:
            print("\r")

    taskset_str = f"taskset --cpu-list {taskset_cores} " if taskset_cores else ""
    nice_str = f"nice -n {nice} " if nice is not None else ""
    retcode = call(
        f"""
        # CCACHE variables are generally in `.bashrc`
        source ~/.bashrc
        which c++ | awk '{{print "which c++: "$1}}'

        {taskset_str}{nice_str}python -u setup.py clean
        {taskset_str}{nice_str}python -u setup.py {setup_mode}
        echo BUILD_DONE
        """,
        shell=True,
        cwd=repo_path,
        env={
            "USE_DISTRIBUTED": "0",
            "BUILD_TEST": build_cfg.build_tests,
            "USE_CUDA": "0",
            "USE_FBGEMM": "0",
            "USE_NNPACK": "0",
            "USE_QNNPACK": "0",
            "USE_XNNPACK": "0",
            "BUILD_CAFFE2_OPS": "0",
            "REL_WITH_DEB_INFO": "1",
            "MKL_THREADING_LAYER": "GNU",
            "USE_NUMA": "0",
            "MAX_JOBS": "" if max_jobs is None else str(max_jobs),

            "CFLAGS": f"-Wno-error=stringop-truncation",  # -I{conda_env}/include/",
        },
        per_line_fn=per_line_fn if show_progress else None,
        conda_env=conda_env,
        task_name="Build PyTorch",
        log_dir=BUILD_LOG_ROOT,
    )

    if not retcode:
        retcode = call(
            'python -c "import torch"',
            shell=True,
            conda_env=conda_env,
            task_name="Test PyTorch",
            log_dir=BUILD_LOG_ROOT,
        )

    if retcode:
        # print(f"Debug unbuildable {checkout}  {conda_env}")
        mark_unbuildable(checkout)
        shutil.rmtree(conda_env)

    return retcode


def build_benchmark_env():
    MUTATION_LOCK.get()
    if os.path.exists(BENCHMARK_ENV_BUILT):
        return

    if not os.path.exists(BENCHMARK_ENV):
        shutil.rmtree(BENCHMARK_ENV, ignore_errors=True)

    make_conda_env(env_path=BENCHMARK_ENV, build_cfg=BuildCfg())

    # By default, build will try to take over all cores. However, this can
    # lead to OOM during some of the memory-intensive parts of compilation.
    max_jobs = max(int(multiprocessing.cpu_count() * 0.9), 1)

    retcode = _build(
        BENCHMARK_BRANCH_ROOT,
        BENCHMARK_BRANCH_NAME,
        "develop",
        BENCHMARK_ENV,
        build_cfg=BuildCfg(),
        show_progress=True,
        taskset_cores=None,
        nice=None,
        max_jobs=max_jobs,
    )

    assert not retcode
    with open(BENCHMARK_ENV_BUILT, "wt") as f:
        pass


def build_clean(
    checkout,
    build_cfg: BuildCfg,
    show_progress: bool = True,
    taskset_cores: Optional[str] = None,
    nice: Optional[str] = None,
    max_jobs: Optional[int] = None,
) -> Optional[str]:
    MUTATION_LOCK.get()
    if check_unbuildable(checkout):
        print(f"{checkout} is known to be unbuildable.")
        return

    try:
        repo_path = os.path.join(BUILD_IN_PROGRESS_ROOT, f"pytorch_{uuid.uuid4()}")
        shutil.copytree(REF_REPO_ROOT, repo_path)
        conda_env = make_conda_env(build_cfg=build_cfg)
        retcode = _build(
            repo_path,
            checkout,
            "install",
            conda_env,
            build_cfg,
            show_progress,
            taskset_cores,
            nice,
            max_jobs,
        )

        return None if retcode else conda_env

    except KeyboardInterrupt:
        print(f"Build stopped: {checkout}")
        raise

    finally:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
