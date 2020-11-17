import collections
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from typing import Callable, Dict, List, NoReturn, Optional
import uuid

from . import misc as misc_utils


DATE_FMT = "%Y-%m-%d"
CONDA_ENV_TEMPLATE = "env_{n:0>2}"
MAX_ACTIVE_ENVS = 200

_NAMESPACE_LOCK = threading.Lock()


_SUBPROCESS_REQUEST_INPUT = "SUBPROCESS_REQUESTS_INPUT: {prompt}"
def request_input(prompt: str) -> str:
    return input(_SUBPROCESS_REQUEST_INPUT.format(prompt=prompt))


def subprocess_requested_input(line: str) -> Optional[str]:
    pattern = _SUBPROCESS_REQUEST_INPUT.format(prompt="(.+)$")
    match = re.match(pattern, line)
    if match:
        return match.group(0)


class BuildConfig:
    def __init__(self, history_path):
        with open(history_path, "rt") as f:
            history = json.load(f)

        self._python_version = collections.defaultdict(lambda: "3.8")
        self._build_tests = collections.defaultdict(lambda: "0")
        self._mkl_version = collections.defaultdict(lambda: "")

        for sha, _, _, _, _ in history:
            if sha == "86c64440c9169d94bffb58b523da1db00c896703":
                break
            self._python_version[sha] = "3.7"

        build_tests = False
        for sha, _, _, _, _ in history:
            if sha == "ddff4efa26d527c99cd9892278a32529ddc77e66":
                build_tests = True

            if build_tests:
                self._build_tests[sha] = "1"

            if sha == "61e509b9922f632a9bb89ed06406df93f8bd2da8":
                build_tests = False
                break

        mkl_conda_releases = (
            ("", "2019-09-16"),
            ("2019.5", "2019-09-15"),
            ("2019.4", "2019-05-15"),
            ("2019.3", "2019-03-15"),
            # 2020.2 missing from release notes.
            ("2019.1", "2018-11-15"),
            ("2019.0", "2018-09-15"),
        )

        for sha, date, _, _, _ in history[18000:]:
            version = ""
            for version, d in mkl_conda_releases:
                dt = (
                    datetime.datetime.strptime(date, DATE_FMT) -
                    datetime.datetime.strptime(d, DATE_FMT)
                )
                if dt.total_seconds() >= 0:
                    break
            if version:
                self._mkl_version[sha] = version

    def python_version(self, sha_or_branch):
        return self._python_version[sha_or_branch]

    def build_tests(self, sha_or_branch):
        return self._build_tests[sha_or_branch]

    def mkl_version(self, sha_or_branch):
        return self._mkl_version[sha_or_branch]


class PytorchBuildHelper:
    def __init__(
        self,
        root_dir: str,
        clean: bool = False,
        soft_clean: bool = False,
        main_loop = True,
    ) -> None:
        self._root_dir = root_dir
        self._log_dir = os.path.join(self._root_dir, "logs")
        self._env_dir = os.path.join(self._root_dir, "envs")
        self._build_dir = os.path.join(self._root_dir, "build")
        self._clean_checkout = os.path.join(self._root_dir, "pytorch")
        self._git_history_path = os.path.join(self._root_dir, "git_history.json")
        self._cannot_build_path = os.path.join(self._root_dir, "cannot_build.txt")

        if main_loop:
            if clean and os.path.exists(root_dir):
                shutil.rmtree(root_dir)

            elif soft_clean:
                for i in [self._log_dir, self._env_dir, self._build_dir]:
                    if os.path.exists(i):
                        shutil.rmtree(i)

            os.makedirs(self._log_dir, exist_ok=True)
            os.makedirs(self._env_dir, exist_ok=True)
            os.makedirs(self._build_dir, exist_ok=True)
            self._checkout_pytorch()

        else:
            # Debug and ad-hoc analysis.
            assert not clean, "Only the main loop may specify `clean`."
            assert not soft_clean, "Only the main loop may specify `soft_clean`."
            assert os.path.exists(self._git_history_path), "No git history"

        self._build_config = BuildConfig(self._git_history_path)
        with open(self._git_history_path, "rt") as f:
            self._git_history = [
                (sha, datetime.datetime.strptime(date, DATE_FMT), author_name, author_email, msg)
                for sha, date, author_name, author_email, msg in json.load(f)
            ]

        with open(self._cannot_build_path, "rt") as f:
            self._unbuildable = set(f.read().splitlines(keepends=False))

    def subprocess_call(
        self,
        args: str,
        shell: bool = False,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        check: bool = False,
        timeout: Optional[float] = None,
        per_line_fn: Optional[Callable[[str], NoReturn]] = None,
        conda_env: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        if task_name is not None:
            print(f"BEGIN: {task_name}")

        cmd = []
        def add_to_cmd(entry: List[str]) -> None:
            if not entry:
                return

            if cmd and cmd[-1][-1] != ";":
                cmd.append("&&")

            cmd.extend(entry)

        if conda_env is not None:
            add_to_cmd(["source", "activate", conda_env])

        if env is not None:
            # For some reason, the env arg doesn't play nice with CCache and
            # passing it will cause caching to not work.
            for k, v in env.items():
                add_to_cmd(["export", f"{k}={repr(v)}"])

        for l in args.splitlines(keepends=False):
            add_to_cmd(shlex.split(l.strip(), comments=True))

        now = datetime.datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S")
        log_prefix = f"{now}__{uuid.uuid4()}"

        cleanup_logs = True
        summary = os.path.join(self._log_dir, f"{log_prefix}_summary.txt")
        stdout = os.path.join(self._log_dir, f"{log_prefix}_stdout.log")
        stderr = os.path.join(self._log_dir, f"{log_prefix}_stderr.log")

        with open(summary, "wt") as f:
            f.write(
                f"Cmd:\n{textwrap.dedent(args).strip()}\n\n"
                f"Parsed:\n{' '.join(cmd)}\n\n"
                f"Stdout: {stdout}\n"
                f"Stderr: {stderr}\n"
                f"Env:\n{json.dumps(env or {}, indent=4)}\n\n"
            )

        stdout_f = open(stdout, "wb")
        stderr_f = open(stderr, "wb")
        stdout_f_read = open(stdout, "rb")

        try:
            proc = subprocess.Popen(
                misc_utils.list2cmdline(cmd) if shell else cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                shell=shell,
                cwd=cwd or os.getcwd(),
            )

            start_time = time.time()
            while True:
                stdout_lines = stdout_f_read.read().decode("utf-8")
                if stdout_lines:
                    for l in stdout_lines.splitlines(keepends=False):
                        if per_line_fn:
                            per_line_fn(l)
                        # TODO: communicate.

                retcode = proc.poll()
                if retcode is not None:
                    break

                if timeout and time.time() - start_time >= timeout:
                    proc.terminate()
                    print("Cmd timed out")
                    retcode = 1
                    break

                time.sleep(0.001)

            if retcode:
                cleanup_logs = False
                print(f"Cmd failed. Logs: {summary}")

        except KeyboardInterrupt:
            proc.terminate()
            raise

        except:
            cleanup_logs = False
            print(f"Cmd failed. Logs: {summary}")
            raise

        finally:
            stdout_f.close()
            stderr_f.close()
            stdout_f_read.close()
            if cleanup_logs:
                os.remove(summary)
                os.remove(stdout)
                os.remove(stderr)

        if not retcode and task_name is not None:
            print(f"DONE:  {task_name}")

        assert not retcode or not check, f"retcode: {retcode}"
        return retcode

    def get_history_since(self, start_date: str):
        t0 = datetime.datetime.strptime(start_date, DATE_FMT)
        output = []
        for sha, date, author_name, author_email, msg in self._git_history:
            if output or (date - t0).total_seconds() >= 0:
                output.append((sha, date, author_name, author_email, msg))
        return output

    def _checkout_pytorch(self):
        if not os.path.exists(self._clean_checkout):
            self.subprocess_call(
                f"git clone git@github.com:pytorch/pytorch.git {self._clean_checkout}",
                check=True,
                task_name="Checkout PyTorch",
            )

        self.subprocess_call(
            """
            git pull
            git checkout fbcode/warm
            git clean -fd
            git submodule sync
            git submodule update --init --recursive
            """,
            shell=True,
            cwd=self._clean_checkout,
            check=True,
            task_name="Update PyTorch",
        )

        lines = []
        sep = "_____PARTITION_____"
        self.subprocess_call(
            f"git log --pretty='format:%H %ai {sep} %aN {sep} %aE {sep} %s' | cat",
            shell=True,
            cwd=self._clean_checkout,
            check=True,
            per_line_fn=lambda l: lines.append(l),
        )

        history = []
        pattern = re.compile(
            r"^([a-z0-9]{40}) ([0-9\-]{10}) [0-9:]+ [\-0-9]+ " +
            sep + r" (.+) " + sep + r" (.+) " + sep + " (.+)$"
        )
        for l in lines[::-1]:
            match = pattern.match(l)
            if match:
                sha, date, author_name, author_email, msg = match.groups()
                # We're only interested in date for the initial sweep, so we
                # don't need to worry about HH:MM:SS.
                date = datetime.datetime.strptime(date, DATE_FMT).strftime(DATE_FMT)
                history.append((sha, date, author_name, author_email, msg))

        assert len(history)
        with open(self._git_history_path, "wt") as f:
            json.dump(history, f)

        with open(self._cannot_build_path, "at") as f:
            pass

    def build_clean(self, checkout, show_progress=True, taskset_cores=None, nice=None, max_jobs=None):
        if self.unbuildable(checkout):
            print(f"{checkout} is known to be unbuildable")
            return

        try:
            pytorch_path = os.path.join(self._build_dir, f"pytorch_{uuid.uuid4()}")
            conda_env = self._make_conda_env(checkout)
            shutil.copytree(self._clean_checkout, pytorch_path)

            self.subprocess_call(
                f"""
                retry () {{ $* || (sleep 1 && $*) || (sleep 2 && $*); }}

                git checkout {checkout}
                git clean -fd

                # Sometimes QNNPack doesn't sync. I have no idea why.
                rm -rf third_party/QNNPACK
                retry git submodule sync

                # History for XNNPack has changed, so this will fail in February/March
                retry git -c submodule."third_party/XNNPACK".update=none submodule update --init --recursive
                retry git submodule update --init --recursive || true
                """,
                shell=True,
                cwd=pytorch_path,
                check=True,
                conda_env=conda_env,
                task_name="(pre) Build PyTorch",
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
            retcode = self.subprocess_call(
                f"""
                # CCACHE variables are generally in `.bashrc`
                source ~/.bashrc
                which c++ | awk '{{print "which c++: "$1}}'

                {taskset_str}{nice_str}python -u setup.py clean
                {taskset_str}{nice_str}python -u setup.py install
                echo BUILD_DONE
                """,
                shell=True,
                cwd=pytorch_path,
                env={
                    "USE_DISTRIBUTED": "0",
                    "BUILD_TEST": self._build_config.build_tests(checkout),
                    "USE_CUDA": "0",
                    "USE_FBGEMM": "0",
                    "USE_NNPACK": "0",
                    "USE_QNNPACK": "0",
                    "BUILD_CAFFE2_OPS": "0",
                    "REL_WITH_DEB_INFO": "1",
                    "MKL_THREADING_LAYER": "GNU",
                    "MAX_JOBS": "" if max_jobs is None else str(max_jobs),

                    "CFLAGS": "-Wno-error=stringop-truncation",
                },
                per_line_fn=per_line_fn if show_progress else None,
                conda_env=conda_env,
                task_name="Build PyTorch",
            )

            if retcode:
                # import pdb
                # pdb.set_trace()
                self.mark_unbuildable(checkout)
                shutil.rmtree(conda_env)
                return

            retcode = self.subprocess_call(
                'python -c "import torch;print(torch.__file__)"',
                shell=True,
                cwd=self._root_dir,
                conda_env=conda_env,
                task_name="(check) Build PyTorch",
            )

            if retcode:
                self.mark_unbuildable(checkout)
                shutil.rmtree(conda_env)
                return

            return conda_env
        except KeyboardInterrupt:
            print(f"Build stopped: {checkout}")
            raise

        finally:
            if os.path.exists(pytorch_path):
                shutil.rmtree(pytorch_path)

    def mark_unbuildable(self, sha_or_branch):
        with open(self._cannot_build_path, "at") as f:
            f.write(f"{sha_or_branch}\n")
        self._unbuildable.add(sha_or_branch)

    def unbuildable(self, sha_or_branch):
        return sha_or_branch in self._unbuildable

    def _make_conda_env(self, sha_or_branch: Optional[str] = None):
        with _NAMESPACE_LOCK:
            active_envs = set(os.listdir(self._env_dir))
            for i in range(MAX_ACTIVE_ENVS):
                env_name = CONDA_ENV_TEMPLATE.format(n=i)
                if env_name not in active_envs:
                    break
            else:
                raise ValueError("Failed to create env. Too many already exist.")

            env_path = os.path.join(self._env_dir, env_name)
            py_version = self._build_config.python_version(sha_or_branch)
            mkl_version = self._build_config.mkl_version(sha_or_branch)
            mkl_spec = f"=={mkl_version}" if mkl_version else ""

            self.subprocess_call(
                f"conda create --no-default-packages -y --prefix {env_path} python={py_version}",
                shell=True,
                check=True,
                task_name=f"Conda env creation: {env_name}"
            )

        self.subprocess_call(
            f"""
            conda install -y numpy ninja pyyaml mkl{mkl_spec} mkl-include setuptools cmake cffi hypothesis typing_extensions pybind11
            pip install cppimport
            """,
            shell=True,
            check=True,
            task_name=f"Conda env install: {env_name}",
            conda_env=env_path,
        )

        return env_path
