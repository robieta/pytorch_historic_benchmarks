import datetime
import json
import os
import shlex
import subprocess
import tempfile
import textwrap
import time
from typing import Callable, Dict, List, NoReturn, Optional
import uuid


# Copied from: https://github.com/python/cpython/blob/1ed83adb0e95305af858bd41af531e487f54fee7/Lib/subprocess.py#L529
def list2cmdline(seq):
    """
    Translate a sequence of arguments into a command line
    string, using the same rules as the MS C runtime:
    1) Arguments are delimited by white space, which is either a
       space or a tab.
    2) A string surrounded by double quotation marks is
       interpreted as a single argument, regardless of white space
       contained within.  A quoted string can be embedded in an
       argument.
    3) A double quotation mark preceded by a backslash is
       interpreted as a literal double quotation mark.
    4) Backslashes are interpreted literally, unless they
       immediately precede a double quotation mark.
    5) If backslashes immediately precede a double quotation mark,
       every pair of backslashes is interpreted as a literal
       backslash.  If the number of backslashes is odd, the last
       backslash escapes the next double quotation mark as
       described in rule 3.
    """

    # See
    # http://msdn.microsoft.com/en-us/library/17w5ykft.aspx
    # or search http://msdn.microsoft.com for
    # "Parsing C++ Command-Line Arguments"
    result = []
    needquote = False
    for arg in map(os.fsdecode, seq):
        bs_buf = []

        # Add a space to separate this argument from the others
        if result:
            result.append(' ')

        needquote = (" " in arg) or ("\t" in arg) or not arg
        if needquote:
            result.append('"')

        for c in arg:
            if c == '\\':
                # Don't know if we need to double yet.
                bs_buf.append(c)
            elif c == '"':
                # Double backslashes.
                result.append('\\' * len(bs_buf)*2)
                bs_buf = []
                result.append('\\"')
            else:
                # Normal char
                if bs_buf:
                    result.extend(bs_buf)
                    bs_buf = []
                result.append(c)

        # Add remaining backslashes, if any.
        if bs_buf:
            result.extend(bs_buf)

        if needquote:
            result.extend(bs_buf)
            result.append('"')

    return ''.join(result)


def now_str():
    return datetime.datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S")


def skip_line(l: str):
    pass


def call(
    args: str,
    shell: bool = False,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = False,
    timeout: Optional[float] = None,
    per_line_fn: Callable[[str], NoReturn] = skip_line,
    conda_env: Optional[str] = None,
    task_name: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> int:
    task_name = task_name or "Unnamed task"
    cleanup_logs: bool = True
    log_prefix = f"{now_str()}__{uuid.uuid4()}"
    concrete_log_dir = log_dir or tempfile.gettempdir()
    summary: str = os.path.join(concrete_log_dir, f"{log_prefix}_summary.txt")
    progress: str = os.path.join(concrete_log_dir, f"{log_prefix}_progress.txt")
    stdout: str = os.path.join(concrete_log_dir, f"{log_prefix}_stdout.log")
    stderr: str = os.path.join(concrete_log_dir, f"{log_prefix}_stderr.log")

    def write_to_progress(l: str, end="\n"):
        with open(progress, "at") as f:
            f.write(f"{l}{end}")

    write_to_progress(f"{now_str()}  BEGIN: {task_name}")

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
        # Popen has an env field, but it prevents the subprocess from inheriting
        # any environment variables not present in `env`. Generally this is not
        # desired since things like `PATH` are necessary and oft overlooked.
        for k, v in env.items():
            add_to_cmd(["export", f"{k}={repr(v)}"])

    for l in args.splitlines(keepends=False):
        add_to_cmd(shlex.split(l.strip(), comments=True))

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
            list2cmdline(cmd) if shell else cmd,
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
                    per_line_fn(l)

            retcode = proc.poll()
            if retcode is not None:
                break

            if timeout and time.time() - start_time >= timeout:
                proc.terminate()
                write_to_progress("Cmd timed out")
                retcode = 1
                break

            time.sleep(0.001)

        if retcode:
            cleanup_logs = False
            write_to_progress(f"Cmd failed. Logs: {summary}")

        return retcode

    except KeyboardInterrupt:
        proc.terminate()
        raise

    except:
        cleanup_logs = False
        write_to_progress(f"Cmd failed. Logs: {summary}")
        raise

    finally:
        stdout_f.close()
        stderr_f.close()
        stdout_f_read.close()
        if cleanup_logs:
            os.remove(progress)
            os.remove(summary)
            os.remove(stdout)
            os.remove(stderr)
