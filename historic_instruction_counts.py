import argparse
import collections
import datetime
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time


# Modes.
MAIN = "main"
MEASURE = "measure"
MEASURE_SUBTASK = "measure_subtask"
DEBUG = "debug"

ROOT = "/tmp/historic_instruction_counts"
PYTORCH_ROOT = os.path.join(ROOT, "pytorch")
HISTORY_PYTORCH_ROOT = os.path.join(ROOT, "pytorch_for_history")
TIMER_PYTORCH_ROOT = os.path.join(ROOT, "pytorch_for_timer")
CANNOT_BUILD_PATH = os.path.join(ROOT, "cannot_build.txt")
ENV_ROOT = os.path.join(ROOT, "envs")  # TODO: wire this up.
RECORDS_PATH = os.path.join(ROOT, "records")
TIMER_REF_ENV = "historic_sweep_ref_env"
# SWEEP_START = "06/01/2020"
SWEEP_START = "01/01/2020"
SWEEP_CADENCE = 7  # days


ENV = {
    "PATH": os.getenv("PATH"),
    "HOME": os.getenv("HOME"),
    "no_proxy": os.getenv("no_proxy"),
    "http_proxy": os.getenv("http_proxy"),
    "https_proxy": os.getenv("https_proxy"),
    "BUILD_CAFFE2_OPS": "0",
    "USE_DISTRIBUTED": "0",
    "BUILD_TEST": "0",
    "USE_CUDA": "0",
    "REL_WITH_DEB_INFO": "1",
    "MKL_THREADING_LAYER": "GNU",
}

# Suggestions
"""
Dashboard
C++ JIT


Meta: run code coverage to systematically find hot spots.
x.shape()
x.stride()
x.view()

Test Python vs. C++ analog
Test TorchScript programs
"""


LOOP_NUMBER = 10000
REPEATS = 5
TIMER_SEC = 20
PASS = "pass"
SCALAR_X = "x = torch.ones((1,))"
SCALAR_XY = "x = torch.ones((1,));y = torch.ones((1,))"
SMALL_X = "x = torch.ones((10,))"
TASKS = (
    # Allocation. (With and without storage)
    ("torch.empty(())", PASS),
    ("torch.empty((1,))", PASS),

    # Data movement and assignment
    ("x.zero_()", SCALAR_X),
    ("x.copy_(x)", SCALAR_X),
    ("x.copy_(y)", SCALAR_XY),
    ("x.contiguous()", SCALAR_X),
    ("x.clone()", SCALAR_X),
    ("x[1]", SMALL_X),
    ("x += 1", SMALL_X),
    ("x.sum()", SMALL_X),

    # Metadata.
    ("x.size()[0]", SCALAR_X),
    ("x.stride(0)", SCALAR_X),
    ("x.view(-1, 2)", SMALL_X),

    # TorchScript (measurement code will create fn_jit.pth)
    ("model(x)", f"{SCALAR_X};model = torch.jit.load('/tmp/fn_jit.pth')"),

    # Autograd
    ("x.backward()", "x = torch.ones((1,)) + torch.ones((1,), requires_grad=True)"),
    ("y = x * w;y.backward()", f"{SCALAR_X}; w = torch.ones((1,), requires_grad=True)"),
    (
        "y = torch.nn.functional.relu(x * w0) * w1;y.backward()",
        f"{SCALAR_X}; w0 = torch.ones((1,), requires_grad=True); w1 = torch.ones((1,), requires_grad=True)"
    ),

)


def build(env_name, repo_path, checkout="fbcode/warm", must_succeed=True, history=None):
    start_time = time.time()
    tag_prefix = "STEP_COMPLETE: "
    clean_complete = "CLEAN_COMPLETE"
    ready_to_install = "READY_TO_INSTALL"

    env = ENV.copy()
    if history is None:
        assert checkout == "fbcode/warm"
    else:
        # For a period in early January, building tests was required.
        # Surfaced by a build refactor in:
        #   https://github.com/pytorch/pytorch/pull/31162
        # And fixed by:
        #   https://github.com/pytorch/pytorch/pull/31965
        for sha, _, _ in history:
            if sha == "ddff4efa26d527c99cd9892278a32529ddc77e66":
                env["BUILD_TEST"] = "1"
            if sha == checkout:
                break
            if sha == "61e509b9922f632a9bb89ed06406df93f8bd2da8":
                env["BUILD_TEST"] = ENV["BUILD_TEST"]
                break
        pass

    # This build is CPU only, so no cudatoolkit or cudnn.
    cmd = f"""
        source ~/.bashrc
        retry () {{ $* || (sleep 1 && $*) || (sleep 2 && $*); }}

        conda env remove --name {env_name} 2> /dev/null || true
        conda create --no-default-packages -yn {env_name} python=3
        echo "{tag_prefix}Conda env creation: {env_name}"

        source activate {env_name}
        conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi hypothesis typing_extensions
        echo "{tag_prefix}Dependency installation."

        cd {repo_path}
        retry git checkout .
        git clean -fd
        retry git checkout {checkout}
        git clean -fd
        python setup.py clean
        echo "{clean_complete}"
        retry git submodule sync
        retry git submodule update --init --recursive
        echo "{tag_prefix}Submodule sync."

        which cc  | awk '{{print "{tag_prefix} which cc:  "$1}}'
        which c++ | awk '{{print "{tag_prefix} which c++: "$1}}'
        echo "{ready_to_install}"
        python setup.py install
        cd {ROOT}
        python -c "import torch;print(torch.__file__)"
        echo "{tag_prefix}Install."
    """
    cmd = " && ".join([
        l.strip() for l in textwrap.dedent(cmd).splitlines(keepends=False)
        if l and not l.startswith("#")
    ])

    pattern_0 = re.compile(tag_prefix)
    pattern_1 = re.compile(r"^\[[0-9]+/[0-9]+\]\s.+$")
    at_line_start = True
    reached_clean = False
    reached_install = False
    with subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
      ) as proc:
        for l in iter(proc.stdout.readline, ""):
            l = l.decode("utf-8").strip()
            if pattern_0.search(l):
                if not at_line_start:
                    print()
                print(l)
                at_line_start = True
            if pattern_1.search(l):
                print(f"\r{l[:160]:<160}", end="")
                at_line_start = False
            if clean_complete in l:
                reached_clean = True
            if ready_to_install in l:
                reached_install = True

            retcode = proc.poll()
            if retcode is not None:
                # print("\n".join([err_l.decode("utf-8").strip() for err_l in proc.stderr.readlines()]))
                break

    # if not reached_install:
    #     subprocess.run(
    #         cmd,
    #         shell=True,
    #         env=ENV,
    #     )

    # If you encounter issues with git authentication, the following command can help:
    #   $ git config --global url.ssh://git@github.com/.insteadOf https://github.com/
    # Although historic submodules are brittle in general.
    assert reached_clean, f"Setup failed: {retcode}"
    if must_succeed:
        assert not retcode, f"Install failed: {retcode}"

    if retcode:
        print(f"Build failed: {time.time() - start_time:.0f} sec.")
    else:
        print(f"Build time: {time.time() - start_time:.0f} sec.")

    return retcode


def get_env_root(env_name):
    path = subprocess.run(
        f"source activate {env_name} && python -c 'import torch;print(torch.__file__)'",
        shell=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
        env=ENV,
    ).stdout.decode("utf-8").strip()
    assert env_name in path
    assert path.endswith("torch/__init__.py")
    return path


def build_ref_env():
    # build(TIMER_REF_ENV, TIMER_PYTORCH_ROOT, checkout="gh/taylorrobie/callgrind_timer")

    # test_log = subprocess.run(
    #     f"source activate {TIMER_REF_ENV} && "
    #     "python -c 'from torch.utils._benchmark import Timer;"
    #     "print(Timer().collect_callgrind())'",
    #     shell=True,
    #     capture_output=True,
    #     check=True,
    #     cwd=ROOT,
    # ).stdout.decode("utf-8")
    # assert "CallgrindStats" in test_log

    path = get_env_root(TIMER_REF_ENV)
    return os.path.join(os.path.split(path)[0], "utils", "_benchmark")


def monkey_patch_timer(timer_path, env_name):
    assert os.path.exists(timer_path)

    env_root = os.path.split(get_env_root(env_name))[0]
    target_path = os.path.join(env_root, "utils", "_benchmark")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    shutil.copytree(timer_path, target_path)


def get_git_history():
    mmddyyyy_fmt = "%m/%d/%Y"
    yyyymmdd_fmt = "%Y-%m-%d"
    t0_datetime = datetime.datetime.strptime(SWEEP_START, mmddyyyy_fmt)
    t0_minus_1 = (t0_datetime - datetime.timedelta(days=1)).strftime(yyyymmdd_fmt)
    log = subprocess.run(
        f"cd {HISTORY_PYTORCH_ROOT} && git checkout fbcode/warm && git log --pretty='format:%H %ai %s' --since {t0_minus_1} | cat",
        shell=True,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")

    output = []
    pattern = re.compile(r"^([a-z0-9]{40}) ([0-9\-]{10}) [0-9:]+ [\-0-9]+ (.+)$")
    for l in log.splitlines(keepends=False)[::-1]:
        match = pattern.match(l)
        if match:
            sha, date, msg = match.groups()
            # We're only interested in date for the initial sweep, so we
            # don't need to worry about HH:MM:SS.
            date = datetime.datetime.strptime(date, yyyymmdd_fmt)
            if (date - t0_datetime).total_seconds() < 0:
                continue
            output.append((sha, date, msg))
    return output


def setup_sweep():
    history = get_git_history()
    t0 = history[0][1]
    sweep_indices = [0]
    for i, (_, date, _) in enumerate(history):
        if (date - history[sweep_indices[-1]][1]).days >= SWEEP_CADENCE:
            sweep_indices.append(i)
    if sweep_indices[-1] != len(history) - 1:
        sweep_indices.append(len(history) - 1)
    return history, sweep_indices


def setup(clean):
    if clean:
        if os.path.exists(ROOT):
            shutil.rmtree(ROOT)
        os.makedirs(ROOT)
        os.makedirs(RECORDS_PATH)
        os.makedirs(ENV_ROOT)

        # This workflow is very sensitive to build caches, so jumping around git
        # history can slow things down. As a result, separate repos are created
        # which live at fbcode/warm and gh/taylorrobie/callgrind_timer respectively.
        for dest in [PYTORCH_ROOT, HISTORY_PYTORCH_ROOT, TIMER_PYTORCH_ROOT]:
            subprocess.run(
                f"git clone git@github.com:pytorch/pytorch.git {dest}",
                shell=True,
                # capture_output=True,
                check=True,
                env=ENV,
            )

        with open(CANNOT_BUILD_PATH, "wt") as f:
            pass

    return build_ref_env()


def run_sweep_batch(batch_indices, history, data=None, timer_path=None):
    with open(CANNOT_BUILD_PATH, "rt") as f:
        unbuildable = set(f.read().splitlines(keepends=False))

    def parse_history(sha, date):
        date_str = date.strftime("%Y_%m_%d")
        result_file = os.path.join(RECORDS_PATH, f"{date_str}_{sha}.counts.pkl")
        return date_str, result_file

    if data is None:
        data = [None for _ in history]

    # Results should have been sorted before saving, but it doesn't hurt
    # to enforce it since the artifact itself enforces no order.
    order = {(stmt, setup): i for i, (stmt, setup) in enumerate(TASKS)}

    for i, ((sha, date, _), d) in enumerate(zip(history, data)):
        if d is not None:
            continue

        date_str, result_file = parse_history(sha, date)
        if not os.path.exists(result_file):
            continue

        with open(result_file, "rb") as f:
            _, run_results = pickle.load(f)

        run_data = [[] for _ in TASKS]
        for stmt, setup, count, median in run_results:
            task_index = order[(stmt, setup)]
            run_data[task_index].append((count, median))
        data[i] = tuple(tuple(d) for d in run_data)

    env_name = "test_env"
    for i, batch_index in enumerate(batch_indices):
        if data[batch_index] is not None:
            continue

        sha, date, msg = history[batch_index]
        date_str, result_file = parse_history(sha, date)
        if sha in unbuildable:
            print(f"Skipping {sha}")
            continue

        print(f"{sha} {date_str} {msg}")
        if build(env_name, PYTORCH_ROOT, checkout=sha, must_succeed=False, history=history):
            with open(CANNOT_BUILD_PATH, "at") as f:
                f.write(f"{sha}\n")
            continue
        monkey_patch_timer(timer_path, env_name)

        measure_success = False
        for _ in range(5):
            try:
                subprocess.run(
                    f"source activate {env_name} && "
                    f"python {os.path.abspath(__file__)} "
                    f"--mode {MEASURE} --result_file {result_file.replace('.counts.pkl', '.pkl')}",
                    shell=True,
                    timeout=10 * 60,  # Ten minutes.
                    capture_output=False,
                    check=True,
                    cwd=ROOT,
                    env=ENV,
                )
                measure_success = True
                break
            except subprocess.TimeoutExpired:
                print("Measurement timed out.")
            except KeyboardInterrupt:
                raise
            except:
                break  # Exclude SHA if any other error occurs.

        if not measure_success:
            print("Failed to measure counts.")
            with open(CANNOT_BUILD_PATH, "at") as f:
                f.write(f"{sha}\n")

        print(f"{i + 1} / {len(batch_indices)}")

    if batch_indices:
        # Recursively call to update data
        run_sweep_batch([], history=history, data=data)

    return data


def next_sweep_batch(history, data, delta_threshold=0.001, verbose=False, autograd_only=False):
    import numpy as np
    with open(CANNOT_BUILD_PATH, "rt") as f:
        unbuildable = set(f.read().splitlines(keepends=False))

    data_indices = []
    intermediate_indices = []
    low_water_mark_count = [np.inf for _ in TASKS]
    low_water_mark_time = [np.inf for _ in TASKS]
    # low_water_mark = [np.inf for _ in TASKS]
    assert data[0] is not None
    for i, ((sha, _, _), d) in enumerate(zip(history, data)):
        if sha in unbuildable:
            continue

        if d:
            data_indices.append(i)
            intermediate_indices.append([])
        else:
            intermediate_indices[-1].append(i)

    output = []
    mismatch_above_threshold = []
    for lower, upper, intermediate in zip(data_indices[:-1], data_indices[1:], intermediate_indices):
        any_mismatch = False
        mismatch_above_threshold.append((lower, upper, []))
        most_recent_counts, most_recent_times = [], []
        for i, (l_d, u_d, (stmt, _)) in enumerate(zip(data[lower], data[upper], TASKS)):
            l, u = [li[0] for li in l_d], [ui[0] for ui in u_d]
            if autograd_only and "backward" not in stmt:
                continue
            l_25, l_75 = np.percentile(l, 25), np.percentile(l, 75)
            u_25, u_75 = np.percentile(u, 25), np.percentile(u, 75)
            rel_diff = abs(np.median(l) - np.median(u)) /  (np.median(l) + np.median(u)) * 2
            mismatch = ((l_25 > u_75) or (u_25 > l_75)) and rel_diff > delta_threshold
            mismatch_above_threshold[-1][2].append(mismatch)
            any_mismatch = any_mismatch or mismatch
            low_water_mark_count[i] = min(low_water_mark_count[i], np.median(l), np.median(u))

            l_t, u_t = np.median([li[1] for li in l_d]), np.median([ui[1] for ui in u_d])
            low_water_mark_time[i] = min(low_water_mark_time[i], l_t, u_t)

            most_recent_counts.append(np.median(u))
            most_recent_times.append(u_t)

        if any_mismatch and intermediate:
            output.append(intermediate[int(len(intermediate) // 2)])

    if verbose:
        sections = [("Creation", 2), ("Pointwise", 8), ("Metadata", 3), ("JIT", 1), ("AutoGrad", 3)]
        section_bounds = np.cumsum([n for _, n in sections])
        section_bounds = [
            (" :" if i + 1 in section_bounds else "  ")
            for i in range(len(TASKS))
        ][:-1] + [" "]

        section_stmts = [stmt for stmt, _ in TASKS][::-1]
        for section, n in sections:
            print(f"{section}\n{'-' * len(section)}")
            for _ in range(n):
                print(f"  {section_stmts.pop()}")
            print()
        print("\n")
        print(" " * 6 + "|".join([section.center(n * 8 - 1) for section, n in sections]))

        current_printed = False
        for lower, upper, mismatch_by_task in mismatch_above_threshold[::-1]:
            if not any(mismatch_by_task):
                continue

            delta_count, delta_t = [], []
            for i, (mismatch, lower_data, upper_data) in enumerate(zip(mismatch_by_task, data[lower], data[upper])):
                if not mismatch:
                    delta_count.append(None)
                    delta_t.append(None)
                    continue

                lower_count = np.median([d[0] for d in lower_data])
                upper_count = np.median([d[0] for d in upper_data])
                delta_count.append((upper_count - lower_count) / low_water_mark_count[i])

                lower_time = np.median([d[1] for d in lower_data])
                upper_time = np.median([d[1] for d in upper_data])
                delta_t.append((upper_time - lower_time) / low_water_mark_time[i])

            def to_row(values):
                width = 6
                good, bad, terminate = "\033[92m", "\033[31m", "\033[0m\033[0m"
                return "".join([
                    (" " * width if v is None else
                    (good if v < 0 else bad) + "\033[1m" + f"{'+' if v > 0 else ''}{v * 100:.1f}".rjust(width) + terminate) + b
                    for v, b in zip(values, section_bounds)]
                )

            if not current_printed:
                print("-" * 220 + "\n" + '-' * 220)
                print(
                    " " * 6 + to_row([c / lwm - 1 for c, lwm in zip(most_recent_counts, low_water_mark_count)]) +
                    f" |  (\u0394 instruction counts){'':>8}Most recent vs. low water mark"
                )
                print(
                    " " * 6 + to_row([t / lwm - 1 for t, lwm in zip(most_recent_times, low_water_mark_time)]) +
                    " |  (\u0394 wall time)"
                )
                print("-" * 220 + "\n" + '-' * 220)
                current_printed = True

            sha, date, msg = history[upper]
            range_str = f"({lower} -> {upper})".ljust(14)
            print("\u0394C  | " + to_row(delta_count) + f" |  {'...' if upper - lower > 1 else msg[:73]}")
            print("\u0394t  | " + to_row(delta_t) + f" |  {range_str}   {date.strftime('%Y-%m-%d')}    {sha}")
            print(" " * 4 + "=" * 216)

    return output


def sweep(timer_path):
    history, initial_sweep_indices = setup_sweep()

    # First do a ~monthly cadence to fill in quickly.
    data = run_sweep_batch(initial_sweep_indices[::4], history, timer_path=timer_path)
    data = run_sweep_batch(initial_sweep_indices, history, data=data, timer_path=timer_path)

    # Progressively ramp down threshold to more quickly zero in on major regressions.
    for delta_threshold in [0.15, 0.05, 0.03, 0.02, 0.01]:
        while True:
            batch_indices = next_sweep_batch(
                history=history,
                data=data,
                delta_threshold=delta_threshold,
                verbose=False,
            )
            if not batch_indices:
                print(f"Complete: delta_threshold={delta_threshold}")
                break
            data = run_sweep_batch(
                batch_indices=batch_indices,
                history=history,
                data=data,
                timer_path=timer_path,
            )


def measure(result_file):
    assert result_file is None or result_file.endswith(".pkl")
    start_time = time.time()
    print("Measurements:")
    print("  Starting measurement.")

    import multiprocessing
    import multiprocessing.dummy
    import numpy as np
    import torch
    from torch.utils._benchmark import Timer

    # TorchScript needs fn to be defined in a real file, not as a string.
    # To get around this we load a serialized model in the Timer setup.
    # It's a hack, but such is life.
    def fn(y: torch.Tensor):
        return y + 1

    fn_jit = torch.jit.script(fn)
    torch.jit.save(fn_jit, "/tmp/fn_jit.pth")

    _ = Timer().collect_callgrind(LOOP_NUMBER)
    def map_fn(stmt_setup):
        stmt, setup = stmt_setup
        task_index = {(stmt, setup): i for i, (stmt, setup) in enumerate(TASKS)}[(stmt, setup)]

        # Stagger startup.
        time.sleep(np.random.rand())

        counts = Timer(stmt, setup=setup).collect_callgrind(LOOP_NUMBER)

        # Unfortunately, if we directly collect times all of the threads will contend
        # for the GIL and the results will be meaningless. So yet more subprocessing
        # is required for proper isolation.
        _, m_file = tempfile.mkstemp()
        try:
            pass
            subprocess.run(
                f"python {os.path.abspath(__file__)} "
                f"--mode {MEASURE_SUBTASK} --result_file {m_file} --task_index {task_index}",
                shell=True,
                timeout=TIMER_SEC + 20,
                capture_output=True,
                check=True,
                cwd=ROOT,
                env=ENV,
            )
            with open(m_file, "rb") as f:
                times = pickle.load(f)
        finally:
            os.remove(m_file)

        return (counts, times), task_index

    tasks = list(TASKS * (REPEATS + 1))
    for stmt, setup in TASKS:
        # For some reason, autograd likes to hang so we massively over-hedge
        if "backward" in stmt:
            tasks.extend([(stmt, setup) for _ in range(4)])

    results, num_complete = [], [0 for _ in TASKS]
    with multiprocessing.dummy.Pool(int(multiprocessing.cpu_count() * 3 / 4)) as pool:
        for result, task_index in pool.imap_unordered(map_fn, tasks, 1):
            num_complete[task_index] += 1
            if all(i >= REPEATS for i in num_complete):
                # We hedge by scheduling an extra instance of each task to
                # reduce straggler effects.
                break
            results.append(result)
            remaining = [max(REPEATS - i, 0) for i in num_complete]
            print(
                f"\r{len(results)} / {len(tasks)}   Remaining: {remaining}  "
                f"{time.time() - start_time:.0f} sec.".ljust(60), end="")
        print()

    order = {(stmt, setup): i for i, (stmt, setup) in enumerate(TASKS)}
    results.sort(key=lambda x: order[(x[0].stmt, x[0].setup)])
    print(f"  All measurements complete: {time.time() - start_time:.0f} sec.")

    if result_file is None:
        # Called for debugging.
        import pdb
        pdb.set_trace()
    else:
        print(f"  Writing to file: {result_file}")
        with open(result_file, "wb") as f:
            pickle.dump([torch.__version__, results], f)

        counts_file = result_file[:-4] + ".counts.pkl"
        print(f"  Writing to file: {counts_file}")
        with open(counts_file, "wb") as f:
            pickle.dump([
                torch.__version__,
                [(r.stmt, r.setup, r.counts(include_lookdict_unicode=False), t.median) for r, t in results]
            ], f)
        print("  Done writing results.")


def measure_subtask(result_file, task_index):
    from torch.utils._benchmark import Timer
    stmt, setup = TASKS[task_index]
    m = Timer(stmt, setup=setup).blocked_autorange(min_run_time=TIMER_SEC)
    with open(result_file, "wb") as f:
        pickle.dump(m, f)


def debug():
    history, _ = setup_sweep()
    data = run_sweep_batch([], history)
    next_sweep_batch(history, data, delta_threshold=0.02, verbose=True)
    return

    # first = "c6d0fdd21537a2c23f94e7a44ac552d24b447906"
    # second = "e7a09b4d17010e60bc95c25f8165ef479d1b9612"

    # for i in os.listdir(RECORDS_PATH):
    #     if i.endswith(".counts.pkl"):
    #         continue
    #     if first in i:
    #         with open(os.path.join(RECORDS_PATH, i), "rb") as f:
    #             _, r0 = pickle.load(f)
    #     if second in i:
    #         with open(os.path.join(RECORDS_PATH, i), "rb") as f:
    #             _, r1 = pickle.load(f)

    # stmt = TASKS[-3][0]
    # for counts, times in r0:
    #     if counts.stmt == stmt:
    #         print(counts, "\n", times, "\n")
    #         c0 = counts.as_standardized()
    #         break
    # print("=" * 80)
    # for counts, times in r1:
    #     if counts.stmt == stmt:
    #         print(counts, "\n", times, "\n")
    #         c1 = counts.as_standardized()
    #         break


def main(clean=False):
    timer_path = setup(clean)
    sweep(timer_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="main", choices=(MAIN, MEASURE, MEASURE_SUBTASK, DEBUG))
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--task_index", type=int, default=None)
    args = parser.parse_args()

    if args.mode == MAIN:
        main()
    elif args.mode == MEASURE:
        measure(args.result_file)
    elif args.mode == MEASURE_SUBTASK:
        measure_subtask(args.result_file, args.task_index)
    elif args.mode == DEBUG:
        debug()
