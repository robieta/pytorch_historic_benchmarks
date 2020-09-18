import argparse
import collections
import datetime
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
import time


# Modes.
MAIN = "main"
MEASURE = "measure"
DEBUG = "debug"

ROOT = "/tmp/historic_instruction_counts"
PYTORCH_ROOT = os.path.join(ROOT, "pytorch")
HISTORY_PYTORCH_ROOT = os.path.join(ROOT, "pytorch_for_history")
TIMER_PYTORCH_ROOT = os.path.join(ROOT, "pytorch_for_timer")
CANNOT_BUILD_PATH = os.path.join(ROOT, "cannot_build.txt")
RECORDS_PATH = os.path.join(ROOT, "records")
TIMER_REF_ENV = "historic_sweep_ref_env"
SWEEP_START = "06/01/2020"
SWEEP_CADENCE = 7  # days


ENV = {
    "PATH": os.getenv("PATH"),
    "BUILD_CAFFE2_OPS": "0",
    "USE_DISTRIBUTED": "0",
    "BUILD_TEST": "0",
    "USE_CUDA": "0",
    "REL_WITH_DEB_INFO": "1",
    "MKL_THREADING_LAYER": "GNU",
}


LOOP_NUMBER = 10000
REPEATS = 9
PASS = "pass"
SCALAR_X = "x = torch.ones((1,))"
SCALAR_XY = "x = torch.ones((1,));y = torch.ones((1,))"
SMALL_X = "x = torch.ones((10,))"
TASKS = (
    ("torch.empty(())", PASS),
    ("torch.empty((1,))", PASS),
    ("x.zero_()", SCALAR_X),
    ("x.copy_(x)", SCALAR_X),
    ("x.copy_(y)", SCALAR_XY),
    ("x.contiguous()", SCALAR_X),
    ("x.clone()", SCALAR_X),
    ("x[1]", SMALL_X),
    ("x += 1", SMALL_X),
    ("x.backward()", "x = torch.ones((1,)) + torch.ones((1,), requires_grad=True)"),
    ("x.backward()", "x = torch.ones((1,), requires_grad=True) + torch.ones((1,))"),
    ("x.backward()", "x = torch.ones((1,), requires_grad=True) + torch.ones((1,), requires_grad=True)"),
)


def build(env_name, repo_path, checkout="fbcode/warm", must_succeed=True):
    start_time = time.time()
    tag_prefix = "STEP_COMPLETE: "
    ready_to_install = "READY_TO_INSTALL"

    # This build is CPU only, so no cudatoolkit or cudnn.
    cmd = f"""
        conda env remove --name {env_name} 2> /dev/null || true
        conda create --no-default-packages -yn {env_name} python=3
        echo "{tag_prefix}Conda env creation: {env_name}"

        source activate {env_name}
        conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi hypothesis typing_extensions
        echo "{tag_prefix}Dependency installation."

        cd {repo_path}
        git checkout .
        git checkout {checkout}
        git submodule sync
        git submodule update --init --recursive
        echo "{tag_prefix}Submodule sync."

        which cc  | awk '{{print "{tag_prefix} which cc:  "$1}}'
        which c++ | awk '{{print "{tag_prefix} which c++: "$1}}'
        python setup.py clean
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
    reached_install = False
    with subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=ENV,
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
            if ready_to_install in l:
                reached_install = True
            retcode = proc.poll()
            if retcode is not None:
                break
    assert reached_install, f"Setup failed: {retcode}"
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
    # if clean:
    #     if os.path.exists(ROOT):
    #         shutil.rmtree(ROOT)
    #     os.makedirs(ROOT)
    #     os.makedirs(RECORDS_PATH)
    #     subprocess.run(
    #         f"git clone git@github.com:pytorch/pytorch.git {PYTORCH_ROOT}",
    #         shell=True,
    #         capture_output=True,
    #         check=True,
    #         env=ENV,
    #     )
    #     # This workflow is very sensitive to build caches, so jumping around git
    #     # history can slow things down. As a result, separate repos are created
    #     # which live at fbcode/warm and gh/taylorrobie/callgrind_timer respectively.
    #     shutil.copytree(PYTORCH_ROOT, HISTORY_PYTORCH_ROOT)
    #     shutil.copytree(PYTORCH_ROOT, TIMER_PYTORCH_ROOT)
    #     with open(CANNOT_BUILD_PATH, "wt") as f:
    #         pass

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
        for stmt, setup, count in run_results:
            task_index = order[(stmt, setup)]
            run_data[task_index].append(count)
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
        if build(env_name, PYTORCH_ROOT, checkout=sha, must_succeed=False):
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
                    timeout=6 * 60,  # Six minutes.
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
    mismatch_count = []
    for lower, upper, intermediate in zip(data_indices[:-1], data_indices[1:], intermediate_indices):
        any_mismatch = False
        mismatch_count.append(0)
        for l, u, (stmt, _) in zip(data[lower], data[upper], TASKS):
            if autograd_only and "backward" not in stmt:
                continue
            l_25, l_75 = np.percentile(l, 25), np.percentile(l, 75)
            u_25, u_75 = np.percentile(u, 25), np.percentile(u, 75)
            rel_diff = abs(np.median(l) - np.median(u)) /  (np.median(l) + np.median(u)) * 2
            mismatch = ((l_25 > u_75) or (u_25 > l_75)) and rel_diff > delta_threshold
            mismatch_count[-1] += int(mismatch)
            any_mismatch = any_mismatch or mismatch

        if any_mismatch and intermediate:
            output.append(intermediate[int(len(intermediate) // 2)])

    if verbose:
        indices = []
        for i, c in enumerate(mismatch_count):
            if c:
                indices.extend([i, i + 1])
        indices = sorted(set(indices))

        boundary_indices = [data_indices[i] for i in indices]
        boundary_mismatch = [([0] + mismatch_count)[i] for i in indices]

        log_data = [data[i] for i in boundary_indices]
        log_data = list(zip(*log_data))
        col_labels = " ".join([f"[{j}]".rjust(6) for j in range(len(boundary_indices))])
        spacer_len = len(boundary_indices) * 7 + 20
        print(f"{'':>20} {col_labels}")

        for (stmt, setup), stmt_data in zip(TASKS, log_data):
            medians = [np.median(d) for d in stmt_data]
            min_median = min(medians)
            regression_str = " ".join([f"{(m / min_median - 1) * 100:>6.1f}" for m in medians])
            print(f"{'-' * spacer_len}\n{stmt:<20} {regression_str}")
        print("=" * spacer_len)

        mismatch_str = f"num \u0394{'':>16}" + " ".join([f"{i if i else '':>6}" for i in boundary_mismatch])
        print(mismatch_str + "\n")
        for i, ind in enumerate(boundary_indices):
            sha, date, msg = history[ind]
            print(f"[{i}]".ljust(6) + f"({ind:>4})    {sha}    {date}    {msg}")
        print(f"\nConverged: {len(boundary_indices) - 1 - len(output)} / {len(boundary_indices) - 1}")
        print("Next batch: " + ", ".join([str(i) for i in output]))


        # print()
        # for i in [i for i, c in enumerate(mismatch_count) if c]:
        #     i_l, i_u = data_indices[i], data_indices[i + 1]
        #     sha, date, _ = history[i_l]
        #     l_fpath = os.path.join(RECORDS_PATH, f"{date.strftime('%Y_%m_%d')}_{sha}.pkl")
        #     sha, date, _ = history[i_u]
        #     u_fpath = os.path.join(RECORDS_PATH, f"{date.strftime('%Y_%m_%d')}_{sha}.pkl")

        #     with open(l_fpath, "rb") as f:
        #         _, prior_results = pickle.load(f)

        #     with open(u_fpath, "rb") as f:
        #         _, new_results = pickle.load(f)

        #     index = 9 * REPEATS
        #     print(prior_results[index].counts(), new_results[index].counts())
        #     deltas = new_results[index].as_standardized().delta(prior_results[index].as_standardized(), inclusive=True)
        #     for c, fn in deltas[:60]:
        #         if fn.startswith("Python/") or fn.startswith("Object"):
        #             continue
        #         print(f"{c:>12} {fn}")
        #     print("...")
        #     for c, fn in deltas[-10:]:
        #         print(f"{c:>12} {fn}")

    return output


def sweep(timer_path):
    history, initial_sweep_indices = setup_sweep()
    data = run_sweep_batch(initial_sweep_indices, history, timer_path=timer_path)

    # Progressively ramp down threshold to more quickly zero in on major regressions.
    for delta_threshold in [0.15, (0.05, True), 0.05, 0.02, 0.01, 0.001, 0.0005, 0.0001]:
        autograd_only = False
        if isinstance(delta_threshold, tuple):
            delta_threshold, autograd_only = delta_threshold

        while True:
            batch_indices = next_sweep_batch(
                history=history,
                data=data,
                delta_threshold=delta_threshold,
                verbose=False,
                autograd_only=autograd_only,
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

    _ = Timer().collect_callgrind(LOOP_NUMBER)
    def map_fn(stmt_setup):
        stmt, setup = stmt_setup
        setup = setup
        return Timer(stmt, setup=setup).collect_callgrind(LOOP_NUMBER)

    tasks = list(TASKS * REPEATS)
    results = []
    with multiprocessing.dummy.Pool(multiprocessing.cpu_count()) as pool:
        for result in pool.imap_unordered(map_fn, tasks, 1):
            results.append(result)

    order = {(stmt, setup): i for i, (stmt, setup) in enumerate(TASKS)}
    results.sort(key=lambda x: order[(x.stmt, x.setup)])
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
                [(r.stmt, r.setup, r.counts(include_lookdict_unicode=False)) for r in results]
            ], f)
        print("  Done writing results.")


def debug():
    history, _ = setup_sweep()
    data = run_sweep_batch([], history)
    # next_sweep_batch(history[858:1629], data[858:1629], delta_threshold=0.05, verbose=True)
    # next_sweep_batch(history[1532:1803], data[1532:1803], delta_threshold=0.05, verbose=True)
    next_sweep_batch(history, data, delta_threshold=0.02, verbose=True)


def main(clean=False):
    timer_path = setup(clean)
    sweep(timer_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="main", choices=(MAIN, MEASURE, DEBUG))
    parser.add_argument("--result_file", type=str, default=None)
    args = parser.parse_args()

    if args.mode == MAIN:
        main()
    elif args.mode == MEASURE:
        measure(args.result_file)
    elif args.mode == DEBUG:
        debug()
