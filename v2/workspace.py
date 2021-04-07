import atexit
import dataclasses
import os

_VERSION = 2
ROOT = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
WORKSPACE_ROOT = os.path.join(ROOT, "workspace", f"sweeping_bisect_v{_VERSION}")

RUNNER_STATE_ROOT = os.path.join(WORKSPACE_ROOT, "runner_state")

BUILD_LOG_ROOT = os.path.join(WORKSPACE_ROOT, "build__logs")
BUILD_IN_PROGRESS_ROOT = os.path.join(WORKSPACE_ROOT, "build__in_progress")
BUILD_COMPLETED_ROOT = os.path.join(WORKSPACE_ROOT, "build__completed")
CANNOT_BUILD = os.path.join(WORKSPACE_ROOT, "cannot_build.txt")

RUN_IN_PROGRESS_ROOT = os.path.join(WORKSPACE_ROOT, "run__in_progress")
RUN_LOG_ROOT = os.path.join(WORKSPACE_ROOT, "run__logs")
RUN_COMPLETED_ROOT = os.path.join(WORKSPACE_ROOT, "run__completed")

# manifoldfs manifold.blobstore pytorch_historic_bisect_artifacts "$(pwd)/overflow"
OVERFLOW_ROOT = os.path.join(WORKSPACE_ROOT, "overflow")

# touch this file to stop the build/test loop.
STOP_FILE = os.path.join(WORKSPACE_ROOT, "stop")

# touch this file to stop the build/test loop.
PDB_FILE = os.path.join(WORKSPACE_ROOT, "pdb")

# touch this file to tell the runner to run the debug script.
CALL_DEBUG_FILE = os.path.join(WORKSPACE_ROOT, "debug")

# touch this file to stop allocations until it is removed.
THROTTLE_FILE = os.path.join(WORKSPACE_ROOT, "throttle")

STATUS_FILE = os.path.join(WORKSPACE_ROOT, "status.txt")

REF_REPO_ROOT = os.path.join(WORKSPACE_ROOT, "pytorch")
BENCHMARK_BRANCH_NAME = "gh/taylorrobie/timer_ci_prep"
BENCHMARK_BRANCH_ROOT = os.path.join(WORKSPACE_ROOT, "pytorch_benchmark_branch")
BENCHMARK_ENV = os.path.join(WORKSPACE_ROOT, "pytorch_benchmark_env")
BENCHMARK_ENV_BUILT = os.path.join(BENCHMARK_ENV, "BENCHMARK_ENV_BUILT")

DATE_FMT = "%Y-%m-%d"
SWEEP_START = "2018-06-01"
SWEEP_CADENCE = 1  # day

TIDY_LOCATIONS = (
    BUILD_LOG_ROOT,
    BUILD_IN_PROGRESS_ROOT,
    RUN_LOG_ROOT,
    RUN_IN_PROGRESS_ROOT,
)


class _MutationLock:
    def __init__(self) -> None:
        self._held = False
        self._lock = os.path.join(WORKSPACE_ROOT, "mutation_lock")

    def get(self) -> None:
        if not self._held and os.path.exists(self._lock):
            raise ValueError(f"Lock {self._lock} is already held.")

        with open(self._lock, "wb") as f:
            pass

        self._held = True
        atexit.register(self.release)

    def release(self):
        if self._held:
            assert os.path.exists(self._lock), self._lock
            os.remove(self._lock)
            self._held = False

MUTATION_LOCK = _MutationLock()


def make_dirs():
    for d in (
        WORKSPACE_ROOT,
        RUNNER_STATE_ROOT,
        BUILD_LOG_ROOT,
        BUILD_IN_PROGRESS_ROOT,
        BUILD_COMPLETED_ROOT,
        RUN_IN_PROGRESS_ROOT,
        RUN_LOG_ROOT,
        RUN_COMPLETED_ROOT,
    ):
        os.makedirs(d, exist_ok=True)
