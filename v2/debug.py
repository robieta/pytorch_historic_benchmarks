from v2.runner import Runner


def make_report(self: Runner):
    import importlib
    import v2.report
    importlib.reload(v2.report)
    v2.report.make_report(self)



def archive(self: Runner):
    import os
    import time
    from v2.workspace import OVERFLOW_ROOT, THROTTLE_FILE

    retain_indices = set(self._initial_sweep_indices[::10])
    for i in range(len(self._history)):
        if i in retain_indices:
            continue

        sha = self._history[i].sha
        if sha in self._state._tested and not self._state._built[sha].startswith(OVERFLOW_ROOT):
            time.sleep(5)

        self._state.archive(sha)

        if os.path.exists(THROTTLE_FILE):
            break


def debug_fn(r: Runner):
    import sys
    import traceback

    try:
        make_report(r)
        archive(r)
    except:
        traceback.print_tb(sys.exc_info()[2])
        raise
