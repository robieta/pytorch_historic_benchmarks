import multiprocessing


NUM_CORES: int = multiprocessing.cpu_count()

# Currently only support DevBig
assert NUM_CORES in (80,)

POOL_SLACK = 8
OPPORTUNISTIC_BUILD_CORE_COUNTS = (32, 16)
BASELINE_BUILD_CORE_COUNT = 8
