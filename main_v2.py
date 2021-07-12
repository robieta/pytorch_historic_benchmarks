import os
os.environ["BENCHMARK_USE_DEV_SHM"] = "1"
os.environ["USE_NOISE_POLICE"] = "1"

from v2 import runner


def main():
    runner.Runner().run()


if __name__ == "__main__":
    main()
