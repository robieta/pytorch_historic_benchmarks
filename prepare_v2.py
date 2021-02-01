import os

from v2 import init_pytorch


def main():
    if os.getenv("CONDA_PREFIX") is not None:
        raise ValueError("This script should not be run in a conda env.")

    init_pytorch.run()


if __name__ == "__main__":
    main()
