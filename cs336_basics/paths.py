import inspect
import pathlib
# from loguru import logger

import cs336_basics
init_file = inspect.getfile(cs336_basics)
# logger.info(f"init_file: {init_file}")
print(f"init_file: {init_file}")

CS336_PATH = pathlib.Path(init_file).parent.parent
# logger.info(f"DSP_REPO_PATH: {DSP_REPO_PATH}")
assert CS336_PATH.exists()

DATA_PATH = CS336_PATH.joinpath("data")
# logger.info(f"DATA_PATH: {DATA_PATH}")
assert DATA_PATH.exists()

TEST_PATH = CS336_PATH.joinpath("tests")
# logger.info(f"TEST_PATH: {TEST_PATH}")
assert TEST_PATH.exists()

def main():
    pass

if __name__ == "__main__":
    main()

