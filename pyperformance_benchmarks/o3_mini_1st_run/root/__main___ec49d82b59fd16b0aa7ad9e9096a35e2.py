import sys
from .main import main


def run() -> None:
    result: int = main("lib2to3.fixes")
    sys.exit(result)


if __name__ == "__main__":
    run()