import sys
from .main import main

def run() -> int:
    return main('lib2to3.fixes')

if __name__ == "__main__":
    sys.exit(run())
