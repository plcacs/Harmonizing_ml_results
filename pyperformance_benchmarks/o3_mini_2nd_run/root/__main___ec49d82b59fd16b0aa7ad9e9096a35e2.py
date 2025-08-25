import sys
from typing import NoReturn
from .main import main

def run() -> NoReturn:
    fixes: str = 'lib2to3.fixes'
    result: int = main(fixes)
    sys.exit(result)

if __name__ == "__main__":
    run()