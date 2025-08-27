import sys
from typing import NoReturn
from .main import main

def run() -> NoReturn:
    sys.exit(main('lib2to3.fixes'))

if __name__ == '__main__':
    run()
