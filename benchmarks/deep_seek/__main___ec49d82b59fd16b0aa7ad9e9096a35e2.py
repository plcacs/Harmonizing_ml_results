import sys
from typing import NoReturn
from .main import main

def entry_point() -> NoReturn:
    sys.exit(main('lib2to3.fixes'))

if __name__ == '__main__':
    entry_point()
