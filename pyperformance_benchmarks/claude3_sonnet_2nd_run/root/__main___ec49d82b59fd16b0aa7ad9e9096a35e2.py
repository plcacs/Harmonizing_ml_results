import sys
from typing import NoReturn
from .main import main

def __annotated_code() -> NoReturn:
    sys.exit(main('lib2to3.fixes'))

__annotated_code()
