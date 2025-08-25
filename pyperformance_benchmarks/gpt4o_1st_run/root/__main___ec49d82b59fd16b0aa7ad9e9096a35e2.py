import sys
from .main import main
from typing import NoReturn

def run_main() -> NoReturn:
    sys.exit(main('lib2to3.fixes'))

run_main()
