import sys
from .main import main

def main_wrapper(module_name: str) -> int:
    return main(module_name)

sys.exit(main_wrapper('lib2to3.fixes'))
