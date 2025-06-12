import sys
from .main import main

def annotated_main(module_name: str) -> int:
    return main(module_name)

sys.exit(annotated_main('lib2to3.fixes'))
