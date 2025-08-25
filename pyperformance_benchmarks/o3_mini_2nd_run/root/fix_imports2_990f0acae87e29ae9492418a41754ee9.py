from typing import Dict
from . import fix_imports

MAPPING: Dict[str, str] = {'whichdb': 'dbm', 'anydbm': 'dbm'}

class FixImports2(fix_imports.FixImports):
    run_order: int = 7
    mapping: Dict[str, str] = MAPPING
