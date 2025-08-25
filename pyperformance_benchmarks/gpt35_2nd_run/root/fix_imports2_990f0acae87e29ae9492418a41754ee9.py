from . import fix_imports
from typing import Dict

MAPPING: Dict[str, str] = {'whichdb': 'dbm', 'anydbm': 'dbm'}

class FixImports2(fix_imports.FixImports):
    run_order: int = 7
    mapping: Dict[str, str] = MAPPING
