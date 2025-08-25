"""Fix incompatible imports and module references that must be fixed after
fix_imports."""
from fix_imports import FixImports
from typing import Dict

MAPPING: Dict[str, str] = {'whichdb': 'dbm', 'anydbm': 'dbm'}

class FixImports2(FixImports):
    run_order: int = 7
    mapping: Dict[str, str] = MAPPING
