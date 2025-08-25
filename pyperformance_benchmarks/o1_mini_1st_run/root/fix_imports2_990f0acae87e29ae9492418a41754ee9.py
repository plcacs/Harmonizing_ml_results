'Fix incompatible imports and module references that must be fixed after\nfix_imports.'
from . import fix_imports

MAPPING: dict[str, str] = {'whichdb': 'dbm', 'anydbm': 'dbm'}

class FixImports2(fix_imports.FixImports):
    run_order: int = 7
    mapping: dict[str, str] = MAPPING
