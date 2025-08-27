from typing import Dict, ClassVar
from . import fix_imports
MAPPING: Dict[str, str] = {'whichdb': 'dbm', 'anydbm': 'dbm'}

class FixImports2(fix_imports.FixImports):
    run_order: ClassVar[int] = 7
    mapping: ClassVar[Dict[str, str]] = MAPPING
