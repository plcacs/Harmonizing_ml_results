from __future__ import annotations
from . import fix_imports
from typing import Dict, ClassVar

MAPPING: Dict[str, str] = {'whichdb': 'dbm', 'anydbm': 'dbm'}

class FixImports2(fix_imports.FixImports):
    run_order: ClassVar[int] = 7
    mapping: ClassVar[Dict[str, str]] = MAPPING