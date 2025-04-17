from typing import Any, Dict
from .. import fixer_base
from ..fixer_util import BlankLine

class FixFuture(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = 'import_from< \'from\' module_name="__future__" \'import\' any >'
    run_order: int = 10

    def transform(self, node: Any, results: Dict[str, Any]) -> BlankLine:
        new: BlankLine = BlankLine()
        new.prefix = node.prefix
        return new
