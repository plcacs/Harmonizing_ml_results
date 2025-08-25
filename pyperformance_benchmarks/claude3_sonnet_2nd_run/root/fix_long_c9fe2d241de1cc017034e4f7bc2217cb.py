"Fixer that turns 'long' into 'int' everywhere.\n"
from lib2to3 import fixer_base
from lib2to3.fixer_util import is_probably_builtin
from typing import Any, Dict, Optional

class FixLong(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'long'"

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        if is_probably_builtin(node):
            node.value = 'int'
            node.changed()
        return None
