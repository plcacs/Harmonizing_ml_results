"Fixer that turns 'long' into 'int' everywhere.\n"
from lib2to3 import fixer_base
from lib2to3.fixer_util import is_probably_builtin
from lib2to3.pytree import Node
from lib2to3.pgen2 import token
from typing import Dict, Any

class FixLong(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'long'"

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        if is_probably_builtin(node):
            node.value = 'int'
            node.changed()
