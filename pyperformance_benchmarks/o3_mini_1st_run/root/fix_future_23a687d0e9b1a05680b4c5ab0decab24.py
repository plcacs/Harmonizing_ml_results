'Remove __future__ imports\n\nfrom __future__ import foo is replaced with an empty line.\n'
from typing import Any, Dict
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import BlankLine

class FixFuture(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = 'import_from< \'from\' module_name="__future__" \'import\' any >'
    run_order: int = 10

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        new: BlankLine = BlankLine()
        new.prefix = node.prefix
        return new