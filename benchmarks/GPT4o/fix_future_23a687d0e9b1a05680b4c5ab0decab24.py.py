from typing import Any

from .. import fixer_base
from ..fixer_util import BlankLine
from lib2to3.pytree import Node

class FixFuture(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = 'import_from< \'from\' module_name="__future__" \'import\' any >'
    run_order: int = 10

    def transform(self, node: Node, results: Any) -> Node:
        new = BlankLine()
        new.prefix = node.prefix
        return new
