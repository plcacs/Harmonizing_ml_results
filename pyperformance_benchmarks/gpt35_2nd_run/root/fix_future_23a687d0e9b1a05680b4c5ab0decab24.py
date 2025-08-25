from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import BlankLine
from lib2to3.pytree import Node

class FixFuture(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = 'import_from< \'from\' module_name="__future__" \'import\' any >'
    run_order: int = 10

    def transform(self, node: Node, results: dict) -> BlankLine:
        new: BlankLine = BlankLine()
        new.prefix = node.prefix
        return new
