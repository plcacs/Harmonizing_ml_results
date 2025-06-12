from .. import fixer_base
from ..fixer_util import Call, Name, parenthesize
from typing import Any, Dict
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node
from lib2to3 import fixer_util

class FixRepr(fixer_base.BaseFix):
    BM_COMPATIBLE: bool = True
    PATTERN: str = "\n              atom < '`' expr=any '`' >\n              "

    def transform(self, node: Node, results: Dict[str, Node]) -> Call:
        expr: Node = results['expr'].clone()
        if expr.type == self.syms.testlist1:
            expr = parenthesize(expr)
        return Call(Name('repr'), [expr], prefix=node.prefix)
