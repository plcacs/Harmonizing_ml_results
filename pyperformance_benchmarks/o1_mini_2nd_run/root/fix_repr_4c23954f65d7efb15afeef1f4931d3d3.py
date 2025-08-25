from typing import Any, Dict
from .. import fixer_base
from ..fixer_util import Call, Name, parenthesize
from lib2to3.pytree import Node

class FixRepr(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              atom < '`' expr=any '`' >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Call:
        expr = results['expr'].clone()
        if expr.type == self.syms.testlist1:
            expr = parenthesize(expr)
        return Call(Name('repr'), [expr], prefix=node.prefix)
