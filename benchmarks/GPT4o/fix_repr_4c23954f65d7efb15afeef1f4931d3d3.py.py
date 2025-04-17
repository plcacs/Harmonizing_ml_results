from typing import Dict, Any
from lib2to3.pytree import Node
from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Call, Name, parenthesize

class FixRepr(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              atom < '`' expr=any '`' >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        expr: Node = results['expr'].clone()
        if expr.type == self.syms.testlist1:
            expr = parenthesize(expr)
        return Call(Name('repr'), [expr], prefix=node.prefix)
