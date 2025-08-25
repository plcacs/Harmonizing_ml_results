from ..fixer_util import LParen, RParen
from typing import Any
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node

class FixParen(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n        atom< ('[' | '(')\n            (listmaker< any\n                comp_for<\n                    'for' NAME 'in'\n                    target=testlist_safe< any (',' any)+ [',']\n                     >\n                    [any]\n                >\n            >\n            |\n            testlist_gexp< any\n                comp_for<\n                    'for' NAME 'in'\n                    target=testlist_safe< any (',' any)+ [',']\n                     >\n                    [any]\n                >\n            >)\n        (']' | ')') >\n    "

    def transform(self, node: Node, results: dict[str, Any]) -> None:
        target: Node = results['target']
        lparen: LParen = LParen()
        lparen.prefix = target.prefix
        target.prefix = ''
        target.insert_child(0, lparen)
        target.append_child(RParen())
