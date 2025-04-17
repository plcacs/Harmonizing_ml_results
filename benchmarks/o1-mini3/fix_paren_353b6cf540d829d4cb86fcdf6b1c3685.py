from .. import fixer_base
from ..fixer_util import LParen, RParen
from typing import Any, Dict
from lib2to3.pytree import Node

class FixParen(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
        atom< ('[' | '(')
            (listmaker< any
                comp_for<
                    'for' NAME 'in'
                    target=testlist_safe< any (',' any)+ [','] >
                    [any]
                >
            >
            |
            testlist_gexp< any
                comp_for<
                    'for' NAME 'in'
                    target=testlist_safe< any (',' any)+ [','] >
                    [any]
                >
            >)
        (']' | ')') >
    """

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        target: Node = results['target']
        lparen: Node = LParen()
        lparen.prefix = target.prefix
        target.prefix = ''
        target.insert_child(0, lparen)
        target.append_child(RParen())
