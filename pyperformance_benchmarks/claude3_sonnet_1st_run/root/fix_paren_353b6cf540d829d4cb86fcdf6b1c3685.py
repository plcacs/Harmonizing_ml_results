'Fixer that adds parentheses where they are required\n\nThis converts ``[x for x in 1, 2]`` to ``[x for x in (1, 2)]``.'
from typing import Dict, Any
from .. import fixer_base
from ..fixer_util import LParen, RParen

class FixParen(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n        atom< ('[' | '(')\n            (listmaker< any\n                comp_for<\n                    'for' NAME 'in'\n                    target=testlist_safe< any (',' any)+ [',']\n                     >\n                    [any]\n                >\n            >\n            |\n            testlist_gexp< any\n                comp_for<\n                    'for' NAME 'in'\n                    target=testlist_safe< any (',' any)+ [',']\n                     >\n                    [any]\n                >\n            >)\n        (']' | ')') >\n    "

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        target = results['target']
        lparen = LParen()
        lparen.prefix = target.prefix
        target.prefix = ''
        target.insert_child(0, lparen)
        target.append_child(RParen())
