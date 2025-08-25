from typing import Any
from lib2to3.pgen2 import token
from lib2to3.fixer_base import BaseFix
from lib2to3.fixes import fixer_util
from lib2to3.pytree import Node

class FixReload(BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    PATTERN: str = "\n    power< 'reload'\n           trailer< lpar='('\n                    ( not(arglist | argument<any '=' any>) obj=any\n                      | obj=arglist<(not argument<any '=' any>) any ','> )\n                    rpar=')' >\n           after=any*\n    >\n    "

    def transform(self, node: Node, results: dict[str, Any]) -> Any:
        if results:
            obj: Node = results['obj']
            if obj:
                if ((obj.type == self.syms.argument) and (obj.children[0].value in {'**', '*'})):
                    return
        names: tuple[str, str] = ('importlib', 'reload')
        new: Any = fixer_util.ImportAndCall(node, results, names)
        fixer_util.touch_import(None, 'importlib', node)
        return new
