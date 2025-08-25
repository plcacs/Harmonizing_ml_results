from typing import Any
from lib2to3.pgen2 import token
from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import ImportAndCall, touch_import

class FixReload(BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    PATTERN: str = "\n    power< 'reload'\n           trailer< lpar='('\n                    ( not(arglist | argument<any '=' any>) obj=any\n                      | obj=arglist<(not argument<any '=' any>) any ','> )\n                    rpar=')' >\n           after=any*\n    >\n    "

    def transform(self, node: Any, results: Any) -> Any:
        if results:
            obj = results['obj']
            if obj:
                if ((obj.type == self.syms.argument) and (obj.children[0].value in {'**', '*'})):
                    return
        names: tuple = ('importlib', 'reload')
        new: Any = ImportAndCall(node, results, names)
        touch_import(None, 'importlib', node)
        return new
