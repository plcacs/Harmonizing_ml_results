from typing import Any, Dict, Optional, Tuple
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import ImportAndCall, touch_import

class FixReload(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    PATTERN: str = "\n    power< 'reload'\n           trailer< lpar='('\n                    ( not(arglist | argument<any '=' any>) obj=any\n                      | obj=arglist<(not argument<any '=' any>) any ','> )\n                    rpar=')' >\n           after=any*\n    >\n    "

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        if results:
            obj = results['obj']
            if obj:
                if (
                    obj.type == self.syms.argument
                    and obj.children[0].value in {'**', '*'}
                ):
                    return
        names: Tuple[str, str] = ('importlib', 'reload')
        new: ImportAndCall = ImportAndCall(node, results, names)
        touch_import(None, 'importlib', node)
        return new
