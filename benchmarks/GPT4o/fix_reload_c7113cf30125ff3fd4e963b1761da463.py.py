'Fixer for reload().\n\nreload(s) -> importlib.reload(s)'
from .. import fixer_base
from ..fixer_util import ImportAndCall, touch_import
from typing import Optional, Dict, Any

class FixReload(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    PATTERN: str = "\n    power< 'reload'\n           trailer< lpar='('\n                    ( not(arglist | argument<any '=' any>) obj=any\n                      | obj=arglist<(not argument<any '=' any>) any ','> )\n                    rpar=')' >\n           after=any*\n    >\n    "

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        if results:
            obj = results['obj']
            if obj:
                if ((obj.type == self.syms.argument) and (obj.children[0].value in {'**', '*'})):
                    return None
        names = ('importlib', 'reload')
        new = ImportAndCall(node, results, names)
        touch_import(None, 'importlib', node)
        return new
