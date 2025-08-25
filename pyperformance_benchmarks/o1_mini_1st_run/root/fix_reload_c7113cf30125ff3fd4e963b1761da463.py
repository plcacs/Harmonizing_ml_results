'Fixer for reload().\n\nreload(s) -> importlib.reload(s)'
from .. import fixer_base
from ..fixer_util import ImportAndCall, touch_import
from typing import Any, Dict, Optional, Tuple

class FixReload(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    PATTERN: str = """
    power< 'reload'
           trailer< lpar='('
                    ( not(arglist | argument<any '=' any>) obj=any
                      | obj=arglist<(not argument<any '=' any>) any ','> )
                    rpar=')' >
           after=any*
    >
    """

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        if results:
            obj: Any = results.get('obj')
            if obj:
                if ((obj.type == self.syms.argument) and (obj.children[0].value in {'**', '*'})):
                    return
        names: Tuple[str, str] = ('importlib', 'reload')
        new = ImportAndCall(node, results, names)
        touch_import(None, 'importlib', node)
        return new
