from typing import Optional, Dict
from .. import fixer_base
from ..fixer_util import ImportAndCall, touch_import
from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Node

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

    def transform(self, node: Node, results: Dict[str, Node]) -> Optional[Node]:
        if results:
            obj: Node = results['obj']
            if obj:
                if ((obj.type == self.syms.argument) and (obj.children[0].value in {'**', '*'})):
                    return
        names: tuple[str, str] = ('importlib', 'reload')
        new: ImportAndCall = ImportAndCall(node, results, names)
        touch_import(None, 'importlib', node)
        return new
