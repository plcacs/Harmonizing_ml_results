from typing import Optional, Dict, Any
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, Call, ArgList, Attr, is_tuple
from lib2to3.pytree import Node
from lib2to3.fixer_base import BaseFix

class FixThrow(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    power< any trailer< '.' 'throw' >
           trailer< '(' args=arglist< exc=any ',' val=any [',' tb=any] > ')' >
    >
    |
    power< any trailer< '.' 'throw' > trailer< '(' exc=any ')' > >
    """

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        syms = self.syms
        exc = results['exc'].clone()
        if exc.type is token.STRING:
            self.cannot_convert(node, 'Python 3 does not support string exceptions')
            return None
        val = results.get('val')
        if val is None:
            return None
        val = val.clone()
        if is_tuple(val):
            args = [c.clone() for c in val.children[1:-1]]
        else:
            val.prefix = ''
            args = [val]
        throw_args = results['args']
        if 'tb' in results:
            tb = results['tb'].clone()
            tb.prefix = ''
            e = Call(exc, args)
            with_tb = (Attr(e, Name('with_traceback')) + [ArgList([tb])])
            throw_args.replace(pytree.Node(syms.power, with_tb))
        else:
            throw_args.replace(Call(exc, args))
        return None
