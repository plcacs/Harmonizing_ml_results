'Fixer for generator.throw(E, V, T).\n\ng.throw(E)       -> g.throw(E)\ng.throw(E, V)    -> g.throw(E(V))\ng.throw(E, V, T) -> g.throw(E(V).with_traceback(T))\n\ng.throw("foo"[, V[, T]]) will warn about string exceptions.'
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, Call, ArgList, Attr, is_tuple
from typing import Dict, Optional, Union, Any, List, cast

class FixThrow(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< any trailer< '.' 'throw' >\n           trailer< '(' args=arglist< exc=any ',' val=any [',' tb=any] > ')' >\n    >\n    |\n    power< any trailer< '.' 'throw' > trailer< '(' exc=any ')' > >\n    "

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        syms = self.syms
        exc = results['exc'].clone()
        if (exc.type is token.STRING):
            self.cannot_convert(node, 'Python 3 does not support string exceptions')
            return None
        val = results.get('val')
        if (val is None):
            return None
        val = val.clone()
        if is_tuple(val):
            args: List[pytree.Node] = [c.clone() for c in val.children[1:(- 1)]]
        else:
            val.prefix = ''
            args = [val]
        throw_args = results['args']
        if ('tb' in results):
            tb = results['tb'].clone()
            tb.prefix = ''
            e = Call(exc, args)
            with_tb = (Attr(e, Name('with_traceback')) + [ArgList([tb])])
            throw_args.replace(pytree.Node(syms.power, with_tb))
        else:
            throw_args.replace(Call(exc, args))
        return None
