'Fixer for \'raise E, V, T\'\n\nraise         -> raise\nraise E       -> raise E\nraise E, V    -> raise E(V)\nraise E, V, T -> raise E(V).with_traceback(T)\nraise E, None, T -> raise E.with_traceback(T)\n\nraise (((E, E\'), E\'\'), E\'\'\'), V -> raise E(V)\nraise "foo", V, T               -> warns about string exceptions\n\n\nCAVEATS:\n1) "raise E, V" will be incorrectly translated if V is an exception\n   instance. The correct Python 3 idiom is\n\n        raise E from V\n\n   but since we can\'t detect instance-hood by syntax alone and since\n   any client code would have to be changed as well, we don\'t automate\n   this.\n'
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, Call, Attr, ArgList, is_tuple
from typing import Dict, Any, Optional

class FixRaise(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    raise_stmt< 'raise' exc=any [',' val=any [',' tb=any]] >\n    "

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        syms = self.syms
        exc: pytree.Node = results['exc'].clone()
        if (exc.type == token.STRING):
            msg: str = 'Python 3 does not support string exceptions'
            self.cannot_convert(node, msg)
            return None
        if is_tuple(exc):
            while is_tuple(exc):
                exc = exc.children[1].children[0].clone()
            exc.prefix = ' '
        if ('val' not in results):
            new: pytree.Node = pytree.Node(syms.raise_stmt, [Name('raise'), exc])
            new.prefix = node.prefix
            return new
        val: pytree.Node = results['val'].clone()
        if is_tuple(val):
            args: List[pytree.Node] = [c.clone() for c in val.children[1:(- 1)]]
        else:
            val.prefix = ''
            args = [val]
        if ('tb' in results):
            tb: pytree.Node = results['tb'].clone()
            tb.prefix = ''
            e: pytree.Node = exc
            if ((val.type != token.NAME) or (val.value != 'None')):
                e = Call(exc, args)
            with_tb: List[pytree.Node] = (Attr(e, Name('with_traceback')) + [ArgList([tb])])
            new = pytree.Node(syms.simple_stmt, ([Name('raise')] + with_tb))
            new.prefix = node.prefix
            return new
        else:
            return pytree.Node(syms.raise_stmt, [Name('raise'), Call(exc, args)], prefix=node.prefix)
