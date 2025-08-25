from __future__ import annotations
from typing import Optional, Dict
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, Call, Attr, ArgList, is_tuple


class FixRaise(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    raise_stmt< 'raise' exc=any [',' val=any [',' tb=any]] >
    """

    def transform(self, node: pytree.Node, results: Dict[str, pytree.Node]) -> Optional[pytree.Node]:
        syms = self.syms
        exc: pytree.Node = results['exc'].clone()
        if exc.type == token.STRING:
            msg: str = 'Python 3 does not support string exceptions'
            self.cannot_convert(node, msg)
            return None
        if is_tuple(exc):
            while is_tuple(exc):
                exc = exc.children[1].children[0].clone()
            exc.prefix = ' '
        if 'val' not in results:
            new: pytree.Node = pytree.Node(syms.raise_stmt, [Name('raise'), exc])
            new.prefix = node.prefix
            return new
        val: pytree.Node = results['val'].clone()
        if is_tuple(val):
            args: list[pytree.Node] = [c.clone() for c in val.children[1:-1]]
        else:
            val.prefix = ''
            args = [val]
        if 'tb' in results:
            tb: pytree.Node = results['tb'].clone()
            tb.prefix = ''
            e: pytree.Node | Call = exc
            if (val.type != token.NAME) or (val.value != 'None'):
                e = Call(exc, args)
            with_tb = Attr(e, Name('with_traceback')) + [ArgList([tb])]
            new = pytree.Node(syms.simple_stmt, [Name('raise')] + with_tb)
            new.prefix = node.prefix
            return new
        else:
            return pytree.Node(syms.raise_stmt, [Name('raise'), Call(exc, args)], prefix=node.prefix)
