from typing import Any
from lib2to3 import pytree
from lib2to3.pgen2 import token
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, Call, Attr, ArgList, is_tuple

class FixRaise(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    raise_stmt< 'raise' exc=any [',' val=any [',' tb=any]] >\n    "

    def transform(self, node: Any, results: Any) -> Any:
        syms: Any = self.syms
        exc: Any = results['exc'].clone()
        if (exc.type == token.STRING):
            msg: str = 'Python 3 does not support string exceptions'
            self.cannot_convert(node, msg)
            return
        if is_tuple(exc):
            while is_tuple(exc):
                exc = exc.children[1].children[0].clone()
            exc.prefix = ' '
        if ('val' not in results):
            new: Any = pytree.Node(syms.raise_stmt, [Name('raise'), exc])
            new.prefix = node.prefix
            return new
        val: Any = results['val'].clone()
        if is_tuple(val):
            args: List[Any] = [c.clone() for c in val.children[1:(- 1)]]
        else:
            val.prefix = ''
            args: List[Any] = [val]
        if ('tb' in results):
            tb: Any = results['tb'].clone()
            tb.prefix = ''
            e: Any = exc
            if ((val.type != token.NAME) or (val.value != 'None')):
                e = Call(exc, args)
            with_tb: Any = (Attr(e, Name('with_traceback')) + [ArgList([tb])])
            new: Any = pytree.Node(syms.simple_stmt, ([Name('raise')] + with_tb))
            new.prefix = node.prefix
            return new
        else:
            return pytree.Node(syms.raise_stmt, [Name('raise'), Call(exc, args)], prefix=node.prefix)
