'Fixer for exec.\n\nThis converts usages of the exec statement into calls to a built-in\nexec() function.\n\nexec code in ns1, ns2 -> exec(code, ns1, ns2)\n'
from typing import Optional, Dict, Any
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import Comma, Name, Call

class FixExec(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    exec_stmt< 'exec' a=any 'in' b=any [',' c=any] >\n    |\n    exec_stmt< 'exec' (not atom<'(' [any] ')'>) a=any >\n    "

    def transform(self, node: Node, results: Dict[str, Node]) -> Call:
        assert results
        syms: Any = self.syms
        a: Node = results['a']
        b: Optional[Node] = results.get('b')
        c: Optional[Node] = results.get('c')
        args: list = [a.clone()]
        args[0].prefix = ''
        if b is not None:
            args.extend([Comma(), b.clone()])
        if c is not None:
            args.extend([Comma(), c.clone()])
        return Call(Name('exec'), args, prefix=node.prefix)
