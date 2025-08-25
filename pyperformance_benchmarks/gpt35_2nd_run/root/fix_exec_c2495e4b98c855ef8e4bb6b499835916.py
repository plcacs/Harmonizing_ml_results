from .. import fixer_base
from ..fixer_util import Comma, Name, Call
from typing import Any

class FixExec(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    exec_stmt< 'exec' a=any 'in' b=any [',' c=any] >\n    |\n    exec_stmt< 'exec' (not atom<'(' [any] ')'>) a=any >\n    "

    def transform(self, node: Any, results: Any) -> Any:
        assert results
        syms = self.syms
        a: Any = results['a']
        b: Any = results.get('b')
        c: Any = results.get('c')
        args: list = [a.clone()]
        args[0].prefix = ''
        if (b is not None):
            args.extend([Comma(), b.clone()])
        if (c is not None):
            args.extend([Comma(), c.clone()])
        return Call(Name('exec'), args, prefix=node.prefix)
