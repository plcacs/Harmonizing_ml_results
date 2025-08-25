from typing import Dict, List, Optional
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import Comma, Name, Call

class FixExec(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "\n    exec_stmt< 'exec' a=any 'in' b=any [',' c=any] >\n    |\n    exec_stmt< 'exec' (not atom<'(' [any] ')'>) a=any >\n    "
    )

    def transform(self, node: Node, results: Dict[str, Optional[Node]]) -> Node:
        assert results
        syms = self.syms
        a: Node = results['a']  # 'a' is always present
        b: Optional[Node] = results.get('b')
        c: Optional[Node] = results.get('c')
        args: List[Node] = [a.clone()]
        args[0].prefix = ''
        if b is not None:
            args.extend([Comma(), b.clone()])
        if c is not None:
            args.extend([Comma(), c.clone()])
        return Call(Name('exec'), args, prefix=node.prefix)