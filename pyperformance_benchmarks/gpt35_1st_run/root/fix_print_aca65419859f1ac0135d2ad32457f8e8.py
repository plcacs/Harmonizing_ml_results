from __future__ import annotations
from typing import List
from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Name, Call, Comma, String
from lib2to3.pytree import Leaf, Node
from lib2to3.pgen2 import token
from lib2to3.patcomp import compile_pattern

parend_expr = compile_pattern("atom< '(' [atom|STRING|NAME] ')' >")

class FixPrint(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              simple_stmt< any* bare='print' any* > | print_stmt\n              "

    def transform(self, node: Node, results: dict[str, Node]) -> Node:
        assert results
        bare_print: Node = results.get('bare')
        if bare_print:
            bare_print.replace(Call(Name('print'), [], prefix=bare_print.prefix))
            return
        assert (node.children[0] == Name('print'))
        args: List[Node] = node.children[1:]
        if ((len(args) == 1) and parend_expr.match(args[0])):
            return
        sep: str = end: str = file: Node = None
        if (args and (args[(- 1)] == Comma())):
            args = args[:(- 1)]
            end = ' '
        if (args and (args[0] == Leaf(token.RIGHTSHIFT, '>>'))):
            assert (len(args) >= 2)
            file = args[1].clone()
            args = args[3:]
        l_args: List[Node] = [arg.clone() for arg in args]
        if l_args:
            l_args[0].prefix = ''
        if ((sep is not None) or (end is not None) or (file is not None)):
            if (sep is not None):
                self.add_kwarg(l_args, 'sep', String(repr(sep)))
            if (end is not None):
                self.add_kwarg(l_args, 'end', String(repr(end)))
            if (file is not None):
                self.add_kwarg(l_args, 'file', file)
        n_stmt: Node = Call(Name('print'), l_args)
        n_stmt.prefix = node.prefix
        return n_stmt

    def add_kwarg(self, l_nodes: List[Node], s_kwd: str, n_expr: Node) -> None:
        n_expr.prefix = ''
        n_argument: Node = Node(self.syms.argument, (Name(s_kwd), Leaf(token.EQUAL, '='), n_expr))
        if l_nodes:
            l_nodes.append(Comma())
            n_argument.prefix = ' '
        l_nodes.append(n_argument)
