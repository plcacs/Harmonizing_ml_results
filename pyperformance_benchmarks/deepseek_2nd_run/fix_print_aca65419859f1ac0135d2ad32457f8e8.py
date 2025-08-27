'Fixer for print.\n\nChange:\n    \'print\'          into \'print()\'\n    \'print ...\'      into \'print(...)\'\n    \'print ... ,\'    into \'print(..., end=" ")\'\n    \'print >>x, ...\' into \'print(..., file=x)\'\n\nNo changes are applied if print_function is imported from __future__\n\n'
from .. import patcomp
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, Call, Comma, String
from typing import Dict, Any, Optional, List, Union

parend_expr = patcomp.compile_pattern("atom< '(' [atom|STRING|NAME] ')' >")

class FixPrint(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              simple_stmt< any* bare='print' any* > | print_stmt\n              "

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        assert results
        bare_print: Optional[pytree.Node] = results.get('bare')
        if bare_print:
            bare_print.replace(Call(Name('print'), [], prefix=bare_print.prefix))
            return None
        assert (node.children[0] == Name('print'))
        args: List[pytree.Node] = node.children[1:]
        if ((len(args) == 1) and parend_expr.match(args[0])):
            return None
        sep: Optional[str] = None
        end: Optional[str] = None
        file: Optional[pytree.Node] = None
        if (args and (args[(- 1)] == Comma())):
            args = args[:(- 1)]
            end = ' '
        if (args and (args[0] == pytree.Leaf(token.RIGHTSHIFT, '>>'))):
            assert (len(args) >= 2)
            file = args[1].clone()
            args = args[3:]
        l_args: List[Union[pytree.Node, pytree.Leaf]] = [arg.clone() for arg in args]
        if l_args:
            l_args[0].prefix = ''
        if ((sep is not None) or (end is not None) or (file is not None)):
            if (sep is not None):
                self.add_kwarg(l_args, 'sep', String(repr(sep)))
            if (end is not None):
                self.add_kwarg(l_args, 'end', String(repr(end)))
            if (file is not None):
                self.add_kwarg(l_args, 'file', file)
        n_stmt: Call = Call(Name('print'), l_args)
        n_stmt.prefix = node.prefix
        return n_stmt

    def add_kwarg(self, l_nodes: List[Union[pytree.Node, pytree.Leaf]], s_kwd: str, n_expr: Union[pytree.Node, pytree.Leaf]) -> None:
        n_expr.prefix = ''
        n_argument: pytree.Node = pytree.Node(self.syms.argument, (Name(s_kwd), pytree.Leaf(token.EQUAL, '='), n_expr))
        if l_nodes:
            l_nodes.append(Comma())
            n_argument.prefix = ' '
        l_nodes.append(n_argument)
