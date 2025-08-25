'Fixer for execfile.\n\nThis converts usages of the execfile function into calls to the built-in\nexec() function.\n'
from typing import Any, Dict, List, Optional

from .. import fixer_base
from ..fixer_util import (
    ArgList,
    Comma,
    Dot,
    LParen,
    Name,
    Node,
    RParen,
    Call,
    String,
    syms,
)


class FixExecfile(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    power< 'execfile' trailer< '(' arglist< filename=any [',' globals=any [',' locals=any ] ] > ')' > >
    |
    power< 'execfile' trailer< '(' filename=any ')' > >
    """

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        assert results
        filename: Node = results['filename']
        globals_node: Optional[Node] = results.get('globals')
        locals_node: Optional[Node] = results.get('locals')
        execfile_paren: Node = node.children[-1].children[-1].clone()
        open_args: ArgList = ArgList(
            [filename.clone(), Comma(), String('"rb"', ' ')],
            rparen=execfile_paren,
        )
        open_call: Node = Node(syms.power, [Name('open'), open_args])
        read: List[Node] = [
            Node(syms.trailer, [Dot(), Name('read')]),
            Node(syms.trailer, [LParen(), RParen()]),
        ]
        open_expr: List[Node] = [open_call] + read
        filename_arg: Node = filename.clone()
        filename_arg.prefix = ' '
        exec_str: String = String("'exec'", ' ')
        compile_args: List[Any] = open_expr + [Comma(), filename_arg, Comma(), exec_str]
        compile_call: Call = Call(Name('compile'), compile_args, '')
        args: List[Any] = [compile_call]
        if globals_node is not None:
            args.extend([Comma(), globals_node.clone()])
        if locals_node is not None:
            args.extend([Comma(), locals_node.clone()])
        return Call(Name('exec'), args, prefix=node.prefix)
