"""Fixer for execfile.

This converts usages of the execfile function into calls to the built-in
exec() function.
"""
from typing import Any, Dict, Optional

from .. import fixer_base
from ..fixer_util import (
    ArgList,
    Call,
    Comma,
    Dot,
    LParen,
    Name,
    Node,
    RParen,
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
        filename = results["filename"]
        globals: Optional[Any] = results.get("globals")
        locals: Optional[Any] = results.get("locals")
        execfile_paren = node.children[-1].children[-1].clone()
        open_args = ArgList(
            [filename.clone(), Comma(), String('"rb"', " ")],
            rparen=execfile_paren,
        )
        open_call = Node(syms.power, [Name("open"), open_args])
        read = [
            Node(syms.trailer, [Dot(), Name("read")]),
            Node(syms.trailer, [LParen(), RParen()]),
        ]
        open_expr = [open_call] + read
        filename_arg = filename.clone()
        filename_arg.prefix = " "
        exec_str = String("'exec'", " ")
        compile_args = open_expr + [Comma(), filename_arg, Comma(), exec_str]
        compile_call = Call(Name("compile"), compile_args, "")
        args = [compile_call]
        if globals is not None:
            args.extend([Comma(), globals.clone()])
        if locals is not None:
            args.extend([Comma(), locals.clone()])
        return Call(Name("exec"), args, prefix=node.prefix)
