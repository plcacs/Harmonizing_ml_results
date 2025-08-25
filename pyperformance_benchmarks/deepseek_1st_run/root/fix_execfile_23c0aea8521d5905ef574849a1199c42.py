'Fixer for execfile.\n\nThis converts usages of the execfile function into calls to the built-in\nexec() function.\n'
from .. import fixer_base
from ..fixer_util import Comma, Name, Call, LParen, RParen, Dot, Node, ArgList, String, syms
from typing import Any, Dict, Optional

class FixExecfile(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< 'execfile' trailer< '(' arglist< filename=any [',' globals=any [',' locals=any ] ] > ')' > >\n    |\n    power< 'execfile' trailer< '(' filename=any ')' > >\n    "

    def transform(self, node: Node, results: Dict[str, Any]) -> Call:
        assert results
        filename: Any = results['filename']
        globals: Optional[Any] = results.get('globals')
        locals: Optional[Any] = results.get('locals')
        execfile_paren: Any = node.children[(- 1)].children[(- 1)].clone()
        open_args: ArgList = ArgList([filename.clone(), Comma(), String('"rb"', ' ')], rparen=execfile_paren)
        open_call: Node = Node(syms.power, [Name('open'), open_args])
        read: list[Node] = [Node(syms.trailer, [Dot(), Name('read')]), Node(syms.trailer, [LParen(), RParen()])]
        open_expr: list[Node] = ([open_call] + read)
        filename_arg: Any = filename.clone()
        filename_arg.prefix = ' '
        exec_str: String = String("'exec'", ' ')
        compile_args: list[Any] = (open_expr + [Comma(), filename_arg, Comma(), exec_str])
        compile_call: Call = Call(Name('compile'), compile_args, '')
        args: list[Any] = [compile_call]
        if (globals is not None):
            args.extend([Comma(), globals.clone()])
        if (locals is not None):
            args.extend([Comma(), locals.clone()])
        return Call(Name('exec'), args, prefix=node.prefix)
