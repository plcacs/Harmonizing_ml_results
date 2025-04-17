'Fixer for execfile.\n\nThis converts usages of the execfile function into calls to the built-in\nexec() function.\n'
from .. import fixer_base
from ..fixer_util import Comma, Name, Call, LParen, RParen, Dot, Node, ArgList, String, syms
from typing import Optional, Dict, Any

class FixExecfile(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< 'execfile' trailer< '(' arglist< filename=any [',' globals=any [',' locals=any ] ] > ')' > >\n    |\n    power< 'execfile' trailer< '(' filename=any ')' > >\n    "

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        assert results
        filename: Node = results['filename']
        globals: Optional[Node] = results.get('globals')
        locals: Optional[Node] = results.get('locals')
        execfile_paren: Node = node.children[(- 1)].children[(- 1)].clone()
        open_args: ArgList = ArgList([filename.clone(), Comma(), String('"rb"', ' ')], rparen=execfile_paren)
        open_call: Node = Node(syms.power, [Name('open'), open_args])
        read: list[Node] = [Node(syms.trailer, [Dot(), Name('read')]), Node(syms.trailer, [LParen(), RParen()])]
        open_expr: list[Node] = ([open_call] + read)
        filename_arg: Node = filename.clone()
        filename_arg.prefix = ' '
        exec_str: String = String("'exec'", ' ')
        compile_args: list[Node] = (open_expr + [Comma(), filename_arg, Comma(), exec_str])
        compile_call: Call = Call(Name('compile'), compile_args, '')
        args: list[Node] = [compile_call]
        if (globals is not None):
            args.extend([Comma(), globals.clone()])
        if (locals is not None):
            args.extend([Comma(), locals.clone()])
        return Call(Name('exec'), args, prefix=node.prefix)
