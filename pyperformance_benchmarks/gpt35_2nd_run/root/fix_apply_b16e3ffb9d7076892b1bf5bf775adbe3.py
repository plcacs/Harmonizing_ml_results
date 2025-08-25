from typing import Any
from lib2to3 import pytree
from lib2to3.pgen2 import token
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Comma, parenthesize

class FixApply(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< 'apply'\n        trailer<\n            '('\n            arglist<\n                (not argument<NAME '=' any>) func=any ','\n                (not argument<NAME '=' any>) args=any [','\n                (not argument<NAME '=' any>) kwds=any] [',']\n            >\n            ')'\n        >\n    >\n    "

    def transform(self, node: pytree.Node, results: dict[str, Any]) -> Call:
        syms: dict[str, int] = self.syms
        assert results
        func: pytree.Node = results['func']
        args: pytree.Node = results['args']
        kwds: pytree.Node = results.get('kwds')
        if args:
            if ((args.type == self.syms.argument) and (args.children[0].value in {'**', '*'})):
                return
        if (kwds and ((kwds.type == self.syms.argument) and (kwds.children[0].value == '**'))):
            return
        prefix: str = node.prefix
        func = func.clone()
        if ((func.type not in (token.NAME, syms.atom)) and ((func.type != syms.power) or (func.children[(- 2)].type == token.DOUBLESTAR))):
            func = parenthesize(func)
        func.prefix = ''
        args = args.clone()
        args.prefix = ''
        if (kwds is not None):
            kwds = kwds.clone()
            kwds.prefix = ''
        l_newargs = [pytree.Leaf(token.STAR, '*'), args]
        if (kwds is not None):
            l_newargs.extend([Comma(), pytree.Leaf(token.DOUBLESTAR, '**'), kwds])
            l_newargs[(- 2)].prefix = ' '
        return Call(func, l_newargs, prefix=prefix)
