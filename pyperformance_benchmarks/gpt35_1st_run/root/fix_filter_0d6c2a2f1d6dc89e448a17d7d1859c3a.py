from typing import Any
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.fixes import fixer_util
from typing import List

class FixFilter(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    filter_lambda=power<\n        'filter'\n        trailer<\n            '('\n            arglist<\n                lambdef< 'lambda'\n                         (fp=NAME | vfpdef< '(' fp=NAME ')'> ) ':' xp=any\n                >\n                ','\n                it=any\n            >\n            ')'\n        >\n        [extra_trailers=trailer*]\n    >\n    |\n    power<\n        'filter'\n        trailer< '(' arglist< none='None' ',' seq=any > ')' >\n        [extra_trailers=trailer*]\n    >\n    |\n    power<\n        'filter'\n        args=trailer< '(' [any] ')' >\n        [extra_trailers=trailer*]\n    >\n    "
    skip_on: str = 'future_builtins.filter'

    def transform(self, node: Node, results: dict) -> Node:
        if self.should_skip(node):
            return
        trailers: List[Node] = []
        if ('extra_trailers' in results):
            for t in results['extra_trailers']:
                trailers.append(t.clone())
        if ('filter_lambda' in results):
            xp: Node = results.get('xp').clone()
            if (xp.type == syms.test):
                xp.prefix = ''
                xp = fixer_util.parenthesize(xp)
            new: Node = fixer_util.ListComp(results.get('fp').clone(), results.get('fp').clone(), results.get('it').clone(), xp)
            new = Node(syms.power, ([new] + trailers), prefix='')
        elif ('none' in results):
            new: Node = fixer_util.ListComp(fixer_util.Name('_f'), fixer_util.Name('_f'), results['seq'].clone(), fixer_util.Name('_f'))
            new = Node(syms.power, ([new] + trailers), prefix='')
        else:
            if fixer_util.in_special_context(node):
                return None
            args: Node = results['args'].clone()
            new: Node = Node(syms.power, [fixer_util.Name('filter'), args], prefix='')
            new = Node(syms.power, ([fixer_util.Name('list'), fixer_util.ArgList([new])] + trailers))
            new.prefix = ''
        new.prefix = node.prefix
        return new
