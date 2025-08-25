from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, ArgList, Call, ListComp, in_special_context
from ..pygram import python_symbols as syms
from ..pytree import Node

class FixMap(fixer_base.ConditionalFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    map_none=power<\n        'map'\n        trailer< '(' arglist< 'None' ',' arg=any [','] > ')' >\n        [extra_trailers=trailer*]\n    >\n    |\n    map_lambda=power<\n        'map'\n        trailer<\n            '('\n            arglist<\n                lambdef< 'lambda'\n                         (fp=NAME | vfpdef< '(' fp=NAME ')'> ) ':' xp=any\n                >\n                ','\n                it=any\n            >\n            ')'\n        >\n        [extra_trailers=trailer*]\n    >\n    |\n    power<\n        'map' args=trailer< '(' [any] ')' >\n        [extra_trailers=trailer*]\n    >\n    "
    skip_on: str = 'future_builtins.map'

    def transform(self, node: Node, results: dict) -> Node:
        if self.should_skip(node):
            return
        trailers: list = []
        if ('extra_trailers' in results):
            for t in results['extra_trailers']:
                trailers.append(t.clone())
        if (node.parent.type == syms.simple_stmt):
            self.warning(node, 'You should use a for loop here')
            new: Node = node.clone()
            new.prefix = ''
            new = Call(Name('list'), [new])
        elif ('map_lambda' in results):
            new: Node = ListComp(results['xp'].clone(), results['fp'].clone(), results['it'].clone())
            new = Node(syms.power, ([new] + trailers), prefix='')
        else:
            if ('map_none' in results):
                new: Node = results['arg'].clone()
                new.prefix = ''
            else:
                if ('args' in results):
                    args: Node = results['args']
                    if ((args.type == syms.trailer) and (args.children[1].type == syms.arglist) and (args.children[1].children[0].type == token.NAME) and (args.children[1].children[0].value == 'None')):
                        self.warning(node, 'cannot convert map(None, ...) with multiple arguments because map() now truncates to the shortest sequence')
                        return
                    new: Node = Node(syms.power, [Name('map'), args.clone()])
                    new.prefix = ''
                if in_special_context(node):
                    return None
            new: Node = Node(syms.power, ([Name('list'), ArgList([new])] + trailers))
            new.prefix = ''
        new.prefix = node.prefix
        return new
