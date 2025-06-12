from __future__ import annotations
from typing import Any, Dict, List, Optional
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, ArgList, Call, ListComp, in_special_context
from ..pygram import python_symbols as syms
from ..pytree import Node

class FixMap(fixer_base.ConditionalFix):
    BM_compatible: bool = True
    PATTERN: str = """
        map_none=power<
            'map'
            trailer< '(' arglist< 'None' ',' arg=any [','] > ')' >
            [extra_trailers=trailer*]
        >
        |
        map_lambda=power<
            'map'
            trailer<
                '('
                arglist<
                    lambdef< 'lambda'
                             (fp=NAME | vfpdef< '(' fp=NAME ')'> ) ':' xp=any
                    >
                    ','
                    it=any
                >
                ')'
            >
            [extra_trailers=trailer*]
        >
        |
        power<
            'map' args=trailer< '(' [any] ')' >
            [extra_trailers=trailer*]
        >
    """
    skip_on: str = 'future_builtins.map'

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        if self.should_skip(node):
            return None
        trailers: List[Node] = []
        if 'extra_trailers' in results:
            for t in results['extra_trailers']:
                trailers.append(t.clone())
        if node.parent.type == syms.simple_stmt:
            self.warning(node, 'You should use a for loop here')
            new: Node = node.clone()
            new.prefix = ''
            new = Call(Name('list'), [new])
        elif 'map_lambda' in results:
            new = ListComp(results['xp'].clone(), results['fp'].clone(), results['it'].clone())
            new = Node(syms.power, [new] + trailers, prefix='')
        else:
            if 'map_none' in results:
                new = results['arg'].clone()
                new.prefix = ''
            else:
                if 'args' in results:
                    args = results['args']
                    if (
                        args.type == syms.trailer
                        and args.children[1].type == syms.arglist
                        and args.children[1].children[0].type == token.NAME
                        and args.children[1].children[0].value == 'None'
                    ):
                        self.warning(node, 'cannot convert map(None, ...) with multiple arguments because map() now truncates to the shortest sequence')
                        return None
                    new = Node(syms.power, [Name('map'), args.clone()])
                    new.prefix = ''
                if in_special_context(node):
                    return None
            new = Node(syms.power, [Name('list'), ArgList([new])] + trailers)
            new.prefix = ''
        new.prefix = node.prefix
        return new
