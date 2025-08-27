from lib2to3 import fixer_base, pytree
from lib2to3.fixer_util import token, syms
from lib2to3.pgen2 import token as token_module
from lib2to3.pytree import Node, Leaf
from typing import Dict, Any, Optional, cast

class FixSetLiteral(fixer_base.BaseFix):
    BM_compatible: bool = True
    explicit: bool = True
    PATTERN: str = "power< 'set' trailer< '('\n                     (atom=atom< '[' (items=listmaker< any ((',' any)* [',']) >\n                                |\n                                single=any) ']' >\n                     |\n                     atom< '(' items=testlist_gexp< any ((',' any)* [',']) > ')' >\n                     )\n                     ')' > >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        single: Optional[Node] = results.get('single')
        items: Node
        if single:
            fake: Node = Node(syms.listmaker, [single.clone()])
            single.replace(fake)
            items = fake
        else:
            items = cast(Node, results['items'])
        literal: list = [Leaf(token.LBRACE, '{')]
        literal.extend((n.clone() for n in items.children))
        literal.append(Leaf(token.RBRACE, '}'))
        literal[-1].prefix = items.next_sibling.prefix
        maker: Node = Node(syms.dictsetmaker, literal)
        maker.prefix = node.prefix
        if len(maker.children) == 4:
            n: Node = maker.children[2]
            n.remove()
            maker.children[-1].prefix = n.prefix
        return maker
