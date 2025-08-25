'\nOptional fixer to transform set() calls to set literals.\n'
from lib2to3 import fixer_base, pytree
from lib2to3.fixer_util import token, syms
from typing import Dict, Any, Optional

class FixSetLiteral(fixer_base.BaseFix):
    BM_compatible: bool = True
    explicit: bool = True
    PATTERN: str = (
        "power< 'set' trailer< '('\n"
        "                     (atom=atom< '[' (items=listmaker< any ((',' any)* [',']) >\n"
        "                                |\n"
        "                                single=any) ']' >\n"
        "                     |\n"
        "                     atom< '(' items=testlist_gexp< any ((',' any)* [',']) > ')' >\n"
        "                     )\n"
        "                     ')' > >\n"
    )

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        single: Optional[pytree.Node] = results.get('single')
        if single:
            fake: pytree.Node = pytree.Node(syms.listmaker, [single.clone()])
            single.replace(fake)
            items: pytree.Node = fake
        else:
            items = results['items']
        literal: list[pytree.Leaf] = [pytree.Leaf(token.LBRACE, '{')]
        literal.extend((n.clone() for n in items.children))
        literal.append(pytree.Leaf(token.RBRACE, '}'))
        literal[(- 1)].prefix = items.next_sibling.prefix
        maker: pytree.Node = pytree.Node(syms.dictsetmaker, literal)
        maker.prefix = node.prefix
        if (len(maker.children) == 4):
            n: pytree.Leaf = maker.children[2]
            n.remove()
            maker.children[(- 1)].prefix = n.prefix
        return maker
