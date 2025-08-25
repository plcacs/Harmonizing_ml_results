from lib2to3 import fixer_base, pytree
from lib2to3.fixer_util import token, syms
from typing import Optional, Dict, Any, List

class FixSetLiteral(fixer_base.BaseFix):
    BM_compatible: bool = True
    explicit: bool = True
    PATTERN: str = (
        "power< 'set' trailer< '('\n"
        "                     (atom=atom< '[' (items=listmaker< any (',' any)* [','] >\n"
        "                                |\n"
        "                                single=any) ']' >\n"
        "                     |\n"
        "                     atom< '(' items=testlist_gexp< any (',' any)* [','] > ')' >\n"
        "                     )\n"
        "                     ')' > >\n"
    )

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        single = results.get('single')
        if single:
            fake: pytree.Node = pytree.Node(syms.listmaker, [single.clone()])
            single.replace(fake)
            items: pytree.Node = fake
        else:
            items = results['items']  # type: pytree.Node
        literal: List[pytree.Leaf] = [pytree.Leaf(token.LBRACE, '{')]
        literal.extend(n.clone() for n in items.children)
        literal.append(pytree.Leaf(token.RBRACE, '}'))
        literal[-1].prefix = items.next_sibling.prefix  # type: ignore
        maker: pytree.Node = pytree.Node(syms.dictsetmaker, literal)
        maker.prefix = node.prefix
        if len(maker.children) == 4:
            n: pytree.Node = maker.children[2]
            n.remove()
            maker.children[-1].prefix = n.prefix
        return maker
