from typing import Generator, Tuple, List, Any, Dict, Iterator
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Assign, Attr, Name, is_tuple, is_list, syms

def find_excepts(nodes: List[pytree.Leaf]) -> Generator[Tuple[pytree.Leaf, pytree.Leaf], None, None]:
    for (i, n) in enumerate(nodes):
        if (n.type == syms.except_clause):
            if (n.children[0].value == 'except'):
                yield (n, nodes[(i + 2)])

class FixExcept(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    try_stmt< 'try' ':' (simple_stmt | suite)\n                  cleanup=(except_clause ':' (simple_stmt | suite))+\n                  tail=(['except' ':' (simple_stmt | suite)]\n                        ['else' ':' (simple_stmt | suite)]\n                        ['finally' ':' (simple_stmt | suite)]) >\n    "

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> pytree.Node:
        syms = self.syms
        tail: List[pytree.Leaf] = [n.clone() for n in results['tail']]
        try_cleanup: List[pytree.Leaf] = [ch.clone() for ch in results['cleanup']]
        for (except_clause, e_suite) in find_excepts(try_cleanup):
            if (len(except_clause.children) == 4):
                (E, comma, N) = except_clause.children[1:4]
                comma.replace(Name('as', prefix=' '))
                if (N.type != token.NAME):
                    new_N: pytree.Leaf = Name(self.new_name(), prefix=' ')
                    target: pytree.Leaf = N.clone()
                    target.prefix = ''
                    N.replace(new_N)
                    new_N = new_N.clone()
                    suite_stmts: List[pytree.Leaf] = e_suite.children
                    for (i, stmt) in enumerate(suite_stmts):
                        if isinstance(stmt, pytree.Node):
                            break
                    if (is_tuple(N) or is_list(N)):
                        assign: pytree.Leaf = Assign(target, Attr(new_N, Name('args')))
                    else:
                        assign = Assign(target, new_N)
                    for child in reversed(suite_stmts[:i]):
                        e_suite.insert_child(0, child)
                    e_suite.insert_child(i, assign)
                elif (N.prefix == ''):
                    N.prefix = ' '
        children: List[pytree.Leaf] = (([c.clone() for c in node.children[:3]] + try_cleanup) + tail)
        return pytree.Node(node.type, children)
