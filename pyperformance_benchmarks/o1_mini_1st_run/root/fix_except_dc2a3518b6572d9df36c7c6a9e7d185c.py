from typing import Iterator, Tuple, List, Dict, Any, Optional
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Assign, Attr, Name, is_tuple, is_list, syms

def find_excepts(nodes: List[pytree.Node]) -> Iterator[Tuple[pytree.Node, pytree.Node]]:
    for i, n in enumerate(nodes):
        if n.type == syms.except_clause:
            if n.children[0].value == 'except':
                yield (n, nodes[i + 2])

class FixExcept(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    try_stmt< 'try' ':' (simple_stmt | suite)
              cleanup=(except_clause ':' (simple_stmt | suite))+
              tail=(['except' ':' (simple_stmt | suite)]
                    ['else' ':' (simple_stmt | suite)]
                    ['finally' ':' (simple_stmt | suite)]) >
    """

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        syms = self.syms
        tail = [n.clone() for n in results['tail']]
        try_cleanup = [ch.clone() for ch in results['cleanup']]
        for except_clause, e_suite in find_excepts(try_cleanup):
            if len(except_clause.children) == 4:
                E, comma, N = except_clause.children[1:4]
                comma.replace(Name('as', prefix=' '))
                if N.type != token.NAME:
                    new_N = Name(self.new_name(), prefix=' ')
                    target = N.clone()
                    target.prefix = ''
                    N.replace(new_N)
                    new_N = new_N.clone()
                    suite_stmts = e_suite.children
                    for i, stmt in enumerate(suite_stmts):
                        if isinstance(stmt, pytree.Node):
                            break
                    if is_tuple(N) or is_list(N):
                        assign = Assign(target, Attr(new_N, Name('args')))
                    else:
                        assign = Assign(target, new_N)
                    for child in reversed(suite_stmts[:i]):
                        e_suite.insert_child(0, child)
                    e_suite.insert_child(i, assign)
                elif N.prefix == '':
                    N.prefix = ' '
        children = ([c.clone() for c in node.children[:3]] + try_cleanup) + tail
        return pytree.Node(node.type, children)
