'Fixer for except statements with named exceptions.\n\nThe following cases will be converted:\n\n- "except E, T:" where T is a name:\n\n    except E as T:\n\n- "except E, T:" where T is not a name, tuple or list:\n\n        except E as t:\n            T = t\n\n    This is done because the target of an "except" clause must be a\n    name.\n\n- "except E, T:" where T is a tuple or list literal:\n\n        except E as t:\n            T = t.args\n'
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Assign, Attr, Name, is_tuple, is_list, syms
from typing import Generator, Tuple, List, Any, Dict, Optional, Union

def find_excepts(nodes: List[Any]) -> Generator[Tuple[Any, Any], None, None]:
    for (i, n) in enumerate(nodes):
        if (n.type == syms.except_clause):
            if (n.children[0].value == 'except'):
                (yield (n, nodes[(i + 2)]))

class FixExcept(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    try_stmt< 'try' ':' (simple_stmt | suite)\n                  cleanup=(except_clause ':' (simple_stmt | suite))+\n                  tail=(['except' ':' (simple_stmt | suite)]\n                        ['else' ':' (simple_stmt | suite)]\n                        ['finally' ':' (simple_stmt | suite)]) >\n    "

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        syms = self.syms
        tail: List[Any] = [n.clone() for n in results['tail']]
        try_cleanup: List[Any] = [ch.clone() for ch in results['cleanup']]
        for (except_clause, e_suite) in find_excepts(try_cleanup):
            if (len(except_clause.children) == 4):
                (E, comma, N) = except_clause.children[1:4]
                comma.replace(Name('as', prefix=' '))
                if (N.type != token.NAME):
                    new_N: Name = Name(self.new_name(), prefix=' ')
                    target: Any = N.clone()
                    target.prefix = ''
                    N.replace(new_N)
                    new_N = new_N.clone()
                    suite_stmts: List[Any] = e_suite.children
                    for (i, stmt) in enumerate(suite_stmts):
                        if isinstance(stmt, pytree.Node):
                            break
                    if (is_tuple(N) or is_list(N)):
                        assign: Assign = Assign(target, Attr(new_N, Name('args')))
                    else:
                        assign = Assign(target, new_N)
                    for child in reversed(suite_stmts[:i]):
                        e_suite.insert_child(0, child)
                    e_suite.insert_child(i, assign)
                elif (N.prefix == ''):
                    N.prefix = ' '
        children: List[Any] = (([c.clone() for c in node.children[:3]] + try_cleanup) + tail)
        return pytree.Node(node.type, children)
