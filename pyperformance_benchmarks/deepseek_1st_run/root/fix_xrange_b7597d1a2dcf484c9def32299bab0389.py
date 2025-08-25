'Fixer that changes xrange(...) into range(...).'
from .. import fixer_base
from ..fixer_util import Name, Call, consuming_calls
from .. import patcomp
from typing import Set, Optional, Dict, Any, Pattern
from lib2to3.pytree import Node, Leaf
from lib2to3.pygram import python_symbols as syms

class FixXrange(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power<\n                 (name='range'|name='xrange') trailer< '(' args=any ')' >\n              rest=any* >\n              "

    def start_tree(self, tree: Node, filename: str) -> None:
        super(FixXrange, self).start_tree(tree, filename)
        self.transformed_xranges: Set[int] = set()

    def finish_tree(self, tree: Node, filename: str) -> None:
        self.transformed_xranges = None

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        name: Leaf = results['name']
        if (name.value == 'xrange'):
            return self.transform_xrange(node, results)
        elif (name.value == 'range'):
            return self.transform_range(node, results)
        else:
            raise ValueError(repr(name))

    def transform_xrange(self, node: Node, results: Dict[str, Any]) -> None:
        name: Leaf = results['name']
        name.replace(Name('range', prefix=name.prefix))
        self.transformed_xranges.add(id(node))

    def transform_range(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        if ((id(node) not in self.transformed_xranges) and (not self.in_special_context(node))):
            range_call: Call = Call(Name('range'), [results['args'].clone()])
            list_call: Call = Call(Name('list'), [range_call], prefix=node.prefix)
            for n in results['rest']:
                list_call.append_child(n)
            return list_call
        return None
    P1: str = "power< func=NAME trailer< '(' node=any ')' > any* >"
    p1: Pattern = patcomp.compile_pattern(P1)
    P2: str = "for_stmt< 'for' any 'in' node=any ':' any* >\n            | comp_for< 'for' any 'in' node=any any* >\n            | comparison< any 'in' node=any any*>\n         "
    p2: Pattern = patcomp.compile_pattern(P2)

    def in_special_context(self, node: Node) -> bool:
        if (node.parent is None):
            return False
        results: Dict[str, Any] = {}
        if ((node.parent.parent is not None) and self.p1.match(node.parent.parent, results) and (results['node'] is node)):
            return (results['func'].value in consuming_calls)
        return (self.p2.match(node.parent, results) and (results['node'] is node))
