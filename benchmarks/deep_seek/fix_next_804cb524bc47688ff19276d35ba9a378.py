"""Fixer for it.next() -> next(it), per PEP 3114."""
from typing import Optional, Dict, Any, List, Union
from ..pgen2 import token
from ..pygram import python_symbols as syms
from .. import fixer_base
from ..fixer_util import Name, Call, find_binding

bind_warning: str = 'Calls to builtin next() possibly shadowed by global binding'

class FixNext(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    power< base=any+ trailer< '.' attr='next' > trailer< '(' ')' > >
    |
    power< head=any+ trailer< '.' attr='next' > not trailer< '(' ')' > >
    |
    classdef< 'class' any+ ':'
              suite< any*
                     funcdef< 'def'
                              name='next'
                              parameters< '(' NAME ')' > any+ >
                     any* > >
    |
    global=global_stmt< 'global' any* 'next' any* >
    """
    order: str = 'pre'
    shadowed_next: bool

    def start_tree(self, tree: Any, filename: str) -> None:
        super(FixNext, self).start_tree(tree, filename)
        n = find_binding('next', tree)
        if n:
            self.warning(n, bind_warning)
            self.shadowed_next = True
        else:
            self.shadowed_next = False

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        assert results
        base: Optional[List[Any]] = results.get('base')
        attr: Optional[Any] = results.get('attr')
        name: Optional[Any] = results.get('name')
        if base:
            if self.shadowed_next:
                attr.replace(Name('__next__', prefix=attr.prefix))
            else:
                base = [n.clone() for n in base]
                base[0].prefix = ''
                node.replace(Call(Name('next', prefix=node.prefix), base))
        elif name:
            n = Name('__next__', prefix=name.prefix)
            name.replace(n)
        elif attr:
            if is_assign_target(node):
                head = results['head']
                if (''.join([str(n) for n in head]).strip() == '__builtin__'):
                    self.warning(node, bind_warning)
                return
            attr.replace(Name('__next__'))
        elif ('global' in results):
            self.warning(node, bind_warning)
            self.shadowed_next = True

def is_assign_target(node: Any) -> bool:
    assign: Optional[Any] = find_assign(node)
    if (assign is None):
        return False
    for child in assign.children:
        if (child.type == token.EQUAL):
            return False
        elif is_subtree(child, node):
            return True
    return False

def find_assign(node: Any) -> Optional[Any]:
    if (node.type == syms.expr_stmt):
        return node
    if ((node.type == syms.simple_stmt) or (node.parent is None)):
        return None
    return find_assign(node.parent)

def is_subtree(root: Any, node: Any) -> bool:
    if (root == node):
        return True
    return any((is_subtree(c, node) for c in root.children)
