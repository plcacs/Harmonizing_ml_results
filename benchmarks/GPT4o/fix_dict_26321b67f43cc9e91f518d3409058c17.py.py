from typing import Any, List, Optional
from .. import pytree
from .. import patcomp
from .. import fixer_base
from ..fixer_util import Name, Call, Dot
from .. import fixer_util

iter_exempt = (fixer_util.consuming_calls | {'iter'})

class FixDict(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    power< head=any+
         trailer< '.' method=('keys'|'items'|'values'|
                              'iterkeys'|'iteritems'|'itervalues'|
                              'viewkeys'|'viewitems'|'viewvalues') >
         parens=trailer< '(' ')' >
         tail=any*
    >
    """

    def transform(self, node: pytree.Node, results: dict) -> Optional[pytree.Node]:
        head: List[pytree.Node] = results['head']
        method: pytree.Leaf = results['method'][0]
        tail: List[pytree.Node] = results['tail']
        syms = self.syms
        method_name: str = method.value
        isiter: bool = method_name.startswith('iter')
        isview: bool = method_name.startswith('view')
        if (isiter or isview):
            method_name = method_name[4:]
        assert (method_name in ('keys', 'items', 'values')), repr(method)
        head = [n.clone() for n in head]
        tail = [n.clone() for n in tail]
        special: bool = ((not tail) and self.in_special_context(node, isiter))
        args: List[Any] = (head + [pytree.Node(syms.trailer, [Dot(), Name(method_name, prefix=method.prefix)]), results['parens'].clone()])
        new: pytree.Node = pytree.Node(syms.power, args)
        if (not (special or isview)):
            new.prefix = ''
            new = Call(Name(('iter' if isiter else 'list')), [new])
        if tail:
            new = pytree.Node(syms.power, ([new] + tail))
        new.prefix = node.prefix
        return new

    P1: str = "power< func=NAME trailer< '(' node=any ')' > any* >"
    p1 = patcomp.compile_pattern(P1)
    P2: str = "for_stmt< 'for' any 'in' node=any ':' any* >\n            | comp_for< 'for' any 'in' node=any any* >\n         "
    p2 = patcomp.compile_pattern(P2)

    def in_special_context(self, node: pytree.Node, isiter: bool) -> bool:
        if (node.parent is None):
            return False
        results: dict = {}
        if ((node.parent.parent is not None) and self.p1.match(node.parent.parent, results) and (results['node'] is node)):
            if isiter:
                return (results['func'].value in iter_exempt)
            else:
                return (results['func'].value in fixer_util.consuming_calls)
        if (not isiter):
            return False
        return (self.p2.match(node.parent, results) and (results['node'] is node))
