from typing import Any, Dict, List, Optional, Sequence, Union
from .. import pytree
from .. import fixer_base
from ..fixer_util import Name, parenthesize

class FixHasKey(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    anchor=power<\n        before=any+\n        trailer< '.' 'has_key' >\n        trailer<\n            '('\n            ( not(arglist | argument<any '=' any>) arg=any\n            | arglist<(not argument<any '=' any>) arg=any ','>\n            )\n            ')'\n        >\n        after=any*\n    >\n    |\n    negation=not_test<\n        'not'\n        anchor=power<\n            before=any+\n            trailer< '.' 'has_key' >\n            trailer<\n                '('\n                ( not(arglist | argument<any '=' any>) arg=any\n                | arglist<(not argument<any '=' any>) arg=any ','>\n                )\n                ')'\n            >\n        >\n    >\n    "

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        assert results
        syms = self.syms
        if ((node.parent.type == syms.not_test) and self.pattern.match(node.parent)):
            return None
        negation: Optional[pytree.Node] = results.get('negation')
        anchor: pytree.Node = results['anchor']
        prefix: str = node.prefix
        before: List[pytree.Node] = [n.clone() for n in results['before']]
        arg: pytree.Node = results['arg'].clone()
        after: Optional[List[pytree.Node]] = results.get('after')
        if after:
            after = [n.clone() for n in after]
        if (arg.type in (syms.comparison, syms.not_test, syms.and_test, syms.or_test, syms.test, syms.lambdef, syms.argument)):
            arg = parenthesize(arg)
        if (len(before) == 1):
            before = before[0]
        else:
            before = pytree.Node(syms.power, before)
        before.prefix = ' '
        n_op: Union[Name, pytree.Node] = Name('in', prefix=' ')
        if negation:
            n_not: Name = Name('not', prefix=' ')
            n_op = pytree.Node(syms.comp_op, (n_not, n_op))
        new: pytree.Node = pytree.Node(syms.comparison, (arg, n_op, before))
        if after:
            new = parenthesize(new)
            new = pytree.Node(syms.power, ((new,) + tuple(after)))
        if (node.parent.type in (syms.comparison, syms.expr, syms.xor_expr, syms.and_expr, syms.shift_expr, syms.arith_expr, syms.term, syms.factor, syms.power)):
            new = parenthesize(new)
        new.prefix = prefix
        return new
