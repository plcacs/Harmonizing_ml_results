"Fixer for has_key().\n\nCalls to .has_key() methods are expressed in terms of the 'in'\noperator:\n\n    d.has_key(k) -> k in d\n\nCAVEATS:\n1) While the primary target of this fixer is dict.has_key(), the\n   fixer will change any has_key() method call, regardless of its\n   class.\n\n2) Cases like this will not be converted:\n\n    m = d.has_key\n    if m(k):\n        ...\n\n   Only *calls* to has_key() are converted. While it is possible to\n   convert the above to something like\n\n    m = d.__contains__\n    if m(k):\n        ...\n\n   this is currently not done.\n"
from .. import pytree
from .. import fixer_base
from ..fixer_util import Name, parenthesize
from typing import Dict, Any, Optional, List, Union

class FixHasKey(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    anchor=power<\n        before=any+\n        trailer< '.' 'has_key' >\n        trailer<\n            '('\n            ( not(arglist | argument<any '=' any>) arg=any\n            | arglist<(not argument<any '=' any>) arg=any ','>\n            )\n            ')'\n        >\n        after=any*\n    >\n    |\n    negation=not_test<\n        'not'\n        anchor=power<\n            before=any+\n            trailer< '.' 'has_key' >\n            trailer<\n                '('\n                ( not(arglist | argument<any '=' any>) arg=any\n                | arglist<(not argument<any '=' any>) arg=any ','>\n                )\n                ')'\n            >\n        >\n    >\n    "

    def transform(self, node: pytree.Base, results: Dict[str, Any]) -> Optional[pytree.Base]:
        assert results
        syms = self.syms
        if ((node.parent.type == syms.not_test) and self.pattern.match(node.parent)):
            return None
        negation: Optional[pytree.Base] = results.get('negation')
        anchor: pytree.Base = results['anchor']
        prefix: str = node.prefix
        before: List[pytree.Base] = [n.clone() for n in results['before']]
        arg: pytree.Base = results['arg'].clone()
        after: Optional[List[pytree.Base]] = results.get('after')
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
