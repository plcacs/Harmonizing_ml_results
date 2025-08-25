from .. import pytree
from .. import fixer_base
from ..fixer_util import Name, parenthesize
from typing import Any, Dict, List, Optional, Union

class FixHasKey(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
        anchor=power<
            before=any+
            trailer< '.' 'has_key' >
            trailer<
                '('
                ( not(arglist | argument<any '=' any>) arg=any
                | arglist<(not argument<any '=' any>) arg=any ','>
                )
                ')'
            >
            after=any*
        >
        |
        negation=not_test<
            'not'
            anchor=power<
                before=any+
                trailer< '.' 'has_key' >
                trailer<
                    '('
                    ( not(arglist | argument<any '=' any>) arg=any
                    | arglist<(not argument<any '=' any>) arg=any ','>
                    )
                    ')'
                >
            >
        >
    """

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        assert results
        syms = self.syms
        if (node.parent.type == syms.not_test) and self.pattern.match(node.parent):
            return None
        negation = results.get('negation')
        anchor: pytree.Node = results['anchor']
        prefix: str = node.prefix
        before_nodes: List[pytree.Node] = [n.clone() for n in results['before']]
        arg: pytree.Node = results['arg'].clone()
        after_nodes: Optional[List[pytree.Node]] = results.get('after')
        if after_nodes:
            after_nodes = [n.clone() for n in after_nodes]
        if arg.type in (
            syms.comparison,
            syms.not_test,
            syms.and_test,
            syms.or_test,
            syms.test,
            syms.lambdef,
            syms.argument,
        ):
            arg = parenthesize(arg)
        if len(before_nodes) == 1:
            before: Union[pytree.Node, List[pytree.Node]] = before_nodes[0]
        else:
            before = pytree.Node(syms.power, before_nodes)
        before.prefix = ' '
        n_op: Union[Name, pytree.Node] = Name('in', prefix=' ')
        if negation:
            n_not = Name('not', prefix=' ')
            n_op = pytree.Node(syms.comp_op, (n_not, n_op))
        new = pytree.Node(syms.comparison, (arg, n_op, before))
        if after_nodes:
            new = parenthesize(new)
            new = pytree.Node(syms.power, (new,) + tuple(after_nodes))
        if node.parent.type in (
            syms.comparison,
            syms.expr,
            syms.xor_expr,
            syms.and_expr,
            syms.shift_expr,
            syms.arith_expr,
            syms.term,
            syms.factor,
            syms.power,
        ):
            new = parenthesize(new)
        new.prefix = prefix
        return new
