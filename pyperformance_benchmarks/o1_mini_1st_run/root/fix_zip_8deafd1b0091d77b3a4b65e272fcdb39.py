from typing import Optional, Dict, Any, List
from .. import fixer_base
from ..pytree import Node
from ..pygram import python_symbols as syms
from ..fixer_util import Name, ArgList, in_special_context


class FixZip(fixer_base.ConditionalFix):
    BM_compatible: bool = True
    PATTERN: str = """
        power< 'zip' args=trailer< '(' [any] ')' > [trailers=trailer*] >
    """
    skip_on: str = 'future_builtins.zip'

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        if self.should_skip(node):
            return
        if in_special_context(node):
            return None
        args: Node = results['args'].clone()
        args.prefix = ''
        trailers: List[Node] = []
        if 'trailers' in results:
            trailers = [n.clone() for n in results['trailers']]
            for n in trailers:
                n.prefix = ''
        new = Node(syms.power, [Name('zip'), args], prefix='')
        new = Node(syms.power, [Name('list'), ArgList([new])] + trailers, prefix='')
        new.prefix = node.prefix
        return new
