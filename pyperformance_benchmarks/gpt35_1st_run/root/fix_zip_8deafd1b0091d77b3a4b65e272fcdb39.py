from typing import Any
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.fixes import fixer_util
from typing import List

class FixZip(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    power< 'zip' args=trailer< '(' [any] ')' > [trailers=trailer*]
    >
    """
    skip_on: str = 'future_builtins.zip'

    def transform(self, node: Node, results: dict[str, Any]) -> Node:
        if self.should_skip(node):
            return
        if fixer_util.in_special_context(node):
            return None
        args: Node = results['args'].clone()
        args.prefix = ''
        trailers: List[Node] = []
        if ('trailers' in results):
            trailers = [n.clone() for n in results['trailers']]
            for n in trailers:
                n.prefix = ''
        new: Node = Node(syms.power, [fixer_util.Name('zip'), args], prefix='')
        new = Node(syms.power, ([fixer_util.Name('list'), fixer_util.ArgList([new])] + trailers))
        new.prefix = node.prefix
        return new
