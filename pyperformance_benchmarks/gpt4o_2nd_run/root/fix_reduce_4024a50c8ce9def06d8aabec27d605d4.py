from lib2to3 import fixer_base
from lib2to3.fixer_util import touch_import
from lib2to3.pytree import Node
from typing import Any, Dict

class FixReduce(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    PATTERN: str = """
    power< 'reduce'
        trailer< '('
            arglist< (
                (not(argument<any '=' any>) any ','
                 not(argument<any '=' any>) any) |
                (not(argument<any '=' any>) any ','
                 not(argument<any '=' any>) any ','
                 not(argument<any '=' any>) any)
            ) >
        ')' >
    >
    """

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        touch_import('functools', 'reduce', node)
