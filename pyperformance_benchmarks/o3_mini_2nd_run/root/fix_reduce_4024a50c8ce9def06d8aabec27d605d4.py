from typing import Any, Dict
from lib2to3 import fixer_base
from lib2to3.fixer_util import touch_import
from lib2to3.pytree import Node

class FixReduce(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    PATTERN: str = (
        "\n    power< 'reduce'\n        trailer< '('\n            arglist< (\n"
        "                (not(argument<any '=' any>) any ','\n                 not(argument<any '=' any>) any) |\n"
        "                (not(argument<any '=' any>) any ','\n                 not(argument<any '=' any>) any ','\n"
        "                 not(argument<any '=' any>) any)\n"
        "            ) >\n        ')' >\n    >\n    "
    )

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        touch_import('functools', 'reduce', node)