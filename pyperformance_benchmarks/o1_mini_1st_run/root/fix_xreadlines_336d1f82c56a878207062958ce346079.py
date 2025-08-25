from .. import fixer_base
from ..fixer_util import Name
from typing import Optional, Any, Dict
from lib2to3.pytree import Node

class FixXreadlines(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "\n    power< call=any+ trailer< '.' 'xreadlines' > trailer< '(' ')' > >\n"
        "    |\n"
        "    power< any+ trailer< '.' no_call='xreadlines' > >\n"
    )

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        no_call = results.get('no_call')
        if no_call:
            return no_call.replace(Name('__iter__', prefix=no_call.prefix))
        else:
            return node.replace([x.clone() for x in results['call']])
