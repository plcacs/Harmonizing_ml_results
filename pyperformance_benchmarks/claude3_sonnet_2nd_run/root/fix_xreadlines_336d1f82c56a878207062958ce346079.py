'Fix "for x in f.xreadlines()" -> "for x in f".\n\nThis fixer will also convert g(f.xreadlines) into g(f.__iter__).'
from .. import fixer_base
from ..fixer_util import Name
from typing import Dict, Optional, List, Any
from lib2to3.pytree import Node

class FixXreadlines(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< call=any+ trailer< '.' 'xreadlines' > trailer< '(' ')' > >\n    |\n    power< any+ trailer< '.' no_call='xreadlines' > >\n    "

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[List[Node]]:
        no_call: Optional[Node] = results.get('no_call')
        if no_call:
            no_call.replace(Name('__iter__', prefix=no_call.prefix))
        else:
            node.replace([x.clone() for x in results['call']])
        return None
