from .. import fixer_base
from ..fixer_util import Name
from typing import Any

class FixXreadlines(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< call: Any + trailer< '.' 'xreadlines' > trailer< '(' ')' > >\n    |\n    power< Any + trailer< '.' no_call: str = 'xreadlines' > >\n    "

    def transform(self, node: Any, results: Any) -> None:
        no_call: Any = results.get('no_call')
        if no_call:
            no_call.replace(Name('__iter__', prefix=no_call.prefix))
        else:
            node.replace([x.clone() for x in results['call']])
