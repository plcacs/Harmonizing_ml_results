from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict, Optional

class FixXreadlines(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< call=any+ trailer< '.' 'xreadlines' > trailer< '(' ')' > >\n    |\n    power< any+ trailer< '.' no_call='xreadlines' > >\n    "

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        no_call: Optional[Any] = results.get('no_call')
        if no_call:
            no_call.replace(Name('__iter__', prefix=no_call.prefix))
        else:
            call: Any = results['call']
            node.replace([x.clone() for x in call])
