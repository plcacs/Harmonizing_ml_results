from .. import fixer_base
from ..fixer_util import Call, Name
from .. import patcomp
from typing import Any, Dict, Optional

context = patcomp.compile_pattern("power< 'eval' trailer< '(' any ')' > >")

class FixInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
              power< 'input' args=trailer< '(' [any] ')' > >
              """

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        if context.match(node.parent.parent):
            return None
        new = node.clone()
        new.prefix = ''
        return Call(Name('eval'), [new], prefix=node.prefix)
