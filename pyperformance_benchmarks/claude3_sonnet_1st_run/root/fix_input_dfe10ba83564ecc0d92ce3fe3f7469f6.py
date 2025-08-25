'Fixer that changes input(...) into eval(input(...)).'
from .. import fixer_base
from ..fixer_util import Call, Name
from .. import patcomp
from typing import Optional, Any
import pytree

context = patcomp.compile_pattern("power< 'eval' trailer< '(' any ')' > >")

class FixInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'input' args=trailer< '(' [any] ')' > >\n              "

    def transform(self, node: pytree.Node, results: dict[str, Any]) -> Optional[pytree.Node]:
        if context.match(node.parent.parent):
            return None
        new = node.clone()
        new.prefix = ''
        return Call(Name('eval'), [new], prefix=node.prefix)
