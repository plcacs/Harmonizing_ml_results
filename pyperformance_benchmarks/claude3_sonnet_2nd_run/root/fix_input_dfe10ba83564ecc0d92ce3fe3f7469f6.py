'Fixer that changes input(...) into eval(input(...)).'
from .. import fixer_base
from ..fixer_util import Call, Name
from .. import patcomp
from typing import Optional, Any
import lib2to3.pytree

context = patcomp.compile_pattern("power< 'eval' trailer< '(' any ')' > >")

class FixInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'input' args=trailer< '(' [any] ')' > >\n              "

    def transform(self, node: lib2to3.pytree.Node, results: dict[str, Any]) -> Optional[lib2to3.pytree.Node]:
        if context.match(node.parent.parent):
            return None
        new = node.clone()
        new.prefix = ''
        return Call(Name('eval'), [new], prefix=node.prefix)
