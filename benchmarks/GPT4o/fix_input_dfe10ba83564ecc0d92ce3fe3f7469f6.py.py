'Fixer that changes input(...) into eval(input(...)).'
from .. import fixer_base
from ..fixer_util import Call, Name
from .. import patcomp
from lib2to3.pytree import Node
from typing import Optional

context = patcomp.compile_pattern("power< 'eval' trailer< '(' any ')' > >")

class FixInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'input' args=trailer< '(' [any] ')' > >\n              "

    def transform(self, node: Node, results: dict) -> Optional[Node]:
        if context.match(node.parent.parent):
            return None
        new: Node = node.clone()
        new.prefix = ''
        return Call(Name('eval'), [new], prefix=node.prefix)
