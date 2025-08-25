from .. import fixer_base
from ..fixer_util import Call, Name
from .. import patcomp
from typing import Any, Dict, Optional
from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Node

context = patcomp.compile_pattern("power< 'eval' trailer< '(' any ')' > >")

class FixInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'input' args=trailer< '(' [any] ')' > >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        if context.match(node.parent.parent):
            return None
        new = node.clone()
        new.prefix = ''
        return Call(Name('eval'), [new], prefix=node.prefix)
