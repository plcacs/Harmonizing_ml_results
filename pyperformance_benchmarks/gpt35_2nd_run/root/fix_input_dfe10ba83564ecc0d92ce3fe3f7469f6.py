from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Call, Name
from lib2to3 import patcomp
from typing import Any

context: Any = patcomp.compile_pattern("power< 'eval' trailer< '(' any ')' > >")

class FixInput(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'input' args=trailer< '(' [any] ')' > >\n              "

    def transform(self, node: Any, results: Any) -> Any:
        if context.match(node.parent.parent):
            return
        new: Any = node.clone()
        new.prefix = ''
        return Call(Name('eval'), [new], prefix=node.prefix)
