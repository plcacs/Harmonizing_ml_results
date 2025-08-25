from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict
from lib2to3.pytree import Node

class FixBuffer(fixer_base.BaseFix):
    BM_compatible: bool = True
    explicit: bool = True
    PATTERN: str = "\n              power< name='buffer' trailer< '(' [any] ')' > any* >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        name = results['name']
        name.replace(Name('memoryview', prefix=name.prefix))
