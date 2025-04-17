from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node

class FixRawInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< name='raw_input' trailer< '(' [any] ')' > any* >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        name: Node = results['name']
        name.replace(Name('input', prefix=name.prefix))
        return node
