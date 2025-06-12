from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Name
from lib2to3.pytree import Node
from typing import Dict, Any

class FixRawInput(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< name='raw_input' trailer< '(' [any] ')' > any* >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        name = results['name']
        name.replace(Name('input', prefix=name.prefix))
