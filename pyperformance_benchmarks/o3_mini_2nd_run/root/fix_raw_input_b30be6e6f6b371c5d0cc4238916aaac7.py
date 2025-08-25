from typing import Dict
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import Name

class FixRawInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "\n              power< name='raw_input' trailer< '(' [any] ')' > any* >\n              "
    )

    def transform(self, node: Node, results: Dict[str, Node]) -> None:
        name: Node = results['name']
        name.replace(Name('input', prefix=name.prefix))
