from .. import fixer_base
from ..fixer_util import Name
from lib2to3.pytree import Node
from typing import Any, Dict

class FixGetcwdu(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'os' trailer< dot='.' name='getcwdu' > any* >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        name = results['name']
        name.replace(Name('getcwd', prefix=name.prefix))
        return node
