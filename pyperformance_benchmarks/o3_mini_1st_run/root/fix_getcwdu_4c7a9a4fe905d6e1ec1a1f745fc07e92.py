from typing import Any, Dict
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import Name

class FixGetcwdu(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'os' trailer< dot='.' name='getcwdu' > any* >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        name = results['name']
        name.replace(Name('getcwd', prefix=name.prefix))
