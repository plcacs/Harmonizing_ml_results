from typing import Any, Dict
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name
from lib2to3.pgen2 import token
from lib2to3.pytree import Node

class FixGetcwdu(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'os' trailer< dot='.' name='getcwdu' > any* >\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        name: Node = results['name']
        name.replace(Name('getcwd', prefix=name.prefix))
