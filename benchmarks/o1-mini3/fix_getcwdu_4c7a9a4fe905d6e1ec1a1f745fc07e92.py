'\nFixer that changes os.getcwdu() to os.getcwd().\n'
from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict
from lib2to3.pytree import Node

class FixGetcwdu(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "power< 'os' trailer< dot='.' name='getcwdu' > any* >"
    )

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        name = results['name']
        name.replace(Name('getcwd', prefix=name.prefix))
