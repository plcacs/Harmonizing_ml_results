from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Optional, Dict
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node

class FixGetcwdu(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "power< 'os' trailer< dot='.' name='getcwdu' > any* >"
    )

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        name = results['name']
        name.replace(Name('getcwd', prefix=name.prefix))
