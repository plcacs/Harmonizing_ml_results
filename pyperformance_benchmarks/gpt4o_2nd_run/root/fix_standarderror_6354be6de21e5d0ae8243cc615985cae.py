'Fixer for StandardError -> Exception.'
from .. import fixer_base
from ..fixer_util import Name
from lib2to3.pytree import Node
from typing import Dict, Any

class FixStandarderror(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              'StandardError'\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        return Name('Exception', prefix=node.prefix)
