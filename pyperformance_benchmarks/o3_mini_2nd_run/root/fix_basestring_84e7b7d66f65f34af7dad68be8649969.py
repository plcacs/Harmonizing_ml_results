from typing import Dict, Any
from lib2to3.pytree import Node

'Fixer for basestring -> str.'
from .. import fixer_base
from ..fixer_util import Name

class FixBasestring(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'basestring'"

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        return Name('str', prefix=node.prefix)