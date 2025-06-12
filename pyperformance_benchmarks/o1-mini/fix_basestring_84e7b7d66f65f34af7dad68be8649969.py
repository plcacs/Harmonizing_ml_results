from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict
from lib2to3.pytree import Node

class FixBasestring(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'basestring'"

    def transform(self, node: Node, results: Dict[str, Any]) -> Name:
        return Name('str', prefix=node.prefix)
