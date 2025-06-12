from typing import Any
from .. import fixer_base
from ..fixer_util import Name
from lib2to3.pytree import Node

class FixStandarderror(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              'StandardError'\n              "

    def transform(self, node: Node, results: Any) -> Node:
        return Name('Exception', prefix=node.prefix)
