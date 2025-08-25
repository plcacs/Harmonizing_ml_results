'Fixer for StandardError -> Exception.'
from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node

class FixStandarderror(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              'StandardError'\n              "

    def transform(self, node: Node, results: Dict[str, Any]) -> Name:
        return Name('Exception', prefix=node.prefix)
