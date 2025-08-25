from typing import Any, Dict
from .. import fixer_base
from ..fixer_util import Name
from lib2to3.fixer_base import BaseFix
from lib2to3.pygram import python_symbols
from lib2to3.fixer_util import Node

class FixBasestring(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'basestring'"

    def transform(self, node: Node, results: Dict[str, Any]) -> Name:
        return Name('str', prefix=node.prefix)
