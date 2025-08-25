'Fixer for basestring -> str.'
from .. import fixer_base
from ..fixer_util import Name
from typing import Optional, Any

class FixBasestring(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'basestring'"

    def transform(self, node: Any, results: dict[str, Any]) -> Name:
        return Name('str', prefix=node.prefix)
