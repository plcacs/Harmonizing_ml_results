'Fixer for basestring -> str.'
from typing import Any, Dict
from .. import fixer_base
from ..fixer_util import Name

class FixBasestring(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'basestring'"

    def transform(self, node: Any, results: Dict[str, Any]) -> Name:
        return Name('str', prefix=node.prefix)