from .. import fixer_base
from ..fixer_util import Name
from typing import Any

class FixBasestring(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'basestring'"

    def transform(self, node: Any, results: Any) -> Any:
        return Name('str', prefix=node.prefix)
