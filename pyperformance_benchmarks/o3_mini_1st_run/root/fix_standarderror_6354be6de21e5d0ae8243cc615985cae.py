from typing import Any, Dict
from .. import fixer_base
from ..fixer_util import Name

class FixStandarderror(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              'StandardError'\n              "

    def transform(self, node: Any, results: Dict[str, Any]) -> Name:
        return Name('Exception', prefix=node.prefix)