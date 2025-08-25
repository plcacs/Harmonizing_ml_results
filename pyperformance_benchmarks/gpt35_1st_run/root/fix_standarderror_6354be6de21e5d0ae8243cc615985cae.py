from .. import fixer_base
from ..fixer_util import Name
from typing import Any

class FixStandarderror(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              'StandardError'\n              "

    def transform(self, node: Any, results: Any) -> Any:
        return Name('Exception', prefix=node.prefix)
