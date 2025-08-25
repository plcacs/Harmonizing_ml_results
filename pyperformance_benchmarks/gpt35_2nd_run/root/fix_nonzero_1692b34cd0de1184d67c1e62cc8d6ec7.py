from .. import fixer_base
from ..fixer_util import Name
from typing import Any

class FixNonzero(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    classdef< 'class' any+ ':'\n              suite< any*\n                     funcdef< 'def' name='__nonzero__'\n                              parameters< '(' NAME: Any ')' > any+ >\n                     any* > >\n    "

    def transform(self, node: Any, results: Any) -> None:
        name: Any = results['name']
        new: Name = Name('__bool__', prefix=name.prefix)
        name.replace(new)
