from typing import Dict
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import Name

class FixNonzero(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "\n    classdef< 'class' any+ ':'\n              suite< any*\n                     funcdef< 'def' name='__nonzero__'\n                              parameters< '(' NAME ')' > any+ >\n                     any* > >\n    "
    )

    def transform(self, node: Node, results: Dict[str, Node]) -> None:
        name = results['name']
        new = Name('__bool__', prefix=name.prefix)
        name.replace(new)