'Remove __future__ imports\n\nfrom __future__ import foo is replaced with an empty line.\n'
from typing import Optional
from .. import fixer_base
from ..fixer_util import BlankLine

class FixFuture(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = 'import_from< \'from\' module_name="__future__" \'import\' any >'
    run_order: int = 10

    def transform(self, node: fixer_base.Node, results: dict) -> BlankLine:
        new: BlankLine = BlankLine()
        new.prefix = node.prefix
        return new
