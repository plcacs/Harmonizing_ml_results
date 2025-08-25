"""Remove __future__ imports

from __future__ import foo is replaced with an empty line.
"""

from .. import fixer_base
from ..fixer_util import BlankLine
from lib2to3.pytree import Node
from typing import Any


class FixFuture(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = 'import_from< \'from\' module_name="__future__" \'import\' any >'
    run_order: int = 10

    def transform(self, node: Node, results: Any) -> BlankLine:
        new: BlankLine = BlankLine()
        new.prefix = node.prefix
        return new
