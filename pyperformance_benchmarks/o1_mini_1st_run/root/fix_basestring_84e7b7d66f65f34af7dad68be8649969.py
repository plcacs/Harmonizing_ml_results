"""Fixer for basestring -> str."""
from typing import Any

from .. import fixer_base
from ..fixer_util import Name
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node


class FixBasestring(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "'basestring'"

    def transform(self, node: Node, results: Any) -> Name:
        return Name('str', prefix=node.prefix)
