from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from typing import Any

class FixNe(fixer_base.BaseFix):
    _accept_type: int = token.NOTEQUAL

    def match(self, node: pytree.Leaf) -> bool:
        return node.value == '<>'

    def transform(self, node: pytree.Leaf, results: Any) -> pytree.Leaf:
        new: pytree.Leaf = pytree.Leaf(token.NOTEQUAL, '!=', prefix=node.prefix)
        return new
