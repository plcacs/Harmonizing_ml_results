from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from typing import Any, Dict

class FixNe(fixer_base.BaseFix):
    _accept_type: int = token.NOTEQUAL

    def match(self, node: pytree.Node) -> bool:
        return node.value == '<>'

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> pytree.Leaf:
        new: pytree.Leaf = pytree.Leaf(token.NOTEQUAL, '!=', prefix=node.prefix)
        return new
