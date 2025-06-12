'Fixer that turns <> into !=.'
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from lib2to3.pytree import Node
from typing import Optional

class FixNe(fixer_base.BaseFix):
    _accept_type: int = token.NOTEQUAL

    def match(self, node: Node) -> bool:
        return (node.value == '<>')

    def transform(self, node: Node, results: dict) -> Optional[pytree.Leaf]:
        new = pytree.Leaf(token.NOTEQUAL, '!=', prefix=node.prefix)
        return new
