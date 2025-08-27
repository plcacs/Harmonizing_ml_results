'Fixer that turns <> into !=.'
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from typing import Any, Optional, Dict

class FixNe(fixer_base.BaseFix):
    _accept_type: int = token.NOTEQUAL

    def match(self, node: Any) -> bool:
        return (node.value == '<>')

    def transform(self, node: Any, results: Optional[Dict[str, Any]]) -> pytree.Leaf:
        new = pytree.Leaf(token.NOTEQUAL, '!=', prefix=node.prefix)
        return new
