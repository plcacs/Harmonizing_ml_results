from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from typing import Any, Dict, Optional

class FixNe(fixer_base.BaseFix):
    _accept_type: int = token.NOTEQUAL

    def match(self, node: pytree.Base) -> bool:
        return (node.value == '<>')

    def transform(self, node: pytree.Base, results: Dict[str, Any]) -> Optional[pytree.Leaf]:
        new = pytree.Leaf(token.NOTEQUAL, '!=', prefix=node.prefix)
        return new
