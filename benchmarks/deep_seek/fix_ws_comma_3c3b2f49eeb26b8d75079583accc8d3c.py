from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from typing import Any, Set, Tuple, Union

class FixWsComma(fixer_base.BaseFix):
    explicit: bool = True
    PATTERN: str = "\n    any<(not(',') any)+ ',' ((not(',') any)+ ',')* [not(',') any]>\n    "
    COMMA: pytree.Leaf = pytree.Leaf(token.COMMA, ',')
    COLON: pytree.Leaf = pytree.Leaf(token.COLON, ':')
    SEPS: Tuple[pytree.Leaf, pytree.Leaf] = (COMMA, COLON)

    def transform(self, node: pytree.Base, results: Any) -> pytree.Base:
        new: pytree.Base = node.clone()
        comma: bool = False
        for child in new.children:
            if (child in self.SEPS):
                prefix: str = child.prefix
                if (prefix.isspace() and ('\n' not in prefix)):
                    child.prefix = ''
                comma = True
            else:
                if comma:
                    prefix = child.prefix
                    if (not prefix):
                        child.prefix = ' '
                comma = False
        return new
