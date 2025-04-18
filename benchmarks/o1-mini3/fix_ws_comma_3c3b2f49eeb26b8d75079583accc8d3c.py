"""
Fixer that changes 'a ,b' into 'a, b'.

This also changes '{a :b}' into '{a: b}', but does not touch other
uses of colons.  It does not touch other uses of whitespace.
"""

from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from typing import Dict, Any

class FixWsComma(fixer_base.BaseFix):
    explicit: bool = True
    PATTERN: str = "\n    any<(not(',') any)+ ',' ((not(',') any)+ ',')* [not(',') any]>\n    "
    COMMA: pytree.Leaf = pytree.Leaf(token.COMMA, ',')
    COLON: pytree.Leaf = pytree.Leaf(token.COLON, ':')
    SEPS: tuple[pytree.Leaf, pytree.Leaf] = (COMMA, COLON)

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> pytree.Node:
        new: pytree.Node = node.clone()
        comma: bool = False
        for child in new.children:
            if child in self.SEPS:
                prefix: str = child.prefix
                if prefix.isspace() and ('\n' not in prefix):
                    child.prefix = ''
                comma = True
            else:
                if comma:
                    prefix: str = child.prefix
                    if not prefix:
                        child.prefix = ' '
                comma = False
        return new
