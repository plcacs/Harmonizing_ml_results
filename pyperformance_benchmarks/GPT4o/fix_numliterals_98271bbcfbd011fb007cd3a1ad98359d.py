from typing import Optional
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Number
from lib2to3.pytree import Node

class FixNumliterals(fixer_base.BaseFix):
    _accept_type = token.NUMBER

    def match(self, node: Node) -> bool:
        return (node.value.startswith('0') or (node.value[(- 1)] in 'Ll'))

    def transform(self, node: Node, results: dict) -> Optional[Node]:
        val = node.value
        if (val[(- 1)] in 'Ll'):
            val = val[:(- 1)]
        elif (val.startswith('0') and val.isdigit() and (len(set(val)) > 1)):
            val = ('0o' + val[1:])
        return Number(val, prefix=node.prefix)
