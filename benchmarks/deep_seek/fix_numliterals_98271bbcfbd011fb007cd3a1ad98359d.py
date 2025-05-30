from __future__ import annotations
from typing import Any
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Number

class FixNumliterals(fixer_base.BaseFix):
    _accept_type: int = token.NUMBER

    def match(self, node: Any) -> bool:
        return (node.value.startswith('0') or (node.value[-1] in 'Ll'))

    def transform(self, node: Any, results: Any) -> Number:
        val = node.value
        if val[-1] in 'Ll':
            val = val[:-1]
        elif val.startswith('0') and val.isdigit() and (len(set(val)) > 1):
            val = '0o' + val[1:]
        return Number(val, prefix=node.prefix)
