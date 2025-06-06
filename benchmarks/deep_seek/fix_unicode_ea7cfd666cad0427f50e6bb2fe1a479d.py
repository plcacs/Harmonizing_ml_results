'Fixer for unicode.\n\n* Changes unicode to str and unichr to chr.\n\n* If "...\\u..." is not unicode literal change it into "...\\\\u...".\n\n* Change u"..." into "...".\n\n'
from ..pgen2 import token
from .. import fixer_base
from typing import Dict, Any, Optional, Union
from ..pytree import Leaf, Node

_mapping: Dict[str, str] = {'unichr': 'chr', 'unicode': 'str'}

class FixUnicode(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "STRING | 'unicode' | 'unichr'"
    unicode_literals: bool = False

    def start_tree(self, tree: Node, filename: str) -> None:
        super(FixUnicode, self).start_tree(tree, filename)
        self.unicode_literals = ('unicode_literals' in tree.future_features)

    def transform(self, node: Union[Leaf, Node], results: Dict[str, Any]) -> Optional[Union[Leaf, Node]]:
        if node.type == token.NAME:
            new: Leaf = node.clone()
            new.value = _mapping[node.value]
            return new
        elif node.type == token.STRING:
            val: str = node.value
            if (not self.unicode_literals) and (val[0] in '\'"') and ('\\' in val):
                val = '\\\\'.join([v.replace('\\u', '\\\\u').replace('\\U', '\\\\U') for v in val.split('\\\\')])
            if val[0] in 'uU':
                val = val[1:]
            if val == node.value:
                return node
            new = node.clone()
            new.value = val
            return new
        return None
