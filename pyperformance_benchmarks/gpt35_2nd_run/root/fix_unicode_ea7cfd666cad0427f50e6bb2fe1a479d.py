from ..pgen2 import token
from .. import fixer_base

_mapping: dict[str, str] = {'unichr': 'chr', 'unicode': 'str'}

class FixUnicode(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "STRING | 'unicode' | 'unichr'"
    unicode_literals: bool

    def start_tree(self, tree, filename) -> None:
        super(FixUnicode, self).start_tree(tree, filename)
        self.unicode_literals = ('unicode_literals' in tree.future_features)

    def transform(self, node, results) -> token:
        if (node.type == token.NAME):
            new: token = node.clone()
            new.value = _mapping[node.value]
            return new
        elif (node.type == token.STRING):
            val: str = node.value
            if ((not self.unicode_literals) and (val[0] in '\'"') and ('\\' in val)):
                val = '\\\\'.join([v.replace('\\u', '\\\\u').replace('\\U', '\\\\U') for v in val.split('\\\\')])
            if (val[0] in 'uU'):
                val = val[1:]
            if (val == node.value):
                return node
            new: token = node.clone()
            new.value = val
            return new
