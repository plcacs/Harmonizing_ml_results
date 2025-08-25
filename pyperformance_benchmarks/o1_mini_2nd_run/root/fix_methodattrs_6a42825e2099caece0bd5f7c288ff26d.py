from typing import Dict, List, Optional

from .. import fixer_base
from ..fixer_util import Name
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node, Leaf

MAP: Dict[str, str] = {
    'im_func': '__func__',
    'im_self': '__self__',
    'im_class': '__self__.__class__'
}

class FixMethodattrs(BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
        power< any+ trailer< '.' attr=('im_func' | 'im_self' | 'im_class') > any* >
    """

    def transform(self, node: Node, results: Dict[str, List[Leaf]]) -> Optional[Node]:
        attr: Leaf = results['attr'][0]
        new: str = MAP[attr.value]
        attr.replace(Name(new, prefix=attr.prefix))
        return node
