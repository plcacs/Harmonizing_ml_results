from typing import Dict, List
from lib2to3.pytree import Node
from .. import fixer_base
from ..fixer_util import Name

MAP: Dict[str, str] = {'im_func': '__func__', 'im_self': '__self__', 'im_class': '__self__.__class__'}

class FixMethodattrs(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< any+ trailer< '.' attr=('im_func' | 'im_self' | 'im_class') > any* >\n    "

    def transform(self, node: Node, results: Dict[str, List[Node]]) -> None:
        attr: Node = results['attr'][0]
        new: str = MAP[attr.value]
        attr.replace(Name(new, prefix=attr.prefix))