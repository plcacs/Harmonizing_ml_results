from .. import fixer_base
from ..fixer_util import Name
from typing import Dict

MAP: Dict[str, str] = {'im_func': '__func__', 'im_self': '__self__', 'im_class': '__self__.__class__'}

class FixMethodattrs(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< any+ trailer< '.' attr=('im_func' | 'im_self' | 'im_class') > any* >\n    "

    def transform(self, node, results) -> None:
        attr = results['attr'][0]
        new: str = MAP[attr.value]
        attr.replace(Name(new, prefix=attr.prefix))
