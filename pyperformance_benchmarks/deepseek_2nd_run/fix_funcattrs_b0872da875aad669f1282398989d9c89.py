'Fix function attribute names (f.func_x -> f.__x__).'
from .. import fixer_base
from ..fixer_util import Name
from typing import Dict, Any

class FixFuncattrs(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power< any+ trailer< '.' attr=('func_closure' | 'func_doc' | 'func_globals'\n                                  | 'func_name' | 'func_defaults' | 'func_code'\n                                  | 'func_dict') > any* >\n    "

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        attr = results['attr'][0]
        attr.replace(Name(('__%s__' % attr.value[5:]), prefix=attr.prefix))
