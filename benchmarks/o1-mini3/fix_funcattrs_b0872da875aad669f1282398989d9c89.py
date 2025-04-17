from .. import fixer_base
from ..fixer_util import Name
from lib2to3.pytree import Node, Leaf
from typing import Dict, Any

class FixFuncattrs(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "\n    power< any+ trailer< '.' attr=('func_closure' | 'func_doc' | 'func_globals'\n"
        "                                  | 'func_name' | 'func_defaults' | 'func_code'\n"
        "                                  | 'func_dict') > any* >\n    "
    )

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        attr: Leaf = results['attr'][0]
        attr.replace(Name(f"__{attr.value[5:]}__", prefix=attr.prefix))
