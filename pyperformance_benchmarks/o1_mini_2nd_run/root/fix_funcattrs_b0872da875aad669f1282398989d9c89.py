from .. import fixer_base
from ..fixer_util import Name
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node
from typing import Any, Dict, List

class FixFuncattrs(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "power< any+ trailer< '.' attr=("
        "'func_closure' | 'func_doc' | 'func_globals' | "
        "'func_name' | 'func_defaults' | 'func_code' | 'func_dict') > any* >"
    )

    def transform(self, node: Node, results: Dict[str, List[Any]]) -> Node:
        attr = results['attr'][0]
        new_name = Name(f"__{attr.value[5:]}__", prefix=attr.prefix)
        attr.replace(new_name)
        return node
