from typing import Dict, Any
from .. import fixer_base
from ..fixer_util import Attr, Call, Name, Number, Subscript, Node, syms
from lib2to3.pytree import Node as PytreeNode

class FixSysExc(fixer_base.BaseFix):
    exc_info: list[str] = ['exc_type', 'exc_value', 'exc_traceback']
    BM_compatible: bool = True
    PATTERN: str = (
        "\n              power< 'sys' trailer< dot='.' attribute=(%s) > >\n              "
        % '|'.join((("'%s'" % e) for e in exc_info))
    )

    def transform(self, node: PytreeNode, results: Dict[str, Any]) -> PytreeNode:
        sys_attr = results['attribute'][0]
        index = Number(self.exc_info.index(sys_attr.value))
        call = Call(Name('exc_info'), prefix=sys_attr.prefix)
        attr = Attr(Name('sys'), call)
        attr[1].children[0].prefix = results['dot'].prefix
        attr.append(Subscript(index))
        return Node(syms.power, attr, prefix=node.prefix)
