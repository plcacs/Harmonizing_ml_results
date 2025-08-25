from typing import Dict, List, Union
from .. import fixer_base
from ..fixer_util import Attr, Call, Name, Number, Subscript, Node, syms

class FixSysExc(fixer_base.BaseFix):
    exc_info: List[str] = ['exc_type', 'exc_value', 'exc_traceback']
    BM_compatible: bool = True
    PATTERN: str = ("\n              power< 'sys' trailer< dot='.' attribute=(%s) > >\n              " % '|'.join((("'%s'" % e) for e in exc_info)))

    def transform(self, node: Node, results: Dict[str, List[Node]]) -> Node:
        sys_attr: Node = results['attribute'][0]
        index: Number = Number(self.exc_info.index(sys_attr.value))
        call: Call = Call(Name('exc_info'), prefix=sys_attr.prefix)
        attr: Attr = Attr(Name('sys'), call)
        attr[1].children[0].prefix = results['dot'].prefix
        attr.append(Subscript(index))
        return Node(syms.power, attr, prefix=node.prefix)
