from typing import Any, Dict, List, Set
from lib2to3.pytree import Node, Leaf
from .. import fixer_base
from ..fixer_util import token

class FixIsinstance(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "\n    power<\n        'isinstance'\n        trailer< '(' arglist< any ',' atom< '('\n            args=testlist_gexp< any+ >\n        ')' > > ')' >\n    >\n    "
    run_order = 6

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        names_inserted: Set[str] = set()
        testlist: Node = results['args']
        args: List[Node] = testlist.children
        new_args: List[Node] = []
        iterator = enumerate(args)
        for (idx, arg) in iterator:
            if (isinstance(arg, Leaf) and (arg.type == token.NAME) and (arg.value in names_inserted)):
                if (idx < (len(args) - 1)) and (args[idx + 1].type == token.COMMA):
                    next(iterator)
                    continue
            else:
                new_args.append(arg)
                if isinstance(arg, Leaf) and (arg.type == token.NAME):
                    names_inserted.add(arg.value)
        if new_args and (new_args[-1].type == token.COMMA):
            del new_args[-1]
        if len(new_args) == 1:
            atom: Node = testlist.parent
            new_args[0].prefix = atom.prefix
            atom.replace(new_args[0])
        else:
            args[:] = new_args
            node.changed()