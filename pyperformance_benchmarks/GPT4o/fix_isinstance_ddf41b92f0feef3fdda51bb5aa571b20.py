from typing import Set, List, Tuple, Iterator, Any
from .. import fixer_base
from ..fixer_util import token
from lib2to3.pytree import Node

class FixIsinstance(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    power<\n        'isinstance'\n        trailer< '(' arglist< any ',' atom< '('\n            args=testlist_gexp< any+ >\n        ')' > > ')' >\n    >\n    "
    run_order: int = 6

    def transform(self, node: Node, results: dict) -> None:
        names_inserted: Set[str] = set()
        testlist: Node = results['args']
        args: List[Node] = testlist.children
        new_args: List[Node] = []
        iterator: Iterator[Tuple[int, Node]] = enumerate(args)
        for idx, arg in iterator:
            if arg.type == token.NAME and arg.value in names_inserted:
                if idx < (len(args) - 1) and args[idx + 1].type == token.COMMA:
                    next(iterator)
                    continue
            else:
                new_args.append(arg)
                if arg.type == token.NAME:
                    names_inserted.add(arg.value)
        if new_args and new_args[-1].type == token.COMMA:
            del new_args[-1]
        if len(new_args) == 1:
            atom: Node = testlist.parent
            new_args[0].prefix = atom.prefix
            atom.replace(new_args[0])
        else:
            args[:] = new_args
            node.changed()
