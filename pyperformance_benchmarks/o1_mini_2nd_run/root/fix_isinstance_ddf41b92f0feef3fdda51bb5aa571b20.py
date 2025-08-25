'Fixer that cleans up a tuple argument to isinstance after the tokens\nin it were fixed.  This is mainly used to remove double occurrences of\ntokens as a leftover of the long -> int / unicode -> str conversion.\n\neg.  isinstance(x, (int, long)) -> isinstance(x, (int, int))\n       -> isinstance(x, int)\n'
from .. import fixer_base
from ..fixer_util import token
from typing import Set, List, Any, Tuple, Iterator, Dict
from lib2to3.pytree import Node

class FixIsinstance(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
    power<
        'isinstance'
        trailer< '(' arglist< any ',' atom< '('
            args=testlist_gexp< any+ >
        ')' > > ')' >
    >
    """
    run_order: int = 6

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
        names_inserted: Set[str] = set()
        testlist = results['args']
        args: List[Any] = testlist.children
        new_args: List[Any] = []
        iterator: Iterator[Tuple[int, Any]] = enumerate(args)
        for idx, arg in iterator:
            if (arg.type == token.NAME) and (arg.value in names_inserted):
                if (idx < (len(args) - 1)) and (args[idx + 1].type == token.COMMA):
                    next(iterator, None)
                    continue
            else:
                new_args.append(arg)
                if arg.type == token.NAME:
                    names_inserted.add(arg.value)
        if new_args and (new_args[-1].type == token.COMMA):
            del new_args[-1]
        if len(new_args) == 1:
            atom: Any = testlist.parent
            new_args[0].prefix = atom.prefix
            atom.replace(new_args[0])
        else:
            args[:] = new_args
            node.changed()
