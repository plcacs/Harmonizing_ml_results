"""
Fixer for itertools.(imap|ifilter|izip) --> (map|filter|zip) and
itertools.ifilterfalse --> itertools.filterfalse (bugs 2360-2363)

imports from itertools are fixed in fix_itertools_import.py

If itertools is imported as something else (ie: import itertools as it;
it.izip(spam, eggs)) method calls will not get fixed.
"""
from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict, List
from lib2to3.pytree import Node


class FixItertools(fixer_base.BaseFix):
    BM_compatible: bool = True
    it_funcs: str = "('imap'|'ifilter'|'izip'|'izip_longest'|'ifilterfalse')"
    PATTERN: str = (
        "\n              power< it='itertools'\n"
        "                  trailer<\n"
        "                     dot='.' func=%(it_funcs)s > trailer< '(' [any] ')' > >\n"
        "              |\n"
        "              power< func=%(it_funcs)s trailer< '(' [any] ')' > >\n"
        "              " % locals()
    )
    run_order: int = 6

    def transform(self, node: Node, results: Dict[str, List[Any]]) -> None:
        prefix = None
        func = results['func'][0]
        if 'it' in results and func.value not in ('ifilterfalse', 'izip_longest'):
            dot = results['dot'][0]
            it = results['it'][0]
            prefix = it.prefix
            it.remove()
            dot.remove()
            func.parent.replace(func)
        prefix = prefix or func.prefix
        func.replace(Name(func.value[1:], prefix=prefix))
