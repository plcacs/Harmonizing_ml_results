from typing import Dict, Any
from .. import fixer_base
from ..fixer_util import Name

class FixItertools(fixer_base.BaseFix):
    BM_compatible: bool = True
    it_funcs: str = "('imap'|'ifilter'|'izip'|'izip_longest'|'ifilterfalse')"
    PATTERN: str = ("\n              power< it='itertools'\n                  trailer<\n                     dot='.' func=%(it_funcs)s > trailer< '(' [any] ')' > >\n              |\n              power< func=%(it_funcs)s trailer< '(' [any] ')' > >\n              " % locals())
    run_order: int = 6

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        prefix: Any = None
        func: Any = results['func'][0]
        if (('it' in results) and (func.value not in ('ifilterfalse', 'izip_longest'))):
            dot: Any = results['dot']
            it: Any = results['it']
            prefix = it.prefix
            it.remove()
            dot.remove()
            func.parent.replace(func)
        prefix = (prefix or func.prefix)
        func.replace(Name(func.value[1:], prefix=prefix))
