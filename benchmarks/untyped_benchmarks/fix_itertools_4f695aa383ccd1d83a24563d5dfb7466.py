
' Fixer for itertools.(imap|ifilter|izip) --> (map|filter|zip) and\n    itertools.ifilterfalse --> itertools.filterfalse (bugs 2360-2363)\n\n    imports from itertools are fixed in fix_itertools_import.py\n\n    If itertools is imported as something else (ie: import itertools as it;\n    it.izip(spam, eggs)) method calls will not get fixed.\n    '
from .. import fixer_base
from ..fixer_util import Name

class FixItertools(fixer_base.BaseFix):
    BM_compatible = True
    it_funcs = "('imap'|'ifilter'|'izip'|'izip_longest'|'ifilterfalse')"
    PATTERN = ("\n              power< it='itertools'\n                  trailer<\n                     dot='.' func=%(it_funcs)s > trailer< '(' [any] ')' > >\n              |\n              power< func=%(it_funcs)s trailer< '(' [any] ')' > >\n              " % locals())
    run_order = 6

    def transform(self, node, results):
        prefix = None
        func = results['func'][0]
        if (('it' in results) and (func.value not in ('ifilterfalse', 'izip_longest'))):
            (dot, it) = (results['dot'], results['it'])
            prefix = it.prefix
            it.remove()
            dot.remove()
            func.parent.replace(func)
        prefix = (prefix or func.prefix)
        func.replace(Name(func.value[1:], prefix=prefix))
