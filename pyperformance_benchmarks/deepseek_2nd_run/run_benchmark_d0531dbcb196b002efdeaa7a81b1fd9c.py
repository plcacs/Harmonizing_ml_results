import functools
import pyperf
from chameleon import PageTemplate
from typing import Dict, List, Any

BIGTABLE_ZPT = '<table xmlns="http://www.w3.org/1999/xhtml"\nxmlns:tal="http://xml.zope.org/namespaces/tal">\n<tr tal:repeat="row python: options[\'table\']">\n<td tal:repeat="c python: row.values()">\n<span tal:define="d python: c + 1"\ntal:attributes="class python: \'column-\' + str(d)"\ntal:content="python: d" />\n</td>\n</tr>\n</table>'

def main() -> None:
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Chameleon template'
    tmpl: PageTemplate = PageTemplate(BIGTABLE_ZPT)
    table: List[Dict[str, int]] = [dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10) for x in range(500)]
    options: Dict[str, Any] = {'table': table}
    func = functools.partial(tmpl, options=options)
    runner.bench_func('chameleon', func)

if (__name__ == '__main__'):
    main()
