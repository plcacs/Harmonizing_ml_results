#!/usr/bin/env python3
from typing import List, Dict, Any, Callable
import functools
import pyperf
from chameleon import PageTemplate

BIGTABLE_ZPT: str = (
    '<table xmlns="http://www.w3.org/1999/xhtml"\n'
    'xmlns:tal="http://xml.zope.org/namespaces/tal">\n'
    '<tr tal:repeat="row python: options[\'table\']">\n'
    '<td tal:repeat="c python: row.values()">\n'
    '<span tal:define="d python: c + 1"\n'
    'tal:attributes="class python: \'column-\' + str(d)"\n'
    'tal:content="python: d" />\n'
    '</td>\n'
    '</tr>\n'
    '</table>'
)

def main() -> None:
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Chameleon template'
    tmpl: PageTemplate = PageTemplate(BIGTABLE_ZPT)
    table: List[Dict[str, int]] = [
        dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10) for _ in range(500)
    ]
    options: Dict[str, List[Dict[str, int]]] = {'table': table}
    func: Callable[[], Any] = functools.partial(tmpl, options=options)
    runner.bench_func('chameleon', func)

if __name__ == '__main__':
    main()