#!/usr/bin/env python3
"""
Render a template using Genshi module.
"""
import pyperf
from genshi.template import MarkupTemplate, NewTextTemplate
from typing import Any, Callable, Dict, List, Tuple
import argparse

BIGTABLE_XML: str = (
    '<table xmlns:py="http://genshi.edgewall.org/">\n'
    '<tr py:for="row in table">\n'
    '<td py:for="c in row.values()" py:content="c"/>\n'
    '</tr>\n'
    '</table>\n'
)
BIGTABLE_TEXT: str = (
    '<table>\n'
    '{% for row in table %}<tr>\n'
    '{% for c in row.values() %}<td>$c</td>{% end %}\n'
    '</tr>{% end %}\n'
    '</table>\n'
)

def bench_genshi(loops: int, tmpl_cls: Callable[[str], Any], tmpl_str: str) -> float:
    tmpl = tmpl_cls(tmpl_str)
    table: List[Dict[str, int]] = [
        {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10}
        for _ in range(1000)
    ]
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        stream = tmpl.generate(table=table)
        stream.render()
    return pyperf.perf_counter() - t0

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.benchmark:
        cmd.append(args.benchmark)

BENCHMARKS: Dict[str, Tuple[Callable[[str], Any], str]] = {
    'xml': (MarkupTemplate, BIGTABLE_XML),
    'text': (NewTextTemplate, BIGTABLE_TEXT)
}

if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Render a template using Genshi module'
    runner.argparser.add_argument('benchmark', nargs='?', choices=sorted(BENCHMARKS))
    args: argparse.Namespace = runner.parse_args()
    if args.benchmark:
        benchmarks = (args.benchmark,)
    else:
        benchmarks = sorted(BENCHMARKS)
    for bench in benchmarks:
        name: str = ('genshi_%s' % bench)
        tmpl_cls, tmpl_str = BENCHMARKS[bench]
        runner.bench_time_func(name, bench_genshi, tmpl_cls, tmpl_str)