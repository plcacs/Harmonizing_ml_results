"""
Render a template using Genshi module.
"""
import pyperf
from genshi.template import MarkupTemplate, NewTextTemplate
from typing import Dict, Tuple, Any, List, Callable, Optional
from argparse import ArgumentParser

BIGTABLE_XML: str = '<table xmlns:py="http://genshi.edgewall.org/">\n<tr py:for="row in table">\n<td py:for="c in row.values()" py:content="c"/>\n</tr>\n</table>\n'
BIGTABLE_TEXT: str = '<table>\n{% for row in table %}<tr>\n{% for c in row.values() %}<td>$c</td>{% end %}\n</tr>{% end %}\n</table>\n'

def bench_genshi(loops: int, tmpl_cls: Any, tmpl_str: str) -> float:
    tmpl = tmpl_cls(tmpl_str)
    table: List[Dict[str, int]] = [dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10) for _ in range(1000)]
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        stream = tmpl.generate(table=table)
        stream.render()
    return pyperf.perf_counter() - t0

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    if args.benchmark:
        cmd.append(args.benchmark)

BENCHMARKS: Dict[str, Tuple[Any, str]] = {
    'xml': (MarkupTemplate, BIGTABLE_XML),
    'text': (NewTextTemplate, BIGTABLE_TEXT)
}

if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Render a template using Genshi module'
    runner.argparser.add_argument('benchmark', nargs='?', choices=sorted(BENCHMARKS))
    args = runner.parse_args()
    benchmarks: Tuple[str, ...] = (args.benchmark,) if args.benchmark else tuple(sorted(BENCHMARKS))
    for bench in benchmarks:
        name: str = f'genshi_{bench}'
        tmpl_cls, tmpl_str = BENCHMARKS[bench]
        runner.bench_time_func(name, bench_genshi, tmpl_cls, tmpl_str)
