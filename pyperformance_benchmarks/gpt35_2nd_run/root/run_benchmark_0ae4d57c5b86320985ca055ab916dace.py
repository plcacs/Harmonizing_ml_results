from typing import List, Dict, Any, Tuple
import pyperf
from genshi.template import MarkupTemplate, NewTextTemplate

BIGTABLE_XML: str = '<table xmlns:py="http://genshi.edgewall.org/">\n<tr py:for="row in table">\n<td py:for="c in row.values()" py:content="c"/>\n</tr>\n</table>\n'
BIGTABLE_TEXT: str = '<table>\n{% for row in table %}<tr>\n{% for c in row.values() %}<td>$c</td>{% end %}\n</tr>{% end %}\n</table>\n'

def bench_genshi(loops: int, tmpl_cls: type, tmpl_str: str) -> float:
    tmpl = tmpl_cls(tmpl_str)
    table: List[Dict[str, int]] = [dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10) for _ in range(1000)]
    range_it: range = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        stream = tmpl.generate(table=table)
        stream.render()
    return (pyperf.perf_counter() - t0)

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    if args.benchmark:
        cmd.append(args.benchmark)

BENCHMARKS: Dict[str, Tuple[type, str]] = {'xml': (MarkupTemplate, BIGTABLE_XML), 'text': (NewTextTemplate, BIGTABLE_TEXT)}

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Render a template using Genshi module'
    runner.argparser.add_argument('benchmark', nargs='?', choices=sorted(BENCHMARKS))
    args: Any = runner.parse_args()
    if args.benchmark:
        benchmarks: Tuple[str] = (args.benchmark,)
    else:
        benchmarks: List[str] = sorted(BENCHMARKS)
    for bench in benchmarks:
        name: str = ('genshi_%s' % bench)
        (tmpl_cls, tmpl_str) = BENCHMARKS[bench]
        runner.bench_time_func(name, bench_genshi, tmpl_cls, tmpl_str)
