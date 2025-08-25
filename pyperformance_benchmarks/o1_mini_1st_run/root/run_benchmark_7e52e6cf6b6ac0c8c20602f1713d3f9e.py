"""Test the performance of pprint.PrettyPrinter.

This benchmark was available as `python -m pprint` until Python 3.12.

Authors: Fred Drake (original), Oleg Iarygin (pyperformance port).
"""
import pyperf
from pprint import PrettyPrinter
from typing import List, Tuple, Dict

printable: List[Tuple[str, Tuple[int, int], List[int], Dict[int, int]]] = ([('string', (1, 2), [3, 4], {5: 6, 7: 8})] * 100000)
p: PrettyPrinter = PrettyPrinter()

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'pprint benchmark'
    if hasattr(p, '_safe_repr'):
        runner.bench_func('pprint_safe_repr', p._safe_repr, printable, {}, None, 0)
    runner.bench_func('pprint_pformat', p.pformat, printable)
