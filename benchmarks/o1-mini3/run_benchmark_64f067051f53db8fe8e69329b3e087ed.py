"""
Benchmark ``loads()`` function of the ``tomli`` module
on a large TOML file of GitHub's real world data generated by
the ``generate_data.py`` script.

It heavily exercises string operations such as concatenation,
subscripting and iteration.

Author: Kumar Aditya
"""
from pathlib import Path
import pyperf
import tomli

DATA_FILE: Path = Path(__file__).parent / 'data' / 'tomli-bench-data.toml'

def bench_tomli_loads(loops: int) -> float:
    data: str = DATA_FILE.read_text('utf-8')
    range_it: range = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        tomli.loads(data)
    return pyperf.perf_counter() - t0

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark tomli.loads()'
    runner.bench_time_func('tomli_loads', bench_tomli_loads)
