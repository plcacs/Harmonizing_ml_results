"""
SQLite benchmark.

The goal of the benchmark is to test CFFI performance and going back and forth
between SQLite and Python a lot. Therefore the queries themselves are really
simple.
"""
import sqlite3
import math
import pyperf
from typing import Optional, Any, Tuple


class AvgLength(object):

    def __init__(self) -> None:
        self.sum: int = 0
        self.count: int = 0

    def step(self, x: Optional[str]) -> None:
        if x is not None:
            self.count += 1
            self.sum += len(x)

    def finalize(self) -> float:
        return self.sum / float(self.count)


def bench_sqlite(loops: int) -> float:
    t0: float = pyperf.perf_counter()
    conn: sqlite3.Connection = sqlite3.connect(':memory:')
    conn.execute('create table cos (x, y, z);')
    for i in range(loops):
        cos_i: float = math.cos(i)
        conn.execute('insert into cos values (?, ?, ?)', [i, cos_i, str(i)])
    conn.create_function('cos', 1, math.cos)
    for (x, cosx1, cosx2) in conn.execute('select x, cos(x), y from cos'):
        assert (math.cos(x) == cosx1 == cosx2)
    conn.create_aggregate('avglength', 1, AvgLength)
    cursor: sqlite3.Cursor = conn.execute('select avglength(z) from cos;')
    _result: Any = cursor.fetchone()[0]
    conn.execute('delete from cos;')
    conn.close()
    return pyperf.perf_counter() - t0


if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark Python aggregate for SQLite'
    runner.bench_time_func('sqlite_synth', bench_sqlite)
