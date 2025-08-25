import copy
import pyperf
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class A:
    string: str
    lst: List[int]
    boolean: bool

def benchmark_reduce(n: int) -> float:
    ' Benchmark where the __reduce__ functionality is used '
    class C(object):
        a: int
        b: int

        def __init__(self) -> None:
            self.a = 1
            self.b = 2

        def __reduce__(self) -> Tuple[type, tuple, dict]:
            return (C, (), self.__dict__)

        def __setstate__(self, state: Dict[str, Any]) -> None:
            self.__dict__.update(state)

    c: C = C()
    t0: float = pyperf.perf_counter()
    for ii in range(n):
        _ = copy.deepcopy(c)
    dt: float = pyperf.perf_counter() - t0
    return dt

def benchmark_memo(n: int) -> float:
    ' Benchmark where the memo functionality is used '
    A_list: List[int] = [1] * 100
    data: Dict[str, Tuple[List[int], List[int], List[int]]] = {'a': (A_list, A_list, A_list), 'b': ([A_list] * 100)}
    t0: float = pyperf.perf_counter()
    for ii in range(n):
        _ = copy.deepcopy(data)
    dt: float = pyperf.perf_counter() - t0
    return dt

def benchmark(n: int) -> float:
    ' Benchmark on some standard data types '
    a: Dict[str, Any] = {'list': [1, 2, 3, 43], 't': (1, 2, 3), 'str': 'hello', 'subdict': {'a': True}}
    dc: A = A('hello', [1, 2, 3], True)
    dt: float = 0.0
    for ii in range(n):
        for jj in range(30):
            t0: float = pyperf.perf_counter()
            _ = copy.deepcopy(a)
            dt += pyperf.perf_counter() - t0
        for s in ['red', 'blue', 'green']:
            dc.string = s
            for kk in range(5):
                dc.lst[0] = kk
                for b in [True, False]:
                    dc.boolean = b
                    t0 = pyperf.perf_counter()
                    _ = copy.deepcopy(dc)
                    dt += pyperf.perf_counter() - t0
    return dt

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'deepcopy benchmark'
    runner.bench_time_func('deepcopy', benchmark)
    runner.bench_time_func('deepcopy_reduce', benchmark_reduce)
    runner.bench_time_func('deepcopy_memo', benchmark_memo)