import pyperf
import gc
from typing import List, Optional

N_LEVELS: int = 1000

def create_recursive_containers(n_levels: int) -> List:
    current_list: List = []
    for n in range(n_levels):
        new_list: List = ([None] * n)
        for index in range(n):
            new_list[index] = current_list
        current_list = new_list
    return current_list

def benchamark_collection(loops: int, n_levels: int) -> float:
    total_time: float = 0.0
    all_cycles: List = create_recursive_containers(n_levels)
    for _ in range(loops):
        gc.collect()
        t0: float = pyperf.perf_counter()
        collected: Optional[int] = gc.collect()
        total_time += (pyperf.perf_counter() - t0)
        assert ((collected is None) or (collected == 0))
    return total_time

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'GC traversal benchmark'
    runner.bench_time_func('gc_traversal', benchamark_collection, N_LEVELS)
