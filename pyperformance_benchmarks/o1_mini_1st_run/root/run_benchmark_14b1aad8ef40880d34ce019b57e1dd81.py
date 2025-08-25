import pyperf
from typing import Callable, List, Tuple

DEFAULT_N: int = 130

def eval_A(i: int, j: int) -> float:
    return (1.0 / (((((i + j) * ((i + j) + 1)) // 2) + i) + 1))

def eval_times_u(func: Callable[[Tuple[int, List[float]]], float], u: List[float]) -> List[float]:
    return [func((i, u)) for i in range(len(u))]

def eval_AtA_times_u(u: List[float]) -> List[float]:
    return eval_times_u(part_At_times_u, eval_times_u(part_A_times_u, u))

def part_A_times_u(i_u: Tuple[int, List[float]]) -> float:
    (i, u) = i_u
    partial_sum: float = 0.0
    for (j, u_j) in enumerate(u):
        partial_sum += (eval_A(i, j) * u_j)
    return partial_sum

def part_At_times_u(i_u: Tuple[int, List[float]]) -> float:
    (i, u) = i_u
    partial_sum: float = 0.0
    for (j, u_j) in enumerate(u):
        partial_sum += (eval_A(j, i) * u_j)
    return partial_sum

def bench_spectral_norm(loops: int) -> float:
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        u: List[float] = [1.0] * DEFAULT_N
        for dummy in range(10):
            v: List[float] = eval_AtA_times_u(u)
            u = eval_AtA_times_u(v)
        vBv: float = 0.0
        vv: float = 0.0
        for (ue, ve) in zip(u, v):
            vBv += (ue * ve)
            vv += (ve * ve)
    return pyperf.perf_counter() - t0

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'MathWorld: "Hundred-Dollar, Hundred-Digit Challenge Problems", Challenge #3.'
    runner.bench_time_func('spectral_norm', bench_spectral_norm)
