import pyperf
from sympy import expand, symbols, integrate, tan, summation
from sympy.core.cache import clear_cache
from typing import Tuple, Any, Callable, List

def bench_expand() -> None:
    (x, y, z) = symbols('x y z')
    expand(((((1 + x) + y) + z) ** 20))

def bench_integrate() -> Any:
    (x, y) = symbols('x y')
    f = ((1 / tan(x)) ** 10)
    return integrate(f, x)

def bench_sum() -> Any:
    (x, i) = symbols('x i')
    return summation(((x ** i) / i), (i, 1, 400))

def bench_str() -> str:
    (x, y, z) = symbols('x y z')
    return str(expand((((x + (2 * y)) + (3 * z)) ** 30)))

def bench_sympy(loops: int, func: Callable[[], Any]) -> float:
    timer = pyperf.perf_counter
    dt = 0.0
    for _ in range(loops):
        clear_cache()
        t0 = timer()
        func()
        dt += (timer() - t0)
    return dt

BENCHMARKS: Tuple[str, str, str, str] = ('expand', 'integrate', 'sum', 'str')

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    if args.benchmark:
        cmd.append(args.benchmark)

if (__name__ == '__main__'):
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'SymPy benchmark'
    runner.argparser.add_argument('benchmark', nargs='?', choices=BENCHMARKS)
    import gc
    gc.disable()
    args = runner.parse_args()
    if args.benchmark:
        benchmarks: Tuple[str, ...] = (args.benchmark,)
    else:
        benchmarks = BENCHMARKS
    for bench in benchmarks:
        name = ('sympy_%s' % bench)
        func = globals()[('bench_' + bench)]
        runner.bench_time_func(name, bench_sympy, func)
