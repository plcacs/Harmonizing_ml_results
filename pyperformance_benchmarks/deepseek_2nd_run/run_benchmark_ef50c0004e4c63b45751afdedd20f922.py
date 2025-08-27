"""
Calculating some of the digits of π.

This benchmark stresses big integer arithmetic.

Adapted from code on:
http://benchmarksgame.alioth.debian.org/
"""
import itertools
import pyperf
from typing import Iterator, List, Tuple, Generator, Callable, Any, Optional

DEFAULT_DIGITS: int = 2000
icount: Callable[[int], Iterator[int]] = itertools.count
islice: Callable[[Iterator[Any], int], Iterator[Any]] = itertools.islice

def gen_x() -> Iterator[Tuple[int, int, int, int]]:
    return map((lambda k: (k, ((4 * k) + 2), 0, ((2 * k) + 1))), icount(1))

def compose(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    (aq, ar, as_, at) = a
    (bq, br, bs, bt) = b
    return ((aq * bq), ((aq * br) + (ar * bt)), ((as_ * bq) + (at * bs)), ((as_ * br) + (at * bt)))

def extract(z: Tuple[int, int, int, int], j: int) -> int:
    (q, r, s, t) = z
    return (((q * j) + r) // ((s * j) + t))

def gen_pi_digits() -> Generator[int, None, None]:
    z: Tuple[int, int, int, int] = (1, 0, 0, 1)
    x: Iterator[Tuple[int, int, int, int]] = gen_x()
    while True:
        y: int = extract(z, 3)
        while y != extract(z, 4):
            z = compose(z, next(x))
            y = extract(z, 3)
        z = compose((10, (-10) * y, 0, 1), z)
        yield y

def calc_ndigits(n: int) -> List[int]:
    return list(islice(gen_pi_digits(), n))

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    cmd.extend(('--digits', str(args.digits)))

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    cmd: Any = runner.argparser
    cmd.add_argument('--digits', type=int, default=DEFAULT_DIGITS, help=('Number of computed pi digits (default: %s)' % DEFAULT_DIGITS))
    args: Any = runner.parse_args()
    runner.metadata['description'] = 'Compute digits of pi.'
    runner.metadata['pidigits_ndigit'] = args.digits
    runner.bench_func('pidigits', calc_ndigits, args.digits)
