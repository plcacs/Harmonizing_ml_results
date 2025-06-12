"""
Calculate `pi` using the decimal module.

The `pidigits` benchmark does a similar thing using regular (long) ints.

- 2024-06-14: Michael Droettboom copied this from
  Modules/_decimal/tests/bench.py in the CPython source and adapted to use
  pyperf.
"""
import decimal
import pyperf
from decimal import Decimal


def pi_decimal() -> Decimal:
    'decimal'
    D = decimal.Decimal
    lasts: Decimal = D(0)
    t: Decimal = D(3)
    s: Decimal = D(3)
    n: Decimal = D(1)
    na: Decimal = D(0)
    d: Decimal = D(0)
    da: Decimal = D(24)
    while s != lasts:
        lasts = s
        n, na = n + na, na + D(8)
        d, da = d + da, da + D(32)
        t = (t * n) / d
        s += t
    return s


def bench_decimal_pi() -> None:
    for prec in [9, 19]:
        decimal.getcontext().prec = prec
        for _ in range(10000):
            _ = pi_decimal()


if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'decimal_pi benchmark'
    args = runner.parse_args()
    runner.bench_func('decimal_pi', bench_decimal_pi)
