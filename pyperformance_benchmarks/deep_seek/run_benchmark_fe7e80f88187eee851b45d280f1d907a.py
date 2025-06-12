"""
Calculate `pi` using the decimal module.

The `pidigits` benchmark does a similar thing using regular (long) ints.

- 2024-06-14: Michael Droettboom copied this from
  Modules/_decimal/tests/bench.py in the CPython source and adapted to use
  pyperf.
"""
import decimal
import pyperf
from typing import Tuple

def pi_decimal() -> decimal.Decimal:
    'decimal'
    D = decimal.Decimal
    lasts: D
    t: D
    s: D
    n: D
    na: D
    d: D
    da: D
    (lasts, t, s, n, na, d, da) = (D(0), D(3), D(3), D(1), D(0), D(0), D(24))
    while s != lasts:
        lasts = s
        n, na = (n + na), (na + 8)
        d, da = (d + da), (da + 32)
        t = (t * n) / d
        s += t
    return s

def bench_decimal_pi() -> None:
    for prec in [9, 19]:
        decimal.getcontext().prec = prec
        for _ in range(10000):
            _ = pi_decimal()

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'decimal_pi benchmark'
    args = runner.parse_args()
    runner.bench_func('decimal_pi', bench_decimal_pi)
