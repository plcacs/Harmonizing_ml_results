"""
Calculate `factorial` using the decimal module.

- 2024-06-14: Michael Droettboom copied this from
  Modules/_decimal/tests/bench.py in the CPython source and adapted to use
  pyperf.
"""
import decimal
import pyperf
from decimal import Decimal
from typing import Union

def factorial(n: Decimal, m: Decimal) -> Union[Decimal, int]:
    if n > m:
        return factorial(m, n)
    elif m == 0:
        return 1
    elif n == m:
        return n
    else:
        midpoint = (n + m) // 2
        return factorial(n, midpoint) * factorial(midpoint + 1, m)

def bench_decimal_factorial() -> None:
    c: decimal.Context = decimal.getcontext()
    c.prec = decimal.MAX_PREC
    c.Emax = decimal.MAX_EMAX
    c.Emin = decimal.MIN_EMIN
    for n in [10000, 100000]:
        _ = factorial(decimal.Decimal(n), Decimal(0))

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'decimal_factorial benchmark'
    args = runner.parse_args()
    runner.bench_func('decimal_factorial', bench_decimal_factorial)
