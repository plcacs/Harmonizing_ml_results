import decimal
import pyperf
from typing import Any

def factorial(n: decimal.Decimal, m: decimal.Decimal) -> decimal.Decimal:
    if (n > m):
        return factorial(m, n)
    elif (m == 0):
        return decimal.Decimal(1)
    elif (n == m):
        return n
    else:
        return (factorial(n, ((n + m) // 2)) * factorial((((n + m) // 2) + 1), m))

def bench_decimal_factorial() -> None:
    c: decimal.Context = decimal.getcontext()
    c.prec = decimal.MAX_PREC
    c.Emax = decimal.MAX_EMAX
    c.Emin = decimal.MIN_EMIN
    for n in [10000, 100000]:
        _: decimal.Decimal = factorial(decimal.Decimal(n), decimal.Decimal(0))
if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'decimal_factorial benchmark'
    args: Any = runner.parse_args()
    runner.bench_func('decimal_factorial', bench_decimal_factorial)
