import decimal
import pyperf
from decimal import Decimal, Context

def factorial(n: Decimal, m: Decimal) -> Decimal:
    if n > m:
        return factorial(m, n)
    elif m == Decimal(0):
        return Decimal(1)
    elif n == m:
        return n
    else:
        mid: Decimal = (n + m) // 2
        return factorial(n, mid) * factorial(mid + Decimal(1), m)

def bench_decimal_factorial() -> None:
    c: Context = decimal.getcontext()
    c.prec = decimal.MAX_PREC
    c.Emax = decimal.MAX_EMAX
    c.Emin = decimal.MIN_EMIN
    for n in [10000, 100000]:
        _ = factorial(Decimal(n), Decimal(0))

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'decimal_factorial benchmark'
    args = runner.parse_args()
    runner.bench_func('decimal_factorial', bench_decimal_factorial)