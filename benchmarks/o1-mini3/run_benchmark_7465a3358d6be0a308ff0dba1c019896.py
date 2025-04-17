from typing import Union
import decimal
import pyperf

def factorial(n: Union[int, decimal.Decimal], m: Union[int, decimal.Decimal]) -> Union[int, decimal.Decimal]:
    if n > m:
        return factorial(m, n)
    elif m == 0:
        return 1
    elif n == m:
        return n
    else:
        mid = (n + m) // 2
        return factorial(n, mid) * factorial(mid + 1, m)

def bench_decimal_factorial() -> None:
    c = decimal.getcontext()
    c.prec = decimal.MAX_PREC
    c.Emax = decimal.MAX_EMAX
    c.Emin = decimal.MIN_EMIN
    for n in [10000, 100000]:
        _ = factorial(decimal.Decimal(n), 0)

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'decimal_factorial benchmark'
    args = runner.parse_args()
    runner.bench_func('decimal_factorial', bench_decimal_factorial)
