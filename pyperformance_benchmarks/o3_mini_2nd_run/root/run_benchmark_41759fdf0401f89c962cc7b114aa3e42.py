import pyperf
from typing import List

DEFAULT_ARG: int = 9

def fannkuch(n: int) -> int:
    count: List[int] = list(range(1, n + 1))
    max_flips: int = 0
    m: int = n - 1
    r: int = n
    perm1: List[int] = list(range(n))
    perm: List[int] = list(range(n))
    perm1_ins = perm1.insert
    perm1_pop = perm1.pop
    while True:
        while r != 1:
            count[r - 1] = r
            r -= 1
        if (perm1[0] != 0) and (perm1[m] != m):
            perm = perm1[:]
            flips_count: int = 0
            k: int = perm[0]
            while k:
                perm[: k + 1] = perm[k::-1]
                flips_count += 1
                k = perm[0]
            if flips_count > max_flips:
                max_flips = flips_count
        while r != n:
            perm1_ins(r, perm1_pop(0))
            count[r] -= 1
            if count[r] > 0:
                break
            r += 1
        else:
            return max_flips

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    arg: int = DEFAULT_ARG
    runner.bench_func('fannkuch', fannkuch, arg)