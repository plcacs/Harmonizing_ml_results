from typing import Iterator, Tuple, Iterable, Optional, TypeVar, List
import pyperf

T = TypeVar('T')

def permutations(iterable: Iterable[T], r: Optional[int] = None) -> Iterator[Tuple[T, ...]]:
    pool: Tuple[T, ...] = tuple(iterable)
    n: int = len(pool)
    if r is None:
        r = n
    indices: List[int] = list(range(n))
    cycles: List[int] = list(range(n - r + 1, n + 1))[::-1]
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j: int = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

def n_queens(queen_count: int) -> Iterator[Tuple[int, ...]]:
    cols = range(queen_count)
    for vec in permutations(cols):
        if queen_count == len(set(vec[i] + i for i in cols)) == len(set(vec[i] - i for i in cols)):
            yield vec

def bench_n_queens(queen_count: int) -> None:
    list(n_queens(queen_count))

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Simple, brute-force N-Queens solver'
    queen_count: int = 8
    runner.bench_func('nqueens', bench_n_queens, queen_count)