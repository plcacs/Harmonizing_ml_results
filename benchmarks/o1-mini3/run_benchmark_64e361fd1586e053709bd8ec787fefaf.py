'Simple, brute-force N-Queens solver.'
import pyperf
from typing import Iterable, Iterator, Tuple, Optional

__author__ = 'collinwinter@google.com (Collin Winter)'

def permutations(iterable: Iterable, r: Optional[int] = None) -> Iterator[Tuple]:
    'permutations(range(3), 2) --> (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)'
    pool = tuple(iterable)
    n = len(pool)
    if r is None:
        r = n
    indices = list(range(n))
    cycles = list(range(((n - r) + 1), (n + 1)))[::-1]
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

def n_queens(queen_count: int) -> Iterator[Tuple[int, ...]]:
    'N-Queens solver.\n\n    Args:\n        queen_count: the number of queens to solve for. This is also the\n            board size.\n\n    Yields:\n        Solutions to the problem. Each yielded value is looks like\n        (3, 8, 2, 1, 4, ..., 6) where each number is the column position for the\n        queen, and the index into the tuple indicates the row.\n    '
    cols = range(queen_count)
    for vec in permutations(cols):
        if queen_count == len(set(vec[i] + i for i in cols)) == len(set(vec[i] - i for i in cols)):
            yield vec

def bench_n_queens(queen_count: int) -> None:
    list(n_queens(queen_count))

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Simple, brute-force N-Queens solver'
    queen_count = 8
    runner.bench_func('nqueens', bench_n_queens, queen_count)
