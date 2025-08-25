from typing import Iterable, Tuple, Generator

def permutations(iterable: Iterable, r: int = None) -> Generator[Tuple, None, None]:
    ...

def n_queens(queen_count: int) -> Generator[Tuple[int, ...], None, None]:
    ...

def bench_n_queens(queen_count: int) -> None:
    ...
