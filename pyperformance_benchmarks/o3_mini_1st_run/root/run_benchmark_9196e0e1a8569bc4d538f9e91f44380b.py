from __future__ import annotations
from collections.abc import Iterator
from typing import Generic, Optional, Sequence, TypeVar
import pyperf

T = TypeVar("T")

class Tree(Generic[T]):
    def __init__(self, left: Optional[Tree[T]], value: T, right: Optional[Tree[T]]) -> None:
        self.left: Optional[Tree[T]] = left
        self.value: T = value
        self.right: Optional[Tree[T]] = right

    def __iter__(self) -> Iterator[T]:
        if self.left:
            yield from self.left
        yield self.value
        if self.right:
            yield from self.right

def tree(input: Sequence[T]) -> Optional[Tree[T]]:
    n: int = len(input)
    if n == 0:
        return None
    i: int = n // 2
    return Tree(tree(input[:i]), input[i], tree(input[i + 1:]))

def bench_generators(loops: int) -> float:
    assert list(tree(range(10))) == list(range(10))
    range_it = range(loops)
    iterable: Optional[Tree[int]] = tree(range(100000))
    t0 = pyperf.perf_counter()
    # Since iterable could be None, safeguard using an empty iterable if necessary.
    for _ in range_it:
        if iterable is not None:
            for _ in iterable:
                pass
    return pyperf.perf_counter() - t0

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark generators'
    runner.bench_time_func('generators', bench_generators)