'\nBenchmark recursive async generators implemented in python\nby traversing a binary tree.\n\nAuthor: Kumar Aditya\n'
from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Optional, Sequence
import pyperf

class Tree:
    left: Optional[Tree]
    value: int
    right: Optional[Tree]

    def __init__(self, left: Optional[Tree], value: int, right: Optional[Tree]) -> None:
        self.left = left
        self.value = value
        self.right = right

    async def __aiter__(self) -> AsyncIterator[int]:
        if self.left:
            async for i in self.left:
                yield i
        yield self.value
        if self.right:
            async for i in self.right:
                yield i

def tree(input: Sequence[int]) -> Optional[Tree]:
    n = len(input)
    if n == 0:
        return None
    i = n // 2
    return Tree(tree(input[:i]), input[i], tree(input[i + 1:]))

async def bench_async_generators() -> None:
    async for _ in tree(range(100000)):
        pass

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark async generators'
    runner.bench_async_func('async_generators', bench_async_generators)
