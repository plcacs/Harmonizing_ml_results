from typing import Callable, List, Tuple, Type
from hypothesis.internal.conjecture.shrinking.common import Shrinker
from hypothesis.internal.conjecture.shrinking.ordering import Ordering
from hypothesis.internal.conjecture.utils import identity

class Collection(Shrinker):

    def setup(self, *, ElementShrinker: Type[Shrinker], min_size: int, to_order: Callable[[int], int] = identity, from_order: Callable[[int], int] = identity) -> None:
        self.ElementShrinker = ElementShrinker
        self.to_order = to_order
        self.from_order = from_order
        self.min_size = min_size

    def make_immutable(self, value: List[int]) -> Tuple[int, ...]:
        return tuple(value)

    def short_circuit(self) -> bool:
        zero = self.from_order(0)
        return self.consider([zero] * self.min_size)

    def left_is_better(self, left: Tuple[int, ...], right: Tuple[int, ...]) -> bool:
        if len(left) < len(right):
            return True
        for v1, v2 in zip(left, right):
            if self.to_order(v1) == self.to_order(v2):
                continue
            return self.to_order(v1) < self.to_order(v2)
        assert list(map(self.to_order, left)) == list(map(self.to_order, right))
        return False

    def run_step(self) -> None:
        zero = self.from_order(0)
        self.consider([zero] * len(self.current))
        for i in reversed(range(len(self.current))):
            self.consider(self.current[:i] + self.current[i + 1:])
        Ordering.shrink(self.current, self.consider, key=self.to_order)
        for i, val in enumerate(self.current):
            self.ElementShrinker.shrink(self.to_order(val), lambda v: self.consider(self.current[:i] + (self.from_order(v),) + self.current[i + 1:]))
