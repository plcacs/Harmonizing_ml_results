from typing import Any, Callable, Sequence, Tuple, TypeVar
from hypothesis.internal.conjecture.shrinking.common import Shrinker
from hypothesis.internal.conjecture.shrinking.ordering import Ordering
from hypothesis.internal.conjecture.utils import identity

T = TypeVar("T")
U = TypeVar("U")

class Collection(Shrinker):
    current: Tuple[T, ...]  # Assuming self.current is maintained as an immutable tuple

    def setup(
        self,
        *,
        ElementShrinker: Any,
        min_size: int,
        to_order: Callable[[T], U] = identity,
        from_order: Callable[[U], T] = identity,
    ) -> None:
        self.ElementShrinker = ElementShrinker
        self.to_order = to_order
        self.from_order = from_order
        self.min_size = min_size

    def make_immutable(self, value: Sequence[T]) -> Tuple[T, ...]:
        return tuple(value)

    def short_circuit(self) -> None:
        zero: T = self.from_order(0)  # type: ignore
        self.consider([zero] * self.min_size)

    def left_is_better(self, left: Sequence[T], right: Sequence[T]) -> bool:
        if len(left) < len(right):
            return True
        for v1, v2 in zip(left, right):
            if self.to_order(v1) == self.to_order(v2):
                continue
            return self.to_order(v1) < self.to_order(v2)
        assert list(map(self.to_order, left)) == list(map(self.to_order, right))
        return False

    def run_step(self) -> None:
        zero: T = self.from_order(0)  # type: ignore
        self.consider([zero] * len(self.current))
        for i in reversed(range(len(self.current))):
            self.consider(self.current[:i] + self.current[i + 1 :])
        Ordering.shrink(self.current, self.consider, key=self.to_order)
        for i, val in enumerate(self.current):
            self.ElementShrinker.shrink(
                self.to_order(val),
                lambda v: self.consider(
                    self.current[:i] + (self.from_order(v),) + self.current[i + 1 :]
                ),
            )