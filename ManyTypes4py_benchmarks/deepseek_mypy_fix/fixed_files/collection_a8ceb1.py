from typing import Any, Callable, Generic, TypeVar, Tuple, Optional, cast
from hypothesis.internal.conjecture.shrinking.common import Shrinker
from hypothesis.internal.conjecture.shrinking.ordering import Ordering
from hypothesis.internal.conjecture.utils import identity

T = TypeVar('T')
S = TypeVar('S', bound=Any)

class Collection(Shrinker, Generic[T, S]):

    def setup(
        self,
        *,
        ElementShrinker: Any,
        min_size: int,
        to_order: Callable[[T], S] = identity,
        from_order: Callable[[S], T] = identity
    ) -> None:
        self.ElementShrinker: Any = ElementShrinker
        self.to_order: Callable[[T], S] = to_order
        self.from_order: Callable[[S], T] = from_order
        self.min_size: int = min_size

    def make_immutable(self, value: Any) -> Tuple[T, ...]:
        return tuple(value)

    def short_circuit(self) -> bool:
        zero: T = self.from_order(cast(S, 0))
        return self.consider([zero] * self.min_size)

    def left_is_better(self, left: Tuple[T, ...], right: Tuple[T, ...]) -> bool:
        if len(left) < len(right):
            return True
        for v1, v2 in zip(left, right):
            o1 = self.to_order(v1)
            o2 = self.to_order(v2)
            if o1 == o2:
                continue
            return o1 < o2  # type: ignore
        assert list(map(self.to_order, left)) == list(map(self.to_order, right))
        return False

    def run_step(self) -> None:
        zero: T = self.from_order(cast(S, 0))
        self.consider([zero] * len(self.current))
        for i in reversed(range(len(self.current))):
            self.consider(self.current[:i] + self.current[i + 1:])
        Ordering.shrink(self.current, self.consider, key=self.to_order)
        for i, val in enumerate(self.current):
            self.ElementShrinker.shrink(
                self.to_order(val),
                lambda v: self.consider(
                    self.current[:i] + (self.from_order(v),) + self.current[i + 1:]
                )
            )