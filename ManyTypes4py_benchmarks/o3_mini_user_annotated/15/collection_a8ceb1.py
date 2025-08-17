from typing import Callable, Sequence, Tuple, List, TypeVar, Protocol, Any

T = TypeVar("T")


class ElementShrinkerProtocol(Protocol):
    def shrink(self, value: int, consider: Callable[[int], None]) -> None:
        ...


class Shrinker:
    # Placeholder for the Shrinker base class.
    current: Sequence[T]

    def consider(self, candidate: Sequence[T]) -> None:
        ...


def identity(x: T) -> T:
    return x


class Collection(Shrinker):
    ElementShrinker: ElementShrinkerProtocol
    to_order: Callable[[T], int]
    from_order: Callable[[int], T]
    min_size: int

    def setup(
        self,
        *,
        ElementShrinker: ElementShrinkerProtocol,
        min_size: int,
        to_order: Callable[[T], int] = identity,
        from_order: Callable[[int], T] = identity,
    ) -> None:
        self.ElementShrinker = ElementShrinker
        self.to_order = to_order
        self.from_order = from_order
        self.min_size = min_size

    def make_immutable(self, value: Sequence[T]) -> Tuple[T, ...]:
        return tuple(value)

    def short_circuit(self) -> None:
        zero: T = self.from_order(0)
        self.consider([zero] * self.min_size)

    def left_is_better(self, left: Sequence[T], right: Sequence[T]) -> bool:
        if len(left) < len(right):
            return True

        # examine elements one by one from the left until an element differs.
        for v1, v2 in zip(left, right):
            if self.to_order(v1) == self.to_order(v2):
                continue
            return self.to_order(v1) < self.to_order(v2)

        # equal length and all values were equal by our ordering, so must be equal
        # by our ordering.
        assert list(map(self.to_order, left)) == list(map(self.to_order, right))
        return False

    def run_step(self) -> None:
        # try all-zero first; we already considered all-zero-and-smallest in
        # short_circuit.
        zero: T = self.from_order(0)
        self.consider([zero] * len(self.current))

        # try deleting each element in turn, starting from the back
        # TODO_BETTER_SHRINK: adaptively delete here by deleting larger chunks at once
        # if early deletes succeed. use find_integer. turns O(n) into O(log(n))
        for i in reversed(range(len(self.current))):
            self.consider(self.current[:i] + self.current[i + 1 :])

        # then try reordering
        Ordering.shrink(self.current, self.consider, key=self.to_order)

        # then try minimizing each element in turn
        for i, val in enumerate(self.current):
            self.ElementShrinker.shrink(
                self.to_order(val),
                lambda v, i=i: self.consider(
                    self.current[:i] + (self.from_order(v),) + self.current[i + 1 :]
                ),
            )


class Ordering:
    @staticmethod
    def shrink(
        current: Sequence[T],
        consider: Callable[[Sequence[T]], None],
        key: Callable[[T], int],
    ) -> None:
        # Placeholder implementation for the ordering based shrinker.
        # In the real implementation this would attempt to reorder `current`
        # to produce a "smaller" list according to the ordering provided by `key`.
        pass