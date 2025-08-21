from __future__ import annotations

from typing import Any, Callable, Generic, Hashable, Optional, Set, Type, TypeVar


V = TypeVar("V", bound=Hashable)
U = TypeVar("U", bound=Hashable)


"""This module implements various useful common functions for shrinking tasks."""


class Shrinker(Generic[V]):
    """A Shrinker object manages a single value and a predicate it should
    satisfy, and attempts to improve it in some direction, making it smaller
    and simpler."""

    def __init__(
        self,
        initial: Any,
        predicate: Callable[[V], bool],
        *,
        full: bool = False,
        debug: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.setup(**kwargs)
        self.current: V = self.make_immutable(initial)
        self.initial: V = self.current
        self.full: bool = full
        self.changes: int = 0
        self.name: Optional[str] = name
        self.__predicate: Callable[[V], bool] = predicate
        self.__seen: Set[V] = set()
        self.debugging_enabled: bool = debug

    @property
    def calls(self) -> int:
        return len(self.__seen)

    def __repr__(self) -> str:
        return '{}({}initial={!r}, current={!r})'.format(
            type(self).__name__,
            '' if self.name is None else f'{self.name!r}, ',
            self.initial,
            self.current,
        )

    def setup(self, **kwargs: Any) -> None:
        """Runs initial setup code.

        Convenience function for children that doesn't require messing
        with the signature of init.
        """

    def delegate(
        self,
        other_class: Type[Shrinker[U]],
        convert_to: Callable[[V], Any],
        convert_from: Callable[[U], Any],
        **kwargs: Any,
    ) -> None:
        """Delegates shrinking to another shrinker class, by converting the
        current value to and from it with provided functions."""
        self.call_shrinker(
            other_class,
            convert_to(self.current),
            lambda v: self.consider(convert_from(v)),
            **kwargs,
        )

    def call_shrinker(
        self,
        other_class: Type[Shrinker[U]],
        initial: Any,
        predicate: Callable[[U], bool],
        **kwargs: Any,
    ) -> U:
        """Calls another shrinker class, passing through the relevant context
        variables.

        Note we explicitly do not pass through full.
        """
        return other_class.shrink(initial, predicate, **kwargs)

    def debug(self, *args: object) -> None:
        if self.debugging_enabled:
            print('DEBUG', self, *args)

    @classmethod
    def shrink(
        cls: Type[Shrinker[V]],
        initial: Any,
        predicate: Callable[[V], bool],
        **kwargs: Any,
    ) -> V:
        """Shrink the value ``initial`` subject to the constraint that it
        satisfies ``predicate``.

        Returns the shrunk value.
        """
        shrinker: Shrinker[V] = cls(initial, predicate, **kwargs)
        shrinker.run()
        return shrinker.current

    def run(self) -> None:
        """Run for an appropriate number of steps to improve the current value.

        If self.full is True, will run until no further improvements can
        be found.
        """
        if self.short_circuit():
            return
        if self.full:
            prev = -1
            while self.changes != prev:
                prev = self.changes
                self.run_step()
        else:
            self.run_step()
        self.debug('COMPLETE')

    def incorporate(self, value: Any) -> bool:
        """Try using ``value`` as a possible candidate improvement.

        Return True if it works.
        """
        v: V = self.make_immutable(value)
        self.check_invariants(v)
        if not self.left_is_better(v, self.current):
            if v != self.current and v == v:
                self.debug(f'Rejected {v!r} as worse than self.current={self.current!r}')
            return False
        if v in self.__seen:
            return False
        self.__seen.add(v)
        if self.__predicate(v):
            self.debug(f'shrinking to {v!r}')
            self.changes += 1
            self.current = v
            return True
        return False

    def consider(self, value: Any) -> bool:
        """Returns True if make_immutable(value) == self.current after calling
        self.incorporate(value)."""
        self.debug(f'considering {value}')
        v: V = self.make_immutable(value)
        if v == self.current:
            return True
        return self.incorporate(v)

    def make_immutable(self, value: Any) -> V:
        """Convert value into an immutable (and hashable) representation of
        itself.

        It is these immutable versions that the shrinker will work on.

        Defaults to just returning the value.
        """
        return value  # type: ignore[return-value]

    def check_invariants(self, value: V) -> None:
        """Make appropriate assertions about the value to ensure that it is
        valid for this shrinker.

        Does nothing by default.
        """

    def short_circuit(self) -> bool:
        """Possibly attempt to do some shrinking.

        If this returns True, the ``run`` method will terminate early
        without doing any more work.
        """
        return False

    def left_is_better(self, left: V, right: V) -> bool:
        """Returns True if the left is strictly simpler than the right
        according to the standards of this shrinker."""
        raise NotImplementedError

    def run_step(self) -> None:
        """Run a single step of the main shrink loop, attempting to improve the
        current value."""
        raise NotImplementedError