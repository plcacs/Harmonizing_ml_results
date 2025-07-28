from typing import Any, Callable, Optional, Set, Type, TypeVar

T = TypeVar("T")


class Shrinker:
    """A Shrinker object manages a single value and a predicate it should
    satisfy, and attempts to improve it in some direction, making it smaller
    and simpler."""

    def __init__(
        self,
        initial: T,
        predicate: Callable[[T], bool],
        *,
        full: bool = False,
        debug: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.setup(**kwargs)
        self.current: T = self.make_immutable(initial)
        self.initial: T = self.current
        self.full: bool = full
        self.changes: int = 0
        self.name: Optional[str] = name
        self.__predicate: Callable[[T], bool] = predicate
        self.__seen: Set[T] = set()
        self.debugging_enabled: bool = debug

    @property
    def calls(self) -> int:
        return len(self.__seen)

    def __repr__(self) -> str:
        return "{}({}initial={!r}, current={!r})".format(
            type(self).__name__,
            "" if self.name is None else f"{self.name!r}, ",
            self.initial,
            self.current,
        )

    def setup(self, **kwargs: Any) -> None:
        """Runs initial setup code.

        Convenience function for children that doesn't require messing
        with the signature of init.
        """
        pass

    def delegate(
        self,
        other_class: Type["Shrinker"],
        convert_to: Callable[[T], T],
        convert_from: Callable[[T], T],
        **kwargs: Any,
    ) -> None:
        """Delegates shrinking to another shrinker class, by converting the
        current value to and from it with provided functions."""
        self.call_shrinker(
            other_class, convert_to(self.current), lambda v: self.consider(convert_from(v)), **kwargs
        )

    def call_shrinker(
        self,
        other_class: Type["Shrinker"],
        initial: T,
        predicate: Callable[[T], bool],
        **kwargs: Any,
    ) -> T:
        """Calls another shrinker class, passing through the relevant context
        variables.

        Note we explicitly do not pass through full.
        """
        return other_class.shrink(initial, predicate, **kwargs)

    def debug(self, *args: Any) -> None:
        if self.debugging_enabled:
            print("DEBUG", self, *args)

    @classmethod
    def shrink(cls: Type["Shrinker"], initial: T, predicate: Callable[[T], bool], **kwargs: Any) -> T:
        """Shrink the value ``initial`` subject to the constraint that it
        satisfies ``predicate``.

        Returns the shrunk value.
        """
        shrinker = cls(initial, predicate, **kwargs)
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
            prev: int = -1
            while self.changes != prev:
                prev = self.changes
                self.run_step()
        else:
            self.run_step()
        self.debug("COMPLETE")

    def incorporate(self, value: T) -> bool:
        """Try using ``value`` as a possible candidate improvement.

        Return True if it works.
        """
        value = self.make_immutable(value)
        self.check_invariants(value)
        if not self.left_is_better(value, self.current):
            if value != self.current and value == value:
                self.debug(f"Rejected {value!r} as worse than self.current={self.current!r}")
            return False
        if value in self.__seen:
            return False
        self.__seen.add(value)
        if self.__predicate(value):
            self.debug(f"shrinking to {value!r}")
            self.changes += 1
            self.current = value
            return True
        return False

    def consider(self, value: T) -> bool:
        """Returns True if make_immutable(value) == self.current after calling
        self.incorporate(value).
        """
        self.debug(f"considering {value}")
        value = self.make_immutable(value)
        if value == self.current:
            return True
        return self.incorporate(value)

    def make_immutable(self, value: T) -> T:
        """Convert value into an immutable (and hashable) representation of
        itself.

        It is these immutable versions that the shrinker will work on.

        Defaults to just returning the value.
        """
        return value

    def check_invariants(self, value: T) -> None:
        """Make appropriate assertions about the value to ensure that it is
        valid for this shrinker.

        Does nothing by default.
        """
        pass

    def short_circuit(self) -> bool:
        """Possibly attempt to do some shrinking.

        If this returns True, the ``run`` method will terminate early
        without doing any more work.
        """
        return False

    def left_is_better(self, left: T, right: T) -> bool:
        """Returns True if the left is strictly simpler than the right
        according to the standards of this shrinker."""
        raise NotImplementedError

    def run_step(self) -> None:
        """Run a single step of the main shrink loop, attempting to improve the
        current value."""
        raise NotImplementedError