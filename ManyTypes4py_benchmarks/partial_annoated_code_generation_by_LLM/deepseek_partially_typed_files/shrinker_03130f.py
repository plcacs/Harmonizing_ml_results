import math
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Optional, Union, cast, Any, Dict, List, Set, Tuple
import attr
from hypothesis.internal.conjecture.choice import ChoiceNode, ChoiceT, choice_equal, choice_from_index, choice_key, choice_permitted, choice_to_index
from hypothesis.internal.conjecture.data import ConjectureData, ConjectureResult, Examples, Status, _Overrun, draw_choice
from hypothesis.internal.conjecture.junkdrawer import endswith, find_integer, replace_all, startswith
from hypothesis.internal.conjecture.shrinking import Bytes, Float, Integer, Ordering, String
from hypothesis.internal.conjecture.shrinking.choicetree import ChoiceTree, prefix_selection_order, random_selection_order
from hypothesis.internal.floats import MAX_PRECISE_INTEGER
if TYPE_CHECKING:
    from random import Random
    from typing import TypeAlias
    from hypothesis.internal.conjecture.engine import ConjectureRunner
ShrinkPredicateT = Callable[[Union[ConjectureResult, _Overrun]], bool]

def sort_key(nodes: Sequence[ChoiceNode]) -> tuple[int, tuple[int, ...]]:
    """Returns a sort key such that "simpler" choice sequences are smaller than
    "more complicated" ones.

    We define sort_key so that x is simpler than y if x is shorter than y or if
    they have the same length and map(choice_to_index, x) < map(choice_to_index, y).

    The reason for using this ordering is:

    1. If x is shorter than y then that means we had to make fewer decisions
       in constructing the test case when we ran x than we did when we ran y.
    2. If x is the same length as y then replacing a choice with a lower index
       choice corresponds to replacing it with a simpler/smaller choice.
    3. Because choices drawn early in generation potentially get used in more
       places they potentially have a more significant impact on the final
       result, so it makes sense to prioritise reducing earlier choices over
       later ones.
    """
    return (len(nodes), tuple((choice_to_index(node.value, node.kwargs) for node in nodes)))
SHRINK_PASS_DEFINITIONS: Dict[str, Any] = {}

@attr.s()
class ShrinkPassDefinition:
    """A shrink pass bundles together a large number of local changes to
    the current shrink target.

    Each shrink pass is defined by some function and some arguments to that
    function. The ``generate_arguments`` function returns all arguments that
    might be useful to run on the current shrink target.

    The guarantee made by methods defined this way is that after they are
    called then *either* the shrink target has changed *or* each of
    ``fn(*args)`` has been called for every ``args`` in ``generate_arguments(self)``.
    No guarantee is made that all of these will be called if the shrink target
    changes.
    """
    run_with_chooser: Callable = attr.ib()

    @property
    def name(self) -> str:
        return self.run_with_chooser.__name__

    def __attrs_post_init__(self) -> None:
        assert self.name not in SHRINK_PASS_DEFINITIONS, self.name
        SHRINK_PASS_DEFINITIONS[self.name] = self

def defines_shrink_pass() -> Callable:
    """A convenient decorator for defining shrink passes."""

    def accept(run_step: Callable) -> Callable:
        ShrinkPassDefinition(run_with_chooser=run_step)

        def run(self: Any) -> None:
            raise NotImplementedError('Shrink passes should not be run directly')
        run.__name__ = run_step.__name__
        run.is_shrink_pass = True
        return run
    return accept

class Shrinker:
    """A shrinker is a child object of a ConjectureRunner which is designed to
    manage the associated state of a particular shrink problem. That is, we
    have some initial ConjectureData object and some property of interest
    that it satisfies, and we want to find a ConjectureData object with a
    shortlex (see sort_key above) smaller buffer that exhibits the same
    property.

    Currently the only property of interest we use is that the status is
    INTERESTING and the interesting_origin takes on some fixed value, but we
    may potentially be interested in other use cases later.
    However we assume that data with a status < VALID never satisfies the predicate.

    The shrinker keeps track of a value shrink_target which represents the
    current best known ConjectureData object satisfying the predicate.
    It refines this value by repeatedly running *shrink passes*, which are
    methods that perform a series of transformations to the current shrink_target
    and evaluate the underlying test function to find new ConjectureData
    objects. If any of these satisfy the predicate, the shrink_target
    is updated automatically. Shrinking runs until no shrink pass can
    improve the shrink_target, at which point it stops. It may also be
    terminated if the underlying engine throws RunIsComplete, but that
    is handled by the calling code rather than the Shrinker.

    =======================
    Designing Shrink Passes
    =======================

    Generally a shrink pass is just any function that calls
    cached_test_function and/or incorporate_new_buffer a number of times,
    but there are a couple of useful things to bear in mind.

    A shrink pass *makes progress* if running it changes self.shrink_target
    (i.e. it tries a shortlex smaller ConjectureData object satisfying
    the predicate). The desired end state of shrinking is to find a
    value such that no shrink pass can make progress, i.e. that we
    are at a local minimum for each shrink pass.

    In aid of this goal, the main invariant that a shrink pass much
    satisfy is that whether it makes progress must be deterministic.
    It is fine (encouraged even) for the specific progress it makes
    to be non-deterministic, but if you run a shrink pass, it makes
    no progress, and then you immediately run it again, it should
    never succeed on the second time. This allows us to stop as soon
    as we have run each shrink pass and seen no progress on any of
    them.

    This means that e.g. it's fine to try each of N deletions
    or replacements in a random order, but it's not OK to try N random
    deletions (unless you have already shrunk at least once, though we
    don't currently take advantage of this loophole).

    Shrink passes need to be written so as to be robust against
    change in the underlying shrink target. It is generally safe
    to assume that the shrink target does not change prior to the
    point of first modification - e.g. if you change no bytes at
    index ``i``, all examples whose start is ``<= i`` still exist,
    as do all blocks, and the data object is still of length
    ``>= i + 1``. This can only be violated by bad user code which
    relies on an external source of non-determinism.

    When the underlying shrink_target changes, shrink
    passes should not run substantially more test_function calls
    on success than they do on failure. Say, no more than a constant
    factor more. In particular shrink passes should not iterate to a
    fixed point.

    This means that shrink passes are often written with loops that
    are carefully designed to do the right thing in the case that no
    shrinks occurred and try to adapt to any changes to do a reasonable
    job. e.g. say we wanted to write a shrink pass that tried deleting
    each individual byte (this isn't an especially good choice,
    but it leads to a simple illustrative example), we might do it
    by iterating over the buffer like so:

    .. code-block:: python

        i = 0
        while i < len(self.shrink_target.buffer):
            if not self.incorporate_new_buffer(
                self.shrink_target.buffer[:i] + self.shrink_target.buffer[i + 1 :]
            ):
                i += 1

    The reason for writing the loop this way is that i is always a
    valid index into the current buffer, even if the current buffer
    changes as a result of our actions. When the buffer changes,
    we leave the index where it is rather than restarting from the
    beginning, and carry on. This means that the number of steps we
    run in this case is always bounded above by the number of steps
    we would run if nothing works.

    Another thing to bear in mind about shrink pass design is that
    they should prioritise *progress*. If you have N operations that
    you need to run, you should try to order them in such a way as
    to avoid stalling, where you have long periods of test function
    invocations where no shrinks happen. This is bad because whenever
    we shrink we reduce the amount of work the shrinker has to do
    in future, and often speed up the test function, so we ideally
    wanted those shrinks to happen much earlier in the process.

    Sometimes stalls are inevitable of course - e.g. if the pass
    makes no progress, then the entire thing is just one long stall,
    but it's helpful to design it so that stalls are less likely
    in typical behaviour.

    The two easiest ways to do this are:

    * Just run the N steps in random order. As long as a
      reasonably large proportion of the operations succeed, this
      guarantees the expected stall length is quite short. The
      book keeping for making sure this does the right thing when
      it succeeds can be quite annoying.
    * When you have any sort of nested loop, loop in such a way
      that both loop variables change each time. This prevents
      stalls which occur when one particular value for the outer
      loop is impossible to make progress on, rendering the entire
      inner loop into a stall.

    However, although progress is good, too much progress can be
    a bad sign! If you're *only* seeing successful reductions,
    that's probably a sign that you are making changes that are
    too timid. Two useful things to offset this:

    * It's worth writing shrink passes which are *adaptive*, in
      the sense that when operations seem to be working really
      well we try to bundle multiple of them together. This can
      often be used to turn what would be O(m) successful calls
      into O(log(m)).
    * It's often worth trying one or two special minimal values
      before trying anything more fine grained (e.g. replacing
      the whole thing with zero).

    """

    def derived_value(fn: Callable) -> property:
        """It's useful during shrinking to have access to derived values of
        the current shrink target.

        This decorator allows you to define these as cached properties. They
        are calculated once, then cached until the shrink target changes, then
        recalculated the next time they are used."""

        def accept(self: 'Shrinker') -> Any:
            try:
                return self.__derived_values[fn.__name__]
            except KeyError:
                return self.__derived_values.setdefault(fn.__name__, fn(self))
        accept.__name__ = fn.__name__
        return property(accept)

    def __init__(self, engine: 'ConjectureRunner', initial: Union[ConjectureData, ConjectureResult], predicate: Optional[ShrinkPredicateT], *, allow_transition: Optional[Callable[[Union[ConjectureData, ConjectureResult], ConjectureData], bool]], explain: bool, in_target_phase: bool=False) -> None:
        """Create a shrinker for a particular engine, with a given starting
        point and predicate. When shrink() is called it will attempt to find an
        example for which predicate is True and which is strictly smaller than
        initial.

        Note that initial is a ConjectureData object, and predicate
        takes ConjectureData objects.
        """
        assert predicate is not None or allow_transition is not None
        self.engine: 'ConjectureRunner' = engine
        self.__predicate: ShrinkPredicateT = predicate or (lambda data: True)
        self.__allow_transition: Callable[[Union[ConjectureData, ConjectureResult], ConjectureData], bool] = allow_transition or (lambda source, destination: True)
        self.__derived_values: Dict[str, Any] = {}
        self.__pending_shrink_explanation: Optional[str] = None
        self.initial_size: int = len(initial.choices)
        self.shrink_target: Union[ConjectureData, ConjectureResult] = initial
        self.clear_change_tracking()
        self.shrinks: int = 0
        self.max_stall: int = 200
        self.initial_calls: int = self.engine.call_count
        self.initial_misaligned: int = self.engine.misaligned_count
        self.calls_at_last_shrink: int = self.initial_calls
        self.passes_by_name: Dict[str, 'ShrinkPass'] = {}
        if in_target_phase:
            from hypothesis.internal.conjecture.engine import BUFFER_SIZE
            self.__extend: int = BUFFER_SIZE
        else:
            self.__extend: int = 0
        self.should_explain: bool = explain

    @derived_value
    def cached_calculations(self) -> Dict[Any, Any]:
        return {}

    def cached(self, *keys: Any) -> Callable:
        def accept(f: Callable) -> Any:
            cache_key: Tuple[Any, ...] = (f.__name__, *keys)
            try:
                return self.cached_calculations[cache_key]
            except KeyError:
                return self.cached_calculations.setdefault(cache_key, f())
        return accept

    def add_new_pass(self, run: Callable) -> 'ShrinkPass':
        """Creates a shrink pass corresponding to calling ``run(self)``"""
        definition: ShrinkPassDefinition = SHRINK_PASS_DEFINITIONS[run]
        p: ShrinkPass = ShrinkPass(run_with_chooser=definition.run_with_chooser, shrinker=self, index=len(self.passes_by_name))
        self.passes_by_name[p.name] = p
        return p

    def shrink_pass(self, name: Union[str, 'ShrinkPass']) -> 'ShrinkPass':
        """Return the ShrinkPass object for the pass with the given name."""
        if isinstance(name, ShrinkPass):
            return name
        if name not in self.passes_by_name:
            self.add_new_pass(name)
        return self.passes_by_name[name]

    @property
    def calls(self) -> int:
        """Return the number of calls that have been made to the underlying
        test function."""
        return self.engine.call_count

    @property
    def misaligned(self) -> int:
        return self.engine.misaligned_count

    def check_calls(self) -> None:
        if self.calls - self.calls_at_last_shrink >= self.max_stall:
            raise StopShrinking

    def cached_test_function_ir(self, nodes: Sequence[ChoiceNode]) -> Optional[Union[ConjectureResult, _Overrun]]:
        for node in nodes:
            if not choice_permitted(node.value, node.kwargs):
                return None
        result: Union[ConjectureResult, _Overrun] = self.engine.cached_test_function_ir([n.value for n in nodes])
        self.incorporate_test_data(result)
        self.check_calls()
        return result

    def consider_new_nodes(self, nodes: Sequence[ChoiceNode]) -> bool:
        nodes = nodes[:len(self.nodes)]
        if startswith(nodes, self.nodes):
            return True
        if sort_key(self.nodes) < sort_key(nodes):
            return False
        previous: Union[ConjectureData, ConjectureResult] = self.shrink_target
        self.cached_test_function_ir(nodes)
        return previous is not self.shrink_target

    def incorporate_test_data(self, data: Union[ConjectureResult, _Overrun]) -> None:
        """Takes a ConjectureData or Overrun object updates the current
        shrink_target if this data represents an improvement over it."""
        if data.status < Status.VALID or data is self.shrink_target:
            return
        if self.__predicate(data) and sort_key(data.nodes) < sort_key(self.shrink_target.nodes) and self.__allow_transition(self.shrink_target, data):
            self.update_shrink_target(data)

    def debug(self, msg: str) -> None:
        self.engine.debug(msg)

    @property
    def random(self) -> 'Random':
        return self.engine.random

    def shrink(self) -> None:
        """Run the full set of shrinks and update shrink_target.

        This method is "mostly idempotent" - calling it twice is unlikely to
        have any effect, though it has a non-zero probability of doing so.
        """
        try:
            self.initial_coarse_reduction()
            self.greedy_shrink()
        except StopShrinking:
            self.should_explain = False
        finally:
            if self.engine.report_debug_info:

                def s(n: int) -> str:
                    return 's' if n != 1 else ''
                total_deleted: int = self.initial_size - len(self.shrink_target.choices)
                calls: int = self.engine.call_count - self.initial_calls
                misaligned: int = self.engine.misaligned_count - self.initial_misaligned
                self.debug(f'---------------------\nShrink pass profiling\n---------------------\n\nShrinking made a total of {calls} call{s(calls)} of which {self.shrinks} shrank and {misaligned} were misaligned. This deleted {total_deleted} choices out of {self.initial_size}.')
                for useful in [True, False]:
                    self.debug('')
                    if useful:
                        self.debug('Useful passes:')
                    else:
                        self.debug('Useless passes:')
                    self.debug('')
                    for p in sorted(self.passes_by_name.values(), key=lambda t: (-t.calls, t.deletions, t.shrinks)):
                        if p.calls == 0:
                            continue
                        if (p