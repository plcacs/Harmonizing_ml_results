import math
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Optional, Union, cast, Tuple, List, Dict, Any
import attr
from hypothesis.internal.conjecture.choice import (
    ChoiceNode,
    ChoiceT,
    choice_equal,
    choice_from_index,
    choice_key,
    choice_permitted,
    choice_to_index,
)
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    ConjectureResult,
    Examples,
    Status,
    _Overrun,
    draw_choice,
)
from hypothesis.internal.conjecture.junkdrawer import endswith, find_integer, replace_all, startswith
from hypothesis.internal.conjecture.shrinking import Bytes, Float, Integer, Ordering, String
from hypothesis.internal.conjecture.shrinking.choicetree import (
    ChoiceTree,
    prefix_selection_order,
    random_selection_order,
)
from hypothesis.internal.floats import MAX_PRECISE_INTEGER
if TYPE_CHECKING:
    from random import Random
    from typing import TypeAlias
    from hypothesis.internal.conjecture.engine import ConjectureRunner

ShrinkPredicateT = Callable[[Union[ConjectureResult, _Overrun]], bool]

def sort_key(nodes: Sequence[ChoiceNode]) -> Tuple[int, Tuple[int, ...]]:
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

SHRINK_PASS_DEFINITIONS: Dict[str, 'ShrinkPassDefinition'] = {}

@attr.s
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
    run_with_chooser: Callable[['Shrinker', Any], Any] = attr.ib()

    @property
    def name(self) -> str:
        return self.run_with_chooser.__name__

    def __attrs_post_init__(self) -> None:
        assert self.name not in SHRINK_PASS_DEFINITIONS, self.name
        SHRINK_PASS_DEFINITIONS[self.name] = self

def defines_shrink_pass() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A convenient decorator for defining shrink passes."""

    def accept(run_step: Callable[['Shrinker', Any], Any]) -> Callable[..., Any]:
        ShrinkPassDefinition(run_with_chooser=run_step)

        def run(self: 'Shrinker') -> None:
            raise NotImplementedError('Shrink passes should not be run directly')
        run.__name__ = run_step.__name__
        run.is_shrink_pass = True  # type: ignore[attr-defined]
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

    ...

    """

    def derived_value(self, fn: Callable[['Shrinker'], Any]) -> property:
        """It's useful during shrinking to have access to derived values of
        the current shrink target.

        This decorator allows you to define these as cached properties. They
        are calculated once, then cached until the shrink target changes, then
        recalculated the next time they are used."""
        
        def accept() -> Any:
            try:
                return self.__derived_values[fn.__name__]
            except KeyError:
                return self.__derived_values.setdefault(fn.__name__, fn(self))
        accept.__name__ = fn.__name__
        return property(accept)

    def __init__(
        self,
        engine: 'ConjectureRunner',
        initial: ConjectureData,
        predicate: Optional[ShrinkPredicateT] = None,
        *,
        allow_transition: Optional[Callable[[ConjectureData, ConjectureData], bool]] = None,
        explain: bool = False,
        in_target_phase: bool = False
    ) -> None:
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
        self.__allow_transition: Callable[[ConjectureData, ConjectureData], bool] = allow_transition or (lambda source, destination: True)
        self.__derived_values: Dict[str, Any] = {}
        self.__pending_shrink_explanation: Optional[Any] = None
        self.initial_size: int = len(initial.choices)
        self.shrink_target: ConjectureData = initial
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
            self.__extend = 0
        self.should_explain: bool = explain

    @derived_value
    def cached_calculations(self) -> Dict[Tuple[Any, ...], Any]:
        return {}

    def cached(self, *keys: Any) -> Callable[[Callable[[], Any]], Any]:
        def accept(f: Callable[[], Any]) -> Any:
            cache_key: Tuple[Any, ...] = (f.__name__, *keys)
            try:
                return self.cached_calculations[cache_key]
            except KeyError:
                return self.cached_calculations.setdefault(cache_key, f())
        return accept

    def add_new_pass(self, run: str) -> 'ShrinkPass':
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

    def cached_test_function_ir(self, nodes: Tuple[ChoiceT, ...]) -> Optional[Union[ConjectureResult, _Overrun]]:
        for node in nodes:
            if not choice_permitted(node.value, node.kwargs):
                return None
        result: Union[ConjectureResult, _Overrun] = self.engine.cached_test_function_ir([n.value for n in nodes])
        self.incorporate_test_data(result)
        self.check_calls()
        return result

    def consider_new_nodes(self, nodes: Tuple[ChoiceT, ...]) -> bool:
        nodes = nodes[:len(self.nodes)]
        if startswith(nodes, self.nodes):
            return True
        if sort_key(self.nodes) < sort_key(nodes):
            return False
        previous: ConjectureData = self.shrink_target
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
                    for p in sorted(self.passes_by_name.values(), key=lambda t: (-t.calls, p.deletions, p.shrinks)):
                        if p.calls == 0:
                            continue
                        if (p.shrinks != 0) != useful:
                            continue
                        self.debug(f'  * {p.name} made {p.calls} call{s(p.calls)} of which {p.shrinks} shrank and {p.misaligned} were misaligned, deleting {p.deletions} choice{s(p.deletions)}.')
                self.debug('')
        self.explain()

    def explain(self) -> None:
        if not self.should_explain or not self.shrink_target.arg_slices:
            return
        self.max_stall = 2 ** 100
        shrink_target: ConjectureData = self.shrink_target
        nodes: Tuple[ChoiceNode, ...] = self.nodes
        choices: Tuple[ChoiceT, ...] = self.choices
        chunks: Dict[Tuple[int, int], List[Tuple[ChoiceT, ...]]] = defaultdict(list)
        seen_passing_seq = self.engine.passing_choice_sequences(prefix=self.nodes[:min(self.shrink_target.arg_slices)[0]])
        for start, end in sorted(self.shrink_target.arg_slices, key=lambda x: (-(x[1] - x[0]), x)):
            if any((startswith(seen, nodes[:start]) and endswith(seen, nodes[end:]) for seen in seen_passing_seq)):
                continue
            n_same_failures: int = 0
            note: str = 'or any other generated value'
            for n_attempt in range(500):
                if n_attempt - 10 > n_same_failures * 5:
                    break
                replacement: List[ChoiceT] = []
                for i in range(start, end):
                    node = nodes[i]
                    if not node.was_forced:
                        value = draw_choice(node.type, node.kwargs, random=self.random)
                        node = node.copy(with_value=value)
                    replacement.append(node.value)
                attempt: Tuple[ChoiceT, ...] = choices[:start] + tuple(replacement) + choices[end:]
                result: Union[ConjectureResult, _Overrun] = self.engine.cached_test_function_ir(attempt, extend='full')
                if result.status is Status.OVERRUN:
                    continue
                result = cast(ConjectureResult, result)
                if not (len(attempt) == len(result.choices) and endswith(result.nodes, nodes[end:])):
                    for ex, res in zip(shrink_target.examples, result.examples):
                        assert ex.start == res.start
                        assert ex.start <= start
                        assert ex.label == res.label
                        if start == ex.start and end == ex.end:
                            res_end: int = res.end
                            break
                    else:
                        raise NotImplementedError('Expected matching prefixes')
                    attempt = choices[:start] + result.choices[start:res_end] + choices[end:]
                    chunks[start, end].append(result.choices[start:res_end])
                    result = self.engine.cached_test_function_ir(attempt)
                    if result.status is Status.OVERRUN:
                        continue
                    result = cast(ConjectureResult, result)
                else:
                    chunks[start, end].append(result.choices[start:end])
                if shrink_target is not self.shrink_target:
                    self.shrink_target.slice_comments.clear()
                    return
                if result.status is Status.VALID:
                    break
                elif self.__predicate(result):
                    n_same_failures += 1
                    if n_same_failures >= 100:
                        self.shrink_target.slice_comments[start, end] = note
                        break
        if len(self.shrink_target.slice_comments) <= 1:
            return
        n_same_failures_together: int = 0
        chunks_by_start_index: List[Tuple[Tuple[int, int], List[Tuple[ChoiceT, ...]]]] = sorted(chunks.items())
        for _ in range(500):
            new_choices: List[ChoiceT] = []
            prev_end: int = 0
            for (start, end), ls in chunks_by_start_index:
                assert prev_end <= start < end, 'these chunks must be nonoverlapping'
                new_choices.extend(choices[prev_end:start])
                new_choices.extend(self.random.choice(ls))
                prev_end = end
            result: Union[ConjectureResult, _Overrun] = self.engine.cached_test_function_ir(tuple(new_choices))
            assert shrink_target is self.shrink_target
            if result.status == Status.VALID:
                self.shrink_target.slice_comments[0, 0] = 'The test sometimes passed when commented parts were varied together.'
                break
            elif self.__predicate(result):
                n_same_failures_together += 1
                if n_same_failures_together >= 100:
                    self.shrink_target.slice_comments[0, 0] = 'The test always failed when commented parts were varied together.'
                    break

    def greedy_shrink(self) -> None:
        """Run a full set of greedy shrinks (that is, ones that will only ever
        move to a better target) and update shrink_target appropriately.

        This method iterates to a fixed point and so is idempotent - calling
        it twice will have exactly the same effect as calling it once.
        """
        self.fixate_shrink_passes([
            'try_trivial_examples',
            node_program('X' * 5),
            node_program('X' * 4),
            node_program('X' * 3),
            node_program('X' * 2),
            node_program('X' * 1),
            'pass_to_descendant',
            'reorder_examples',
            'minimize_duplicated_nodes',
            'minimize_individual_nodes',
            'redistribute_numeric_pairs',
            'lower_integers_together',
        ])

    def initial_coarse_reduction(self) -> None:
        """Performs some preliminary reductions that should not be
        repeated as part of the main shrink passes.

        The main reason why these can't be included as part of shrink
        passes is that they have much more ability to make the test
        case "worse". e.g. they might rerandomise part of it, significantly
        increasing the value of individual nodes, which works in direct
        opposition to the lexical shrinking and will frequently undo
        its work.
        """
        self.reduce_each_alternative()

    @derived_value
    def examples_starting_at(self) -> Tuple[Tuple[int, ...], ...]:
        result: List[List[int]] = [[] for _ in self.shrink_target.nodes]
        for i, ex in enumerate(self.examples):
            if ex.start < len(result):
                result[ex.start].append(i)
        return tuple(map(tuple, result))

    def reduce_each_alternative(self) -> None:
        """This is a pass that is designed to rerandomise use of the
        one_of strategy or things that look like it, in order to try
        to move from later strategies to earlier ones in the branch
        order.

        It does this by trying to systematically lower each value it
        finds that looks like it might be the branch decision for
        one_of, and then attempts to repair any changes in shape that
        this causes.
        """
        i: int = 0
        while i < len(self.shrink_target.nodes):
            nodes: Tuple[ChoiceNode, ...] = self.shrink_target.nodes
            node: ChoiceNode = nodes[i]
            if node.type == 'integer' and (not node.was_forced) and (node.value <= 10) and (node.kwargs['min_value'] == 0):
                assert isinstance(node.value, int)
                zero_attempt: Optional[Union[ConjectureResult, _Overrun]] = self.cached_test_function_ir(
                    nodes[:i] + (node.copy(with_value=0),) + nodes[i + 1:]
                )
                if zero_attempt is not self.shrink_target and zero_attempt is not None and (zero_attempt.status >= Status.VALID):
                    changed_shape: bool = False
                    if len(zero_attempt.nodes) != len(nodes):
                        changed_shape = True
                    else:
                        for j in range(i + 1, len(nodes)):
                            zero_node = zero_attempt.nodes[j]
                            orig_node = nodes[j]
                            if zero_node.type != orig_node.type or not choice_permitted(orig_node.value, zero_node.kwargs):
                                changed_shape = True
                                break
                    if not changed_shape:
                        for j in range(i + 1, len(nodes)):
                            zero_node = zero_attempt.nodes[j]
                            orig_node = nodes[j]
                            if zero_node.type != orig_node.type or not choice_permitted(orig_node.value, zero_node.kwargs):
                                changed_shape = True
                                break
                    if changed_shape:
                        for v in range(node.value):
                            if self.try_lower_node_as_alternative(i, v):
                                break
            i += 1

    def try_lower_node_as_alternative(self, i: int, v: int) -> bool:
        """Attempt to lower `self.shrink_target.nodes[i]` to `v`,
        while rerandomising and attempting to repair any subsequent
        changes to the shape of the test case that this causes."""
        nodes: Tuple[ChoiceNode, ...] = self.shrink_target.nodes
        initial_attempt: Optional[Union[ConjectureResult, _Overrun]] = self.cached_test_function_ir(
            nodes[:i] + (nodes[i].copy(with_value=v),) + nodes[i + 1:]
        )
        if initial_attempt is self.shrink_target:
            self.lower_common_node_offset()
            return True
        prefix: Tuple[ChoiceT, ...] = nodes[:i] + (nodes[i].copy(with_value=v),)
        initial: ConjectureData = self.shrink_target
        examples: Tuple[int, ...] = self.examples_starting_at[i]
        for _ in range(3):
            random_attempt: Union[ConjectureResult, _Overrun] = self.engine.cached_test_function_ir(
                [n.value for n in prefix],
                extend=len(nodes)
            )
            if random_attempt.status < Status.VALID:
                continue
            self.incorporate_test_data(random_attempt)
            for j in examples:
                initial_ex = initial.examples[j]
                attempt_ex = random_attempt.examples[j]
                contents = random_attempt.nodes[attempt_ex.start:attempt_ex.end]
                self.consider_new_nodes(nodes[:i] + contents + nodes[initial_ex.end:])
                if initial is not self.shrink_target:
                    return True
        return False

    @derived_value
    def shrink_pass_choice_trees(self) -> Dict['ShrinkPass', ChoiceTree]:
        return defaultdict(ChoiceTree)

    def fixate_shrink_passes(self, passes: List[Union[str, 'ShrinkPass']]) -> None:
        """Run steps from each pass in ``passes`` until the current shrink target
        is a fixed point of all of them."""
        passes = list(map(self.shrink_pass, passes))
        any_ran: bool = True
        while any_ran:
            any_ran = False
            reordering: Dict['ShrinkPass', int] = {}
            can_discard: bool = self.remove_discarded()
            calls_at_loop_start: int = self.calls
            max_calls_per_failing_step: int = 1
            for sp in passes:
                if can_discard:
                    can_discard = self.remove_discarded()
                before_sp: ConjectureData = self.shrink_target
                failures: int = 0
                max_failures: int = 20
                while failures < max_failures:
                    self.max_stall = max(
                        self.max_stall,
                        2 * max_calls_per_failing_step + (self.calls - calls_at_loop_start)
                    )
                    prev: ConjectureData = self.shrink_target
                    initial_calls: int = self.calls
                    if not sp.step(random_order=failures >= max_failures // 2):
                        break
                    any_ran = True
                    if initial_calls != self.calls:
                        if prev is not self.shrink_target:
                            failures = 0
                        else:
                            max_calls_per_failing_step = max(
                                max_calls_per_failing_step, self.calls - initial_calls
                            )
                            failures += 1
                if self.shrink_target is before_sp:
                    reordering[sp] = 1
                elif len(self.choices) < len(before_sp.choices):
                    reordering[sp] = -1
                else:
                    reordering[sp] = 0
            passes.sort(key=reordering.__getitem__)

    @property
    def nodes(self) -> Tuple[ChoiceNode, ...]:
        return self.shrink_target.nodes

    @property
    def choices(self) -> Tuple[ChoiceT, ...]:
        return self.shrink_target.choices

    @property
    def examples(self) -> Examples:
        return self.shrink_target.examples

    @derived_value
    def examples_by_label(self) -> Dict[Any, Tuple[Any, ...]]:
        """An index of all examples grouped by their label, with
        the examples stored in their normal index order."""
        examples_by_label: defaultdict[Any, List[Any]] = defaultdict(list)
        for ex in self.examples:
            examples_by_label[ex.label].append(ex)
        return {k: tuple(v) for k, v in examples_by_label.items()}

    @derived_value
    def distinct_labels(self) -> List[Any]:
        return sorted(self.examples_by_label, key=str)

    def update_shrink_target(self, new_target: ConjectureResult) -> None:
        """Update the shrink target to a new target."""
        assert isinstance(new_target, ConjectureResult)
        self.shrinks += 1
        self.max_stall = max(self.max_stall, (self.calls - self.calls_at_last_shrink) * 2)
        self.calls_at_last_shrink = self.calls
        self.shrink_target = new_target
        self.__derived_values = {}

    def minimize_nodes(self, nodes: List[ChoiceNode]) -> None:
        """Minimize the given nodes."""
        choice_type: str = nodes[0].type
        value: Any = nodes[0].value
        kwargs: Dict[str, Any] = nodes[0].kwargs
        assert all((node.type == choice_type and choice_equal(node.value, value) for node in nodes))
        if choice_type == 'integer':
            shrink_towards: int = kwargs['shrink_towards']
            self.try_shrinking_nodes(nodes, abs(shrink_towards - value))
            Integer.shrink(abs(shrink_towards - value), lambda n: self.try_shrinking_nodes(nodes, shrink_towards + n))
            Integer.shrink(abs(shrink_towards - value), lambda n: self.try_shrinking_nodes(nodes, shrink_towards - n))
        elif choice_type == 'float':
            self.try_shrinking_nodes(nodes, abs(value))
            Float.shrink(abs(value), lambda val: self.try_shrinking_nodes(nodes, val))
            Float.shrink(abs(value), lambda val: self.try_shrinking_nodes(nodes, -val))
        elif choice_type == 'boolean':
            assert value is True
            self.try_shrinking_nodes(nodes, False)
        elif choice_type == 'bytes':
            Bytes.shrink(value, lambda val: self.try_shrinking_nodes(nodes, val), min_size=kwargs['min_size'])
        elif choice_type == 'string':
            String.shrink(
                value,
                lambda val: self.try_shrinking_nodes(nodes, val),
                intervals=kwargs['intervals'],
                min_size=kwargs['min_size'],
            )
        else:
            raise NotImplementedError

    def try_shrinking_nodes(self, nodes: List[ChoiceNode], n: int) -> bool:
        """Attempts to replace each node in the nodes list with n. Returns
        True if it succeeded (which may include some additional modifications
        to shrink_target).

        In current usage it is expected that each of the nodes currently have
        the same value and ir type, although this is not essential. Note that
        n must be < the node at min(nodes) or this is not a valid shrink.

        This method will attempt to do some small amount of work to delete data
        that occurs after the end of the nodes. This is useful for cases where
        there is some size dependency on the value of a node.
        """
        if any((node.index >= len(self.nodes) for node in nodes)):
            return False
        initial_attempt: Optional[Union[ConjectureResult, _Overrun]] = replace_all(
            self.nodes,
            [(node.index, node.index + 1, [node.copy(with_value=n)]) for node in nodes],
        )
        attempt: Optional[Union[ConjectureResult, _Overrun]] = self.cached_test_function_ir(initial_attempt)
        if attempt is None:
            return False
        if attempt is self.shrink_target:
            self.lower_common_node_offset()
            return True
        if attempt.status is Status.OVERRUN:
            return False
        if attempt.status is Status.INVALID:
            return False
        if attempt.misaligned_at is not None:
            index, attempt_choice_type, attempt_kwargs, _attempt_forced = attempt.misaligned_at
            node: ChoiceNode = self.nodes[index]
            if node.type != attempt_choice_type:
                return False
            if node.was_forced:
                return False
            if node.type in {'string', 'bytes'}:
                if node.kwargs['min_size'] <= attempt_kwargs['min_size']:
                    return False
                return self.consider_new_nodes(
                    initial_attempt[:node.index]
                    + [
                        initial_attempt[node.index].copy(
                            with_kwargs=attempt_kwargs,
                            with_value=initial_attempt[node.index].value[:attempt_kwargs['min_size']],
                        )
                    ]
                    + initial_attempt[node.index:]
                )
        lost_nodes: int = len(self.nodes) - len(attempt.nodes)
        if lost_nodes <= 0:
            return False
        start: int = nodes[0].index
        end: int = nodes[-1].index + 1
        regions_to_delete: set[Tuple[int, int]] = {(end, end + lost_nodes)}
        for ex in self.examples:
            if ex.start > start:
                continue
            if ex.end <= end:
                continue
            if ex.index >= len(attempt.examples):
                continue
            replacement = attempt.examples[ex.index]
            in_original = [c for c in ex.children if c.start >= end]
            in_replaced = [c for c in replacement.children if c.start >= end]
            if len(in_replaced) >= len(in_original) or not in_replaced:
                continue
            regions_to_delete.add((in_original[0].start, in_original[-len(in_replaced)].start))
        for u, v in sorted(regions_to_delete, key=lambda x: x[1] - x[0], reverse=True):
            try_with_deleted: Tuple[ChoiceT, ...] = initial_attempt[:u] + initial_attempt[v:]
            if self.consider_new_nodes(try_with_deleted):
                return True
        return False

    def remove_discarded(self) -> bool:
        """Try removing all bytes marked as discarded.

        This is primarily to deal with data that has been ignored while
        doing rejection sampling - e.g. as a result of an integer range, or a
        filtered strategy.

        Such data will also be handled by the adaptive_example_deletion pass,
        but that pass is necessarily more conservative and will try deleting
        each interval individually. The common case is that all data drawn and
        rejected can just be thrown away immediately in one block, so this pass
        will be much faster than trying each one individually when it works.

        returns False if there is discarded data and removing it does not work,
        otherwise returns True.
        """
        while self.shrink_target.has_discards:
            discarded: List[Tuple[int, int]] = []
            for ex in self.shrink_target.examples:
                if ex.choice_count > 0 and ex.discarded and (not discarded or ex.start >= discarded[-1][-1]):
                    discarded.append((ex.start, ex.end))
            if not discarded:
                break
            attempt: List[ChoiceT] = list(self.nodes)
            for u, v in reversed(discarded):
                del attempt[u:v]
            if not self.consider_new_nodes(tuple(attempt)):
                return False
        return True

    def explain(self) -> None:
        if not self.should_explain or not self.shrink_target.arg_slices:
            return
        self.max_stall = 2 ** 100
        shrink_target: ConjectureData = self.shrink_target
        nodes: Tuple[ChoiceNode, ...] = self.nodes
        choices: Tuple[ChoiceT, ...] = self.choices
        chunks: Dict[Tuple[int, int], List[Tuple[ChoiceT, ...]]] = defaultdict(list)
        seen_passing_seq = self.engine.passing_choice_sequences(prefix=self.nodes[:min(self.shrink_target.arg_slices)[0]])
        for start, end in sorted(self.shrink_target.arg_slices, key=lambda x: (-(x[1] - x[0]), x)):
            if any((startswith(seen, nodes[:start]) and endswith(seen, nodes[end:]) for seen in seen_passing_seq)):
                continue
            n_same_failures: int = 0
            note: str = 'or any other generated value'
            for n_attempt in range(500):
                if n_attempt - 10 > n_same_failures * 5:
                    break
                replacement: List[ChoiceT] = []
                for i in range(start, end):
                    node = nodes[i]
                    if not node.was_forced:
                        value = draw_choice(node.type, node.kwargs, random=self.random)
                        node = node.copy(with_value=value)
                    replacement.append(node.value)
                attempt: Tuple[ChoiceT, ...] = choices[:start] + tuple(replacement) + choices[end:]
                result: Union[ConjectureResult, _Overrun] = self.engine.cached_test_function_ir(attempt, extend='full')
                if result.status is Status.OVERRUN:
                    continue
                result = cast(ConjectureResult, result)
                if not (len(attempt) == len(result.choices) and endswith(result.nodes, nodes[end:])):
                    for ex, res in zip(shrink_target.examples, result.examples):
                        assert ex.start == res.start
                        assert ex.start <= start
                        assert ex.label == res.label
                        if start == ex.start and end == ex.end:
                            res_end: int = res.end
                            break
                    else:
                        raise NotImplementedError('Expected matching prefixes')
                    attempt = choices[:start] + result.choices[start:res_end] + choices[end:]
                    chunks[start, end].append(result.choices[start:res_end])
                    result = self.engine.cached_test_function_ir(attempt)
                    if result.status is Status.OVERRUN:
                        continue
                    result = cast(ConjectureResult, result)
                else:
                    chunks[start, end].append(result.choices[start:end])
                if shrink_target is not self.shrink_target:
                    self.shrink_target.slice_comments.clear()
                    return
                if result.status is Status.VALID:
                    break
                elif self.__predicate(result):
                    n_same_failures += 1
                    if n_same_failures >= 100:
                        self.shrink_target.slice_comments[start, end] = note
                        break
        if len(self.shrink_target.slice_comments) <= 1:
            return
        n_same_failures_together: int = 0
        chunks_by_start_index: List[Tuple[Tuple[int, int], List[Tuple[ChoiceT, ...]]]] = sorted(chunks.items())
        for _ in range(500):
            new_choices: List[ChoiceT] = []
            prev_end: int = 0
            for (start, end), ls in chunks_by_start_index:
                assert prev_end <= start < end, 'these chunks must be nonoverlapping'
                new_choices.extend(choices[prev_end:start])
                new_choices.extend(self.random.choice(ls))
                prev_end = end
            result: Union[ConjectureResult, _Overrun] = self.engine.cached_test_function_ir(tuple(new_choices))
            assert shrink_target is self.shrink_target
            if result.status == Status.VALID:
                self.shrink_target.slice_comments[0, 0] = 'The test sometimes passed when commented parts were varied together.'
                break
            elif self.__predicate(result):
                n_same_failures_together += 1
                if n_same_failures_together >= 100:
                    self.shrink_target.slice_comments[0, 0] = 'The test always failed when commented parts were varied together.'
                    break

    def lower_common_node_offset(self) -> None:
        """Sometimes we find ourselves in a situation where changes to one part
        of the choice sequence unlock changes to other parts. Sometimes this is
        good, but sometimes this can cause us to exhibit exponential slow
        downs!

        e.g. suppose we had the following:

        m = draw(integers(min_value=0))
        n = draw(integers(min_value=0))
        assert abs(m - n) > 1

        If this fails then we'll end up with a loop where on each iteration we
        reduce each of m and n by 2 - m can't go lower because of n, then n
        can't go lower because of m.

        This will take us O(m) iterations to complete, which is exponential in
        the data size, as we gradually zig zag our way towards zero.

        This can only happen if we're failing to reduce the size of the choice
        sequence: The number of iterations that reduce the length of the choice
        sequence is bounded by that length.

        So what we do is this: We keep track of which blocks are changing, and
        then if there's some non-zero common offset to them we try and minimize
        them all at once by lowering that offset.

        This may not work, and it definitely won't get us out of all possible
        exponential slow downs (an example of where it doesn't is where the
        shape of the blocks changes as a result of this bouncing behaviour),
        but it fails fast when it doesn't work and gets us out of a really
        nastily slow case when it does.
        """
        if len(self.__changed_nodes) <= 1:
            return
        changed: List[ChoiceNode] = []
        for i in sorted(self.__changed_nodes):
            node: ChoiceNode = self.nodes[i]
            if node.trivial or node.type != 'integer':
                continue
            changed.append(node)
        if not changed:
            return
        ints: List[int] = [abs(node.value - node.kwargs['shrink_towards']) for node in changed]
        offset: int = min(ints)
        assert offset > 0
        for i in range(len(ints)):
            ints[i] -= offset
        st: ConjectureData = self.shrink_target

        def offset_node(node: ChoiceNode, n: int) -> Tuple[int, int, List[ChoiceT]]:
            return (
                node.index,
                node.index + 1,
                [node.copy(with_value=node.kwargs['shrink_towards'] + n)]
            )

        def consider(n: int, sign: int) -> bool:
            return self.consider_new_nodes(
                replace_all(
                    st.nodes,
                    [offset_node(node, sign * (n + v)) for node, v in zip(changed, ints)]
                )
            )
        Integer.shrink(offset, lambda n: consider(n, 1))
        Integer.shrink(offset, lambda n: consider(n, -1))
        self.clear_change_tracking()

    def clear_change_tracking(self) -> None:
        self.__last_checked_changed_at = self.shrink_target
        self.__all_changed_nodes: set[int] = set()

    def mark_changed(self, i: int) -> None:
        self.__all_changed_nodes.add(i)

class StopShrinking(Exception):
    pass

def shrink_pass_family(f: Callable[..., Any]) -> Callable[..., Any]:
    
    def accept(*args: Any) -> str:
        name: str = '{}({})'.format(f.__name__, ', '.join(map(repr, args)))
        if name not in SHRINK_PASS_DEFINITIONS:

            def run(self: Shrinker, chooser: Any) -> None:
                return f(self, chooser, *args)
            run.__name__ = name
            defines_shrink_pass()(run)
        assert name in SHRINK_PASS_DEFINITIONS
        return name
    return accept

@shrink_pass_family
def node_program(chooser: Any, description: str) -> str:
    n: int = len(description)
    i: int = chooser.choose(range(len(self.nodes) - n + 1))
    if not self.run_node_program(i, description, original=self.shrink_target):
        return

    def offset_left(k: int) -> int:
        return i - k * n
    i = offset_left(find_integer(lambda k: self.run_node_program(offset_left(k), description, original=self.shrink_target)))
    original: ConjectureData = self.shrink_target
    find_integer(lambda k: self.run_node_program(i, description, original=original, repeats=k))

@attr.s(slots=True, eq=False)
class ShrinkPass:
    run_with_chooser: Callable[['Shrinker', Any], Any] = attr.ib()
    index: int = attr.ib()
    shrinker: Shrinker = attr.ib()
    last_prefix: Tuple[Any, ...] = attr.ib(default=())
    successes: int = attr.ib(default=0)
    calls: int = attr.ib(default=0)
    misaligned: int = attr.ib(default=0)
    shrinks: int = attr.ib(default=0)
    deletions: int = attr.ib(default=0)

    def step(self, *, random_order: bool = False) -> bool:
        tree: ChoiceTree = self.shrinker.shrink_pass_choice_trees[self]
        if tree.exhausted:
            return False
        initial_shrinks: int = self.shrinker.shrinks
        initial_calls: int = self.shrinker.calls
        initial_misaligned: int = self.shrinker.misaligned
        size: int = len(self.shrinker.shrink_target.choices)
        self.shrinker.engine.explain_next_call_as(self.name)
        if random_order:
            selection_order: Any = random_selection_order(self.shrinker.random)
        else:
            selection_order: Any = prefix_selection_order(self.last_prefix)
        try:
            self.last_prefix = tree.step(selection_order, lambda chooser: self.run_with_chooser(self.shrinker, chooser))
        finally:
            self.calls += self.shrinker.calls - initial_calls
            self.misaligned += self.shrinker.misaligned - initial_misaligned
            self.shrinks += self.shrinker.shrinks - initial_shrinks
            self.deletions += size - len(self.shrinker.shrink_target.choices)
            self.shrinker.engine.clear_call_explanation()
        return True

    @property
    def name(self) -> str:
        return self.run_with_chooser.__name__
