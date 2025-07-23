import math
from collections import defaultdict
from collections.abc import Sequence, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    TypeVar,
    Generic,
    Set,
    DefaultDict,
    Iterator,
    Iterable,
)
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
from hypothesis.internal.conjecture.junkdrawer import (
    endswith,
    find_integer,
    replace_all,
    startswith,
)
from hypothesis.internal.conjecture.shrinking import (
    Bytes,
    Float,
    Integer,
    Ordering,
    String,
)
from hypothesis.internal.conjecture.shrinking.choicetree import (
    ChoiceTree,
    prefix_selection_order,
    random_selection_order,
)
from hypothesis.internal.floats import MAX_PRECISE_INTEGER

if TYPE_CHECKING:
    from random import Random
    from typing import TypeAlias, Protocol
    from hypothesis.internal.conjecture.engine import ConjectureRunner

T = TypeVar("T")
ShrinkPredicateT = Callable[[Union[ConjectureResult, _Overrun]], bool]


def sort_key(nodes: Sequence[ChoiceNode]) -> Tuple[int, Tuple[int, ...]]:
    return (
        len(nodes),
        tuple((choice_to_index(node.value, node.kwargs) for node in nodes),
    )


SHRINK_PASS_DEFINITIONS: Dict[str, "ShrinkPassDefinition"] = {}


@attr.s()
class ShrinkPassDefinition:
    run_with_chooser: Callable[["Shrinker", "Chooser"], Any] = attr.ib()

    @property
    def name(self) -> str:
        return self.run_with_chooser.__name__

    def __attrs_post_init__(self) -> None:
        assert self.name not in SHRINK_PASS_DEFINITIONS, self.name
        SHRINK_PASS_DEFINITIONS[self.name] = self


def defines_shrink_pass() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def accept(run_step: Callable[..., Any]) -> Callable[..., Any]:
        ShrinkPassDefinition(run_with_chooser=run_step)

        def run(self: Any) -> None:
            raise NotImplementedError("Shrink passes should not be run directly")

        run.__name__ = run_step.__name__
        run.is_shrink_pass = True
        return run

    return accept


class Shrinker:
    def derived_value(fn: Callable[["Shrinker"], T]) -> property:
        def accept(self: "Shrinker") -> T:
            try:
                return self.__derived_values[fn.__name__]
            except KeyError:
                return self.__derived_values.setdefault(fn.__name__, fn(self))

        accept.__name__ = fn.__name__
        return property(accept)

    def __init__(
        self,
        engine: "ConjectureRunner",
        initial: ConjectureData,
        predicate: Optional[ShrinkPredicateT],
        *,
        allow_transition: Optional[Callable[[ConjectureData, ConjectureData], bool]],
        explain: bool,
        in_target_phase: bool = False,
    ):
        self.engine = engine
        self.__predicate = predicate or (lambda data: True)
        self.__allow_transition = allow_transition or (lambda source, destination: True)
        self.__derived_values: Dict[str, Any] = {}
        self.__pending_shrink_explanation: Optional[str] = None
        self.initial_size = len(initial.choices)
        self.shrink_target = initial
        self.clear_change_tracking()
        self.shrinks = 0
        self.max_stall = 200
        self.initial_calls = self.engine.call_count
        self.initial_misaligned = self.engine.misaligned_count
        self.calls_at_last_shrink = self.initial_calls
        self.passes_by_name: Dict[str, "ShrinkPass"] = {}
        if in_target_phase:
            from hypothesis.internal.conjecture.engine import BUFFER_SIZE

            self.__extend = BUFFER_SIZE
        else:
            self.__extend = 0
        self.should_explain = explain

    @derived_value
    def cached_calculations(self) -> Dict[Tuple[str, ...], Any]:
        return {}

    def cached(self, *keys: Any) -> Callable[[Callable[[], T]], T]:
        def accept(f: Callable[[], T]) -> T:
            cache_key = (f.__name__, *keys)
            try:
                return self.cached_calculations[cache_key]
            except KeyError:
                return self.cached_calculations.setdefault(cache_key, f())

        return accept

    def add_new_pass(self, run: str) -> "ShrinkPass":
        definition = SHRINK_PASS_DEFINITIONS[run]
        p = ShrinkPass(
            run_with_chooser=definition.run_with_chooser,
            shrinker=self,
            index=len(self.passes_by_name),
        )
        self.passes_by_name[p.name] = p
        return p

    def shrink_pass(self, name: Union[str, "ShrinkPass"]) -> "ShrinkPass":
        if isinstance(name, ShrinkPass):
            return name
        if name not in self.passes_by_name:
            self.add_new_pass(name)
        return self.passes_by_name[name]

    @property
    def calls(self) -> int:
        return self.engine.call_count

    @property
    def misaligned(self) -> int:
        return self.engine.misaligned_count

    def check_calls(self) -> None:
        if self.calls - self.calls_at_last_shrink >= self.max_stall:
            raise StopShrinking

    def cached_test_function_ir(
        self, nodes: Sequence[ChoiceNode]
    ) -> Optional[ConjectureResult]:
        for node in nodes:
            if not choice_permitted(node.value, node.kwargs):
                return None
        result = self.engine.cached_test_function_ir([n.value for n in nodes])
        self.incorporate_test_data(result)
        self.check_calls()
        return result

    def consider_new_nodes(self, nodes: Sequence[ChoiceNode]) -> bool:
        nodes = nodes[: len(self.nodes)]
        if startswith(nodes, self.nodes):
            return True
        if sort_key(self.nodes) < sort_key(nodes):
            return False
        previous = self.shrink_target
        self.cached_test_function_ir(nodes)
        return previous is not self.shrink_target

    def incorporate_test_data(self, data: Union[ConjectureResult, _Overrun]) -> None:
        if data.status < Status.VALID or data is self.shrink_target:
            return
        if (
            self.__predicate(data)
            and sort_key(data.nodes) < sort_key(self.shrink_target.nodes)
            and self.__allow_transition(self.shrink_target, data)
        ):
            self.update_shrink_target(cast(ConjectureResult, data))

    def debug(self, msg: str) -> None:
        self.engine.debug(msg)

    @property
    def random(self) -> "Random":
        return self.engine.random

    def shrink(self) -> None:
        try:
            self.initial_coarse_reduction()
            self.greedy_shrink()
        except StopShrinking:
            self.should_explain = False
        finally:
            if self.engine.report_debug_info:

                def s(n: int) -> str:
                    return "s" if n != 1 else ""

                total_deleted = self.initial_size - len(self.shrink_target.choices)
                calls = self.engine.call_count - self.initial_calls
                misaligned = self.engine.misaligned_count - self.initial_misaligned
                self.debug(
                    f"---------------------\nShrink pass profiling\n---------------------\n\nShrinking made a total of {calls} call{s(calls)} of which {self.shrinks} shrank and {misaligned} were misaligned. This deleted {total_deleted} choices out of {self.initial_size}."
                )
                for useful in [True, False]:
                    self.debug("")
                    if useful:
                        self.debug("Useful passes:")
                    else:
                        self.debug("Useless passes:")
                    self.debug("")
                    for p in sorted(
                        self.passes_by_name.values(),
                        key=lambda t: (-t.calls, t.deletions, t.shrinks),
                    ):
                        if p.calls == 0:
                            continue
                        if (p.shrinks != 0) != useful:
                            continue
                        self.debug(
                            f"  * {p.name} made {p.calls} call{s(p.calls)} of which {p.shrinks} shrank and {p.misaligned} were misaligned, deleting {p.deletions} choice{s(p.deletions)}."
                        )
                self.debug("")
        self.explain()

    def explain(self) -> None:
        if not self.should_explain or not self.shrink_target.arg_slices:
            return
        self.max_stall = 2**100
        shrink_target = self.shrink_target
        nodes = self.nodes
        choices = self.choices
        chunks: DefaultDict[Tuple[int, int], List[List[int]]] = defaultdict(list)
        seen_passing_seq = self.engine.passing_choice_sequences(
            prefix=self.nodes[: min(self.shrink_target.arg_slices)[0]]
        )
        for start, end in sorted(
            self.shrink_target.arg_slices, key=lambda x: (-(x[1] - x[0]), x)
        ):
            if any(
                startswith(seen, nodes[:start]) and endswith(seen, nodes[end:])
                for seen in seen_passing_seq
            ):
                continue
            n_same_failures = 0
            note = "or any other generated value"
            for n_attempt in range(500):
                if n_attempt - 10 > n_same_failures * 5:
                    break
                replacement = []
                for i in range(start, end):
                    node = nodes[i]
                    if not node.was_forced:
                        value = draw_choice(
                            node.type, node.kwargs, random=self.random
                        )
                        node = node.copy(with_value=value)
                    replacement.append(node.value)
                attempt = choices[:start] + tuple(replacement) + choices[end:]
                result = self.engine.cached_test_function_ir(attempt, extend="full")
                if result.status is Status.OVERRUN:
                    continue
                result = cast(ConjectureResult, result)
                if not (
                    len(attempt) == len(result.choices)
                    and endswith(result.nodes, nodes[end:])
                ):
                    for ex, res in zip(shrink_target.examples, result.examples):
                        assert ex.start == res.start
                        assert ex.start <= start
                        assert ex.label == res.label
                        if start == ex.start and end == ex.end:
                            res_end = res.end
                            break
                    else:
                        raise NotImplementedError("Expected matching prefixes")
                    attempt = (
                        choices[:start]
                        + result.choices[start:res_end]
                        + choices[end:]
                    )
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
        n_same_failures_together = 0
        chunks_by_start_index = sorted(chunks.items())
        for _ in range(500):
            new_choices = []
            prev_end = 0
            for (start, end), ls in chunks_by_start_index:
                assert prev_end <= start < end, "these chunks must be nonoverlapping"
                new_choices.extend(choices[prev_end:start])
                new_choices.extend(self.random.choice(ls))
                prev_end = end
            result = self.engine.cached_test_function_ir(new_choices)
            assert shrink_target is self.shrink_target
            if result.status == Status.VALID:
                self.shrink_target.slice_comments[
                    0, 0
                ] = "The test sometimes passed when commented parts were varied together."
                break
            elif self.__predicate(result):
                n_same_failures_together += 1
                if n_same_failures_together >= 100:
                    self.shrink_target.slice_comments[
                        0, 0
                    ] = "The test always failed when commented parts were varied together."
                    break

    def greedy_shrink(self) -> None:
        self.fixate_shrink_passes(
            [
                "try_trivial_examples",
                node_program("X" * 5),
                node_program("X" * 4),
                node_program("X" * 3),
                node_program("X" * 2),
                node_program("X" * 1),
                "pass_to_descendant",
                "reorder_examples",
                "minimize_duplicated_nodes",
                "minimize_individual_nodes",
                "redistribute_numeric_pairs",
                "lower_integers_together",
            ]
        )

    def initial_coarse_reduction(self) -> None:
        self.reduce_each_alternative()

    @derived_value
    def examples_starting_at(self) -> Tuple[Tuple[int, ...], ...]:
        result = [[] for _ in self.shrink_target.nodes]
        for i, ex in enumerate(self.examples):
            if ex.start < len(result):
                result[ex.start].append(i)
        return tuple(map(tuple, result))

    def reduce_each_alternative(self) -> None:
        i = 0
        while i < len(self.shrink_target.nodes):
            nodes = self.shrink_target.nodes
            node = nodes[i]
            if (
                node.type == "integer"
                and (not node.was_forced)
                and (node.value <= 10)
                and (node.kwargs["min_value"] == 0)
            ):
                assert isinstance(node.value, int)
                zero_attempt = self.cached_test_function_ir(
                    nodes[:i] + (nodes[i].copy(with_value=0),) + nodes[i + 1 :]
                )
                if (
                    zero_attempt is not self.shrink_target
                    and zero_attempt is not None
                    and (zero_attempt.status >= Status.VALID)
                ):
                    changed_shape = len(zero_attempt.nodes) != len(nodes)
                    if not changed_shape:
                        for j in range(i + 1, len(nodes)):
                            zero_node = zero_attempt.nodes[j]
                            orig_node = nodes[j]
                            if (
                                zero_node.type != orig_node.type
                                or not choice_permitted(orig_node.value, zero_node.kwargs)
                            ):
                                changed_shape = True
                                break
                    if changed_shape:
                        for v in range(node.value):
                            if self.try_lower_node_as_alternative(i, v):
                                break
            i += 1

    def try_lower_node_as_alternative(self, i: int, v: int) -> bool:
        nodes = self.shrink_target.nodes
        initial_attempt = self.cached_test_function_ir(
            nodes[:i] + (nodes[i].copy(with_value=v),) + nodes[i + 1 :]
        )
        if initial_attempt is self.shrink_target:
            return True
        prefix = nodes[:i] + (nodes[i].copy(with_value=v),)
        initial = self.shrink_target
        examples = self.examples_starting_at[i]
        for _ in range(3):
            random_attempt = self.engine.cached_test_function_ir(
                [n.value for n in prefix], extend=len(nodes)
            )
            if random_attempt.status < Status.VALID:
                continue
            self.incorporate_test_data(random_attempt)
            for j in examples:
                initial_ex = initial.examples[j]
                attempt_ex = random_attempt.examples[j]
                contents = random_attempt.nodes[attempt_ex.start : attempt_ex.end]
                self.consider_new_nodes(
                    nodes[:i] + contents + nodes[initial_ex.end :]
                )
                if initial is not self.shrink_target:
                    return True
        return False

    @derived_value
    def shrink_pass_choice_trees(self) -> DefaultDict["ShrinkPass", ChoiceTree]:
        return defaultdict(ChoiceTree)

    def fixate_shrink_passes(self, passes: List[Union[str, "ShrinkPass"]]) -> None:
        passes = list(map(self.shrink_pass, passes))
        any_ran = True
        while any_ran:
            any_ran = False
            reordering: Dict["ShrinkPass", int] = {}
            can_discard = self.remove_discarded()
            calls_at_loop_start = self.calls
            max_calls_per_failing_step = 1
            for sp in passes:
                if can_discard:
                    can_discard = self.remove_discarded()
                before_sp = self.shrink_target
                failures = 0
                max_failures = 20
                while failures < max_failures:
                    self.max_stall = max(
                        self.max_stall,
                        2 * max_calls_per_failing_step
                        + (self.calls - calls_at_loop_start),
                    )
                    prev = self.shrink_target
                    initial_calls =