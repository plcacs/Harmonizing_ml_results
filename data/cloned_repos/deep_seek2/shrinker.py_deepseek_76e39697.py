# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Optional, Union, cast, Dict, List, Tuple, Set, Any

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
    from typing import TypeAlias

    from hypothesis.internal.conjecture.engine import ConjectureRunner

ShrinkPredicateT: "TypeAlias" = Callable[[Union[ConjectureResult, _Overrun]], bool]


def sort_key(nodes: Sequence[ChoiceNode]) -> Tuple[int, Tuple[int, ...]]:
    return (
        len(nodes),
        tuple(choice_to_index(node.value, node.kwargs) for node in nodes),
    )


SHRINK_PASS_DEFINITIONS: Dict[str, "ShrinkPassDefinition"] = {}


@attr.s()
class ShrinkPassDefinition:
    run_with_chooser: Callable = attr.ib()

    @property
    def name(self) -> str:
        return self.run_with_chooser.__name__

    def __attrs_post_init__(self) -> None:
        assert self.name not in SHRINK_PASS_DEFINITIONS, self.name
        SHRINK_PASS_DEFINITIONS[self.name] = self


def defines_shrink_pass():
    def accept(run_step: Callable) -> Callable:
        ShrinkPassDefinition(run_with_chooser=run_step)

        def run(self) -> None:
            raise NotImplementedError("Shrink passes should not be run directly")

        run.__name__ = run_step.__name__
        run.is_shrink_pass = True
        return run

    return accept


class Shrinker:
    def derived_value(fn: Callable) -> property:
        def accept(self) -> Any:
            try:
                return self.__derived_values[fn.__name__]
            except KeyError:
                return self.__derived_values.setdefault(fn.__name__, fn(self))

        accept.__name__ = fn.__name__
        return property(accept)

    def __init__(
        self,
        engine: "ConjectureRunner",
        initial: Union[ConjectureData, ConjectureResult],
        predicate: Optional[ShrinkPredicateT],
        *,
        allow_transition: Optional[
            Callable[[Union[ConjectureData, ConjectureResult], ConjectureData], bool]
        ],
        explain: bool,
        in_target_phase: bool = False,
    ):
        assert predicate is not None or allow_transition is not None
        self.engine = engine
        self.__predicate = predicate or (lambda data: True)
        self.__allow_transition = allow_transition or (lambda source, destination: True)
        self.__derived_values: Dict[str, Any] = {}
        self.__pending_shrink_explanation = None

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

    def cached(self, *keys: Any) -> Callable[[Callable], Any]:
        def accept(f: Callable) -> Any:
            cache_key = (f.__name__, *keys)
            try:
                return self.cached_calculations[cache_key]
            except KeyError:
                return self.cached_calculations.setdefault(cache_key, f())

        return accept

    def add_new_pass(self, run: Callable) -> "ShrinkPass":
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

    def cached_test_function_ir(self, nodes: Sequence[ChoiceNode]) -> Optional[ConjectureResult]:
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

    def incorporate_test_data(self, data: Union[ConjectureData, ConjectureResult, _Overrun]) -> None:
        if data.status < Status.VALID or data is self.shrink_target:
            return
        if (
            self.__predicate(data)
            and sort_key(data.nodes) < sort_key(self.shrink_target.nodes)
            and self.__allow_transition(self.shrink_target, data)
        ):
            self.update_shrink_target(data)

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
                    "---------------------\n"
                    "Shrink pass profiling\n"
                    "---------------------\n\n"
                    f"Shrinking made a total of {calls} call{s(calls)} of which "
                    f"{self.shrinks} shrank and {misaligned} were misaligned. This deleted {total_deleted} choices out "
                    f"of {self.initial_size}."
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
                            f"  * {p.name} made {p.calls} call{s(p.calls)} of which "
                            f"{p.shrinks} shrank and {p.misaligned} were misaligned, "
                            f"deleting {p.deletions} choice{s(p.deletions)}."
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
        chunks: Dict[Tuple[int, int], List[Tuple[ChoiceT, ...]]] = defaultdict(list)

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
                        value = draw_choice(node.type, node.kwargs, random=self.random)
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
                        choices[:start] + result.choices[start:res_end] + choices[end:]
                    )
                    chunks[(start, end)].append(result.choices[start:res_end])
                    result = self.engine.cached_test_function_ir(attempt)

                    if result.status is Status.OVERRUN:
                        continue
                    result = cast(ConjectureResult, result)
                else:
                    chunks[(start, end)].append(result.choices[start:end])

                if shrink_target is not self.shrink_target:
                    self.shrink_target.slice_comments.clear()
                    return
                if result.status is Status.VALID:
                    break
                elif self.__predicate(result):
                    n_same_failures += 1
                    if n_same_failures >= 100:
                        self.shrink_target.slice_comments[(start, end)] = note
                        break

        if len(self.shrink_target.slice_comments) <= 1:
            return
        n_same_failures_together = 0
        chunks_by_start_index = sorted(chunks.items())
        for _ in range(500):
            new_choices: List[ChoiceT] = []
            prev_end = 0
            for (start, end), ls in chunks_by_start_index:
                assert prev_end <= start < end, "these chunks must be nonoverlapping"
                new_choices.extend(choices[prev_end:start])
                new_choices.extend(self.random.choice(ls))
                prev_end = end

            result = self.engine.cached_test_function_ir(new_choices)

            assert shrink_target is self.shrink_target
            if result.status == Status.VALID:
                self.shrink_target.slice_comments[(0, 0)] = (
                    "The test sometimes passed when commented parts were varied together."
                )
                break
            elif self.__predicate(result):
                n_same_failures_together += 1
                if n_same_failures_together >= 100:
                    self.shrink_target.slice_comments[(0, 0)] = (
                        "The test always failed when commented parts were varied together."
                    )
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
                and not node.was_forced
                and node.value <= 10
                and node.kwargs["min_value"] == 0
            ):
                assert isinstance(node.value, int)

                zero_attempt = self.cached_test_function_ir(
                    nodes[:i] + (nodes[i].copy(with_value=0),) + nodes[i + 1 :]
                )
                if (
                    zero_attempt is not self.shrink_target
                    and zero_attempt is not None
                    and zero_attempt.status >= Status.VALID
                ):
                    changed_shape = len(zero_attempt.nodes) != len(nodes)

                    if not changed_shape:
                        for j in range(i + 1, len(nodes)):
                            zero_node = zero_attempt.nodes[j]
                            orig_node = nodes[j]
                            if (
                                zero_node.type != orig_node.type
                                or not choice_permitted(
                                    orig_node.value, zero_node.kwargs
                                )
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
                self.consider_new_nodes(nodes[:i] + contents + nodes[initial_ex.end :])
                if initial is not self.shrink_target:
                    return True
        return False

    @derived_value
    def shrink_pass_choice_trees(self) -> Dict["ShrinkPass", ChoiceTree]:
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
                    initial_calls = self.calls
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
    def examples_by_label(self) -> Dict[str, List[ChoiceNode]]:
        examples_by_label = defaultdict(list)
        for ex in self.examples:
            examples_by_label[ex.label].append(ex)
        return dict(examples_by_label)

    @derived_value
    def distinct_labels(self) -> List[str]:
        return sorted(self.examples_by_label, key=str)

    @defines_shrink_pass()
    def pass_to_descendant(self, chooser: Any) -> None:
        label = chooser.choose(
            self.distinct_labels, lambda l: len(self.examples_by_label[l]) >= 2
        )

        ls = self.examples_by_label[label]
        i = chooser.choose(range(len(ls) - 1))
        ancestor = ls[i]

        if i + 1 == len(ls) or ls[i + 1].start >= ancestor.end:
            return

        @self.cached(label, i)
        def descendants() -> List[ChoiceNode]:
            lo = i + 1
            hi = len(ls)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if ls[mid].start >= ancestor.end:
                    hi = mid
                else:
                    lo = mid
            return [t for t in ls[i + 1 : hi] if t.choice_count < ancestor.choice_count]

        descendant = chooser.choose(descendants, lambda ex: ex.choice_count > 0)

        assert ancestor.start <= descendant.start
        assert ancestor.end >= descendant.end
        assert descendant.choice_count < ancestor.choice_count

        self.consider_new_nodes(
            self.nodes[: ancestor.start]
            + self.nodes[descendant.start : descendant.end]
            + self.nodes[ancestor.end :]
        )

    def lower_common_node_offset(self) -> None:
        if len(self.__changed_nodes) <= 1:
            return

        changed = []
        for i in sorted(self.__changed_nodes):
            node = self.nodes[i]
            if node.trivial or node.type != "integer":
                continue
            changed.append(node)

        if not changed:
            return

        ints = [abs(node.value - node.kwargs["shrink_towards"]) for node in changed]
        offset = min(ints)
        assert offset > 0

        for i in range(len(ints)):
            ints[i] -= offset

        st = self.shrink_target

        def offset_node(node: ChoiceNode, n: int) -> Tuple[int, int, List[ChoiceNode]]:
            return (
                node.index,
                node.index + 1,
                [node.copy(with_value=node.kwargs["shrink_towards"] + n)],
            )

        def consider(n: int, sign: int) -> bool:
            return self.consider_new_nodes(
                replace_all(
                    st.nodes,
                    [
                        offset_node(node, sign * (n + v))
                        for node, v in zip(changed, ints)
                    ],
                )
            )

        Integer.shrink(offset, lambda n: consider(n, 1))
        Integer.shrink(offset, lambda n: consider(n, -1))
        self.clear_change_tracking()

    def clear_change_tracking(self) -> None:
        self.__last_checked_changed_at = self.shrink_target
        self.__all_changed_nodes = set()

    def mark_changed(self, i: int) -> None:
        self.__changed_nodes.add(i)

    @property
    def __changed_nodes(self) -> Set[int]:
        if self.__last_checked_changed_at is self.shrink_target:
            return self.__all_changed_nodes

        prev_target = self.__last_checked_changed_at
        new_target = self.shrink_target
        assert prev_target is not new_target
        prev_nodes = prev_target.nodes
        new_nodes = new_target.nodes
        assert sort_key(new_target.nodes) < sort_key(prev_target.nodes)

        if len(prev_nodes) != len(new_nodes) or any(
            n1.type != n2.type for n1, n2 in zip(prev_nodes, new_nodes)
        ):
            self.__all_changed_nodes = set()
        else:
            assert len(prev_nodes) == len(new_nodes)
            for i, (n1, n2) in enumerate(zip(prev_nodes, new_nodes)):
                assert n1.type == n2.type
                if not choice_equal(n1.value, n2.value):
                    self.__all_changed_nodes.add(i)

        return self.__all_changed_nodes

    def update_shrink_target(self, new_target: ConjectureResult) -> None:
        assert isinstance(new_target, ConjectureResult)
        self.shrinks += 1
        self.max_stall = max(
            self.max_stall, (self.calls - self.calls_at_last_shrink) * 2
        )
        self.calls_at_last_shrink = self.calls
        self.shrink_target = new_target
        self.__derived_values = {}

    def try_shrinking_nodes(self, nodes: List[ChoiceNode], n: Any) -> bool:
        if any(node.index >= len(self.nodes) for node in nodes):
            return False

        initial_attempt = replace_all(
            self.nodes,
            [(node.index, node.index + 1, [node.copy(with_value=n)]) for node in nodes],
        )

        attempt = self.cached_test_function_ir(initial_attempt)

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
            (index, attempt_choice_type, attempt_kwargs, _attempt_forced) = (
                attempt.misaligned_at
            )
            node = self.nodes[index]
            if node.type != attempt_choice_type:
                return False
            if node.was_forced:
                return False

            if node.type in {"string", "bytes"}:
                if node.kwargs["min_size"] <= attempt_kwargs["min_size"]:
                    return False
                return self.consider_new_nodes(
                    initial_attempt[: node.index]
                    + [
                        initial_attempt[node.index].copy(
                            with_kwargs=attempt_kwargs,
                            with_value=initial_attempt[node.index].value[
                                : attempt_kwargs["min_size"]
                            ],
                        )
                    ]
                    + initial_attempt[node.index :]
                )

        lost_nodes = len(self.nodes) - len(attempt.nodes)
        if lost_nodes <= 0:
            return False

        start = nodes[0].index
        end = nodes[-1].index + 1
        regions_to_delete = {(end, end + lost_nodes)}

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

            regions_to_delete.add(
                (in_original[0].start, in_original[-len(in_replaced)].start)
            )

        for u, v in sorted(regions_to_delete, key=lambda x: x[1] - x[0], reverse=True):
            try_with_deleted = initial_attempt[:u] + initial_attempt[v:]
            if self.consider_new_nodes(try_with_deleted):
                return True

        return False

    def remove_discarded(self) -> bool:
        while self.shrink_target.has_discards:
            discarded = []

            for ex in self.shrink_target.examples:
                if (
                    ex.choice_count > 0
                    and ex.discarded
                    and (not discarded or ex.start >= discarded[-1][-1])
                ):
                    discarded.append((ex.start, ex.end))

            if not discarded:
                break

            attempt = list(self.nodes)
            for u, v in reversed(discarded):
                del attempt[u:v]

            if not self.consider_new_nodes(tuple(attempt)):
                return False
        return True

    @derived_value
    def duplicated_nodes(self) -> List[List[ChoiceNode]]:
        duplicates = defaultdict(list)
        for node in self.nodes:
            duplicates[(node.type, choice_key(node.value))].append(node)
        return list(duplicates.values())

    @defines_shrink_pass()
    def minimize_duplicated_nodes(self, chooser: Any) -> None:
        nodes = chooser.choose(self.duplicated_nodes)
        nodes = [node for node in nodes if not node.trivial]
        if len(nodes) <= 1:
            return

        self.minimize_nodes(nodes)

    @defines_shrink_pass()
    def redistribute_numeric_pairs(self, chooser: Any) -> None:
        def can_choose_node(node: ChoiceNode) -> bool:
            return node.type in {"integer", "float"} and not (
                node.type == "float"
                and (math.isnan(node.value) or abs(node.value) >= MAX_PRECISE_INTEGER)
            )

        node1 = chooser.choose(
            self.nodes,
            lambda node: can_choose_node(node) and not node.trivial,
        )
        node2 = chooser.choose(
            self.nodes,
            lambda node: can_choose_node(node)
            and not node.was_forced
            and node1.index < node.index <= node1.index + 4,
        )

        m: Union[int, float] = node1.value
        n: Union[int, float] = node2.value

        def boost(k: int) -> bool:
            if k > m:
                return False

            try:
                v1 = m - k
                v2 = n + k
            except OverflowError:
                return False

            if node1.type == "float" and v2 >= MAX_PRECISE_INTEGER:
                return False

            return self.consider_new_nodes(
                self.nodes[: node1.index]
                + (node1.copy(with_value=v1),)
                + self.nodes[node1.index + 1 : node2.index]
                + (node2.copy(with_value=v2),)
                + self.nodes[node2.index + 1 :]
            )

        find_integer(boost)

    @defines_shrink_pass()
    def lower_integers_together(self, chooser: Any) -> None:
        node1 = chooser.choose(
            self.nodes, lambda n: n.type == "integer" and not n.trivial
        )
        node2 = self.nodes[
            chooser.choose(
                range(node1.index + 1, min(len(self.nodes), node1.index + 3 + 1)),
                lambda i: self.nodes[i].type == "integer"
                and not self.nodes[i].was_forced,
            )
        ]

        shrink_towards = node1.kwargs["shrink_towards"]

        def consider(n: int) -> bool:
            return self.consider_new_nodes(
                self.nodes[: node1.index]
                + (node1.copy(with_value=node1.value - n),)
                + self.nodes[node1.index + 1 : node2.index]
                + (node2.copy(with_value=node2.value - n),)
                + self.nodes[node2.index + 1 :]
            )

        find_integer(lambda n: consider(shrink_towards - n))
        find_integer(lambda n: consider(n - shrink_towards))

    def minimize_nodes(self, nodes: List[ChoiceNode]) -> None:
        choice_type = nodes[0].type
        value = nodes[0].value
        kwargs = nodes[0].kwargs
        assert all(
            node.type == choice_type and choice_equal(node.value, value)
            for node in nodes
        )

        if choice_type == "integer":
            shrink_towards = kwargs["shrink_towards"]
            self.try_shrinking_nodes(nodes, abs(shrink_towards - value))
            Integer.shrink(
                abs(shrink_towards - value),
                lambda n: self.try_shrinking_nodes(nodes, shrink_towards + n),
            )
            Integer.shrink(
                abs(shrink_towards - value),
                lambda n: self.try_shrinking_nodes(nodes, shrink_towards - n),
            )
        elif choice_type == "float":
            self.try_shrinking_nodes(nodes, abs(value))
            Float.shrink(
                abs(value),
                lambda val: self.try_shrinking_nodes(nodes, val),
            )
            Float.shrink(
                abs(value),
                lambda val: self.try_shrinking_nodes(nodes, -val),
            )
        elif choice_type == "boolean":
            assert value is True
            self.try_shrinking_nodes(nodes, False)
        elif choice_type == "bytes":
            Bytes.shrink(
                value,
                lambda val: self.try_shrinking_nodes(nodes, val),
                min_size=kwargs["min_size"],
            )
        elif choice_type == "string":
            String.shrink(
                value,
                lambda val: self.try_shrinking_nodes(nodes, val),
                intervals=kwargs["intervals"],
                min_size=kwargs["min_size"],
            )
        else:
            raise NotImplementedError

    @defines_shrink_pass()
    def try_trivial_examples(self, chooser: Any) -> None:
        i = chooser.choose(range(len(self.examples)))

        prev = self.shrink_target
        nodes = self.shrink_target.nodes
        ex = self.examples[i]
        prefix = nodes[: ex.start]
        replacement = tuple(
            [
                (
                    node
                    if node.was_forced
                    else node.copy(
                        with_value=choice_from_index(0, node.type, node.kwargs)
                    )
                )
                for node in nodes[ex.start : ex.end]
            ]
        )
        suffix = nodes[ex.end :]
        attempt = self.cached_test_function_ir(prefix + replacement + suffix)

        if self.shrink_target is not prev:
            return

        if isinstance(attempt, ConjectureResult):
            new_ex = attempt.examples[i]
            new_replacement = attempt.nodes[new_ex.start : new_ex.end]
            self.consider_new_nodes(prefix + new_replacement + suffix)

    @defines_shrink_pass()
    def minimize_individual_nodes(self, chooser: Any) -> None:
        node = chooser.choose(self.nodes, lambda node: not node.trivial)
        initial_target = self.shrink_target

        self.minimize_nodes([node])
        if self.shrink_target is not initial_target:
            return

        if node.type != "integer":
            return

        lowered = (
            self.nodes[: node.index]
            + (node.copy(with_value=node.value - 1),)
            + self.nodes[node.index + 1 :]
        )
        attempt = self.cached_test_function_ir(lowered)
        if (
            attempt is None
            or attempt.status < Status.VALID
            or len(attempt.nodes) == len(self.nodes)
            or len(attempt.nodes) == node.index + 1
        ):
            return

        @self.cached(node.index)
        def first_example_after_node() -> int:
            lo = 0
            hi = len(self.examples)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                ex = self.examples[mid]
                if ex.start >= node.index:
                    hi = mid
                else:
                    lo = mid
            return hi

        if chooser.choose([True, False]):
            ex = self.examples[
                chooser.choose(
                    range(first_example_after_node, len(self.examples)),
                    lambda i: self.examples[i].choice_count > 0,
                )
            ]
            self.consider_new_nodes(lowered[: ex.start] + lowered[ex.end :])
        else:
            node = self.nodes[chooser.choose(range(node.index + 1, len(self.nodes)))]
            self.consider_new_nodes(lowered[: node.index] + lowered[node.index + 1 :])

    @defines_sh