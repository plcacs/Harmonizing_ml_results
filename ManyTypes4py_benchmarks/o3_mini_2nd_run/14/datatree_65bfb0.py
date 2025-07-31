#!/usr/bin/env python3
from __future__ import annotations
import math
from collections.abc import Generator
from random import Random
from typing import TYPE_CHECKING, AbstractSet, Any, Dict, Final, List, Optional, Set, Tuple, Union, cast

import attr
from hypothesis.errors import FlakyReplay, FlakyStrategyDefinition, HypothesisException, StopTest
from hypothesis.internal import floats as flt
from hypothesis.internal.conjecture.choice import (
    BooleanKWargs,
    BytesKWargs,
    ChoiceKwargsT,
    ChoiceT,
    ChoiceTypeT,
    FloatKWargs,
    IntegerKWargs,
    StringKWargs,
    choice_from_index,
)
from hypothesis.internal.conjecture.data import ConjectureData, DataObserver, Status
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.floats import count_between_floats, float_to_int, int_to_float, sign_aware_lte

if TYPE_CHECKING:
    from hypothesis.vendor.pretty import RepresentationPrinter

ChildrenCacheValueT: Final = tuple[Generator[ChoiceT, None, None], List[ChoiceT], Set[ChoiceT]]

class PreviouslyUnseenBehaviour(HypothesisException):
    pass

_FLAKY_STRAT_MSG: Final[str] = 'Inconsistent data generation! Data generation behaved differently between different runs. Is your data generation depending on external state?'
EMPTY: Final[AbstractSet[Any]] = frozenset()

@attr.s(slots=True)
class Killed:
    next_node: TreeNode = attr.ib()

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        assert cycle is False
        p.text('Killed')

def _node_pretty(choice_type: str, value: Any, kwargs: Any, *, forced: bool) -> str:
    forced_marker = ' [forced]' if forced else ''
    return f'{choice_type} {value!r}{forced_marker} {kwargs}'

@attr.s(slots=True)
class Branch:
    kwargs: Any = attr.ib()
    choice_type: str = attr.ib()
    children: Dict[ChoiceT, TreeNode] = attr.ib(repr=False)

    @property
    def max_children(self) -> int:
        max_children = compute_max_children(self.choice_type, self.kwargs)
        assert max_children > 0
        return max_children

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        assert cycle is False
        for i, (value, child) in enumerate(self.children.items()):
            if i > 0:
                p.break_()
            p.text(_node_pretty(self.choice_type, value, self.kwargs, forced=False))
            with p.indent(2):
                p.break_()
                p.pretty(child)

@attr.s(slots=True, frozen=True)
class Conclusion:
    status: Status = attr.ib()
    interesting_origin: Optional[InterestingOrigin] = attr.ib()

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        assert cycle is False
        o = self.interesting_origin
        origin = '' if o is None else f', {o.exc_type.__name__} at {o.filename}:{o.lineno}'
        p.text(f'Conclusion ({self.status!r}{origin})')

MAX_CHILDREN_EFFECTIVELY_INFINITE: Final[int] = 100000

def _count_distinct_strings(*, alphabet_size: int, min_size: int, max_size: int) -> int:
    definitely_too_large = max_size * math.log(alphabet_size) > math.log(MAX_CHILDREN_EFFECTIVELY_INFINITE)
    if definitely_too_large:
        return MAX_CHILDREN_EFFECTIVELY_INFINITE
    return sum((alphabet_size ** k for k in range(min_size, max_size + 1)))

def compute_max_children(choice_type: str, kwargs: Any) -> int:
    if choice_type == 'integer':
        kwargs = cast(IntegerKWargs, kwargs)
        min_value = kwargs['min_value']
        max_value = kwargs['max_value']
        if min_value is None and max_value is None:
            return 2 ** 128 - 1
        if min_value is not None and max_value is not None:
            return max_value - min_value + 1
        assert (min_value is None) ^ (max_value is None)
        return 2 ** 127
    elif choice_type == 'boolean':
        kwargs = cast(BooleanKWargs, kwargs)
        p_val = kwargs['p']
        if p_val <= 2 ** (-64) or p_val >= 1 - 2 ** (-64):
            return 1
        return 2
    elif choice_type == 'bytes':
        kwargs = cast(BytesKWargs, kwargs)
        return _count_distinct_strings(alphabet_size=2 ** 8, min_size=kwargs['min_size'], max_size=kwargs['max_size'])
    elif choice_type == 'string':
        kwargs = cast(StringKWargs, kwargs)
        min_size = kwargs['min_size']
        max_size = kwargs['max_size']
        intervals = kwargs['intervals']
        if len(intervals) == 0:
            return 1
        if len(intervals) == 1 and max_size > MAX_CHILDREN_EFFECTIVELY_INFINITE:
            return MAX_CHILDREN_EFFECTIVELY_INFINITE
        return _count_distinct_strings(alphabet_size=len(intervals), min_size=min_size, max_size=max_size)
    elif choice_type == 'float':
        kwargs = cast(FloatKWargs, kwargs)
        min_value_f = kwargs['min_value']
        max_value_f = kwargs['max_value']
        smallest_nonzero_magnitude = kwargs['smallest_nonzero_magnitude']
        count = count_between_floats(min_value_f, max_value_f)
        min_point = max(min_value_f, -flt.next_down(smallest_nonzero_magnitude))
        max_point = min(max_value_f, flt.next_down(smallest_nonzero_magnitude))
        if min_point > max_point:
            return count
        count -= count_between_floats(min_point, max_point)
        if sign_aware_lte(min_value_f, -0.0) and sign_aware_lte(-0.0, max_value_f):
            count += 1
        if sign_aware_lte(min_value_f, 0.0) and sign_aware_lte(0.0, max_value_f):
            count += 1
        return count
    raise NotImplementedError(f'unhandled choice_type {choice_type}')

def _floats_between(a: float, b: float) -> Generator[float, None, None]:
    for n in range(float_to_int(a), float_to_int(b) + 1):
        yield int_to_float(n)

def all_children(choice_type: str, kwargs: Any) -> Generator[ChoiceT, None, None]:
    if choice_type != 'float':
        for index in range(compute_max_children(choice_type, kwargs)):
            yield choice_from_index(index, choice_type, kwargs)
    else:
        kwargs = cast(FloatKWargs, kwargs)
        min_value = kwargs['min_value']
        max_value = kwargs['max_value']
        smallest_nonzero_magnitude = kwargs['smallest_nonzero_magnitude']
        if sign_aware_lte(min_value, -0.0) and sign_aware_lte(-0.0, max_value):
            yield (-0.0)
        if sign_aware_lte(min_value, 0.0) and sign_aware_lte(0.0, max_value):
            yield 0.0
        if flt.is_negative(min_value):
            if flt.is_negative(max_value):
                max_point = min(max_value, -smallest_nonzero_magnitude)
                yield from _floats_between(max_point, min_value)
            else:
                yield from _floats_between(-smallest_nonzero_magnitude, min_value)
                yield from _floats_between(smallest_nonzero_magnitude, max_value)
        else:
            min_point = max(min_value, smallest_nonzero_magnitude)
            yield from _floats_between(min_point, max_value)

@attr.s(slots=True)
class TreeNode:
    kwargs: List[Any] = attr.ib(factory=list)
    values: List[Any] = attr.ib(factory=list)
    choice_types: List[str] = attr.ib(factory=list)
    __forced: Optional[Set[int]] = attr.ib(default=None, init=False)
    transition: Optional[Union[Branch, Conclusion, Killed]] = attr.ib(default=None)
    is_exhausted: bool = attr.ib(default=False, init=False)

    @property
    def forced(self) -> AbstractSet[int]:
        if not self.__forced:
            return EMPTY  # type: ignore
        return self.__forced

    def mark_forced(self, i: int) -> None:
        assert 0 <= i < len(self.values)
        if self.__forced is None:
            self.__forced = set()
        self.__forced.add(i)

    def split_at(self, i: int) -> None:
        if i in self.forced:
            raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
        assert not self.is_exhausted
        key = self.values[i]
        child = TreeNode(
            choice_types=self.choice_types[i + 1:],
            kwargs=self.kwargs[i + 1:],
            values=self.values[i + 1:],
            transition=self.transition,
        )
        self.transition = Branch(kwargs=self.kwargs[i], choice_type=self.choice_types[i], children={key: child})
        if self.__forced is not None:
            child.__forced = {j - i - 1 for j in self.__forced if j > i}
            self.__forced = {j for j in self.__forced if j < i}
        child.check_exhausted()
        del self.choice_types[i:]
        del self.values[i:]
        del self.kwargs[i:]
        assert len(self.values) == len(self.kwargs) == len(self.choice_types) == i

    def check_exhausted(self) -> bool:
        if not self.is_exhausted and self.transition is not None and (len(self.forced) == len(self.values)):
            if isinstance(self.transition, (Conclusion, Killed)):
                self.is_exhausted = True
            elif len(self.transition.children) == self.transition.max_children:
                self.is_exhausted = all((v.is_exhausted for v in self.transition.children.values()))
        return self.is_exhausted

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        assert cycle is False
        indent = 0
        for i, (choice_type, kwargs, value) in enumerate(zip(self.choice_types, self.kwargs, self.values)):
            with p.indent(indent):
                if i > 0:
                    p.break_()
                p.text(_node_pretty(choice_type, value, kwargs, forced=(i in self.forced)))
            indent += 2
        with p.indent(indent):
            if len(self.values) > 0:
                p.break_()
            if self.transition is not None:
                p.pretty(self.transition)
            else:
                p.text('unknown')

class DataTree:
    def __init__(self) -> None:
        self.root: TreeNode = TreeNode()
        self._children_cache: Dict[int, Tuple[Generator[ChoiceT, None, None], List[ChoiceT], Set[ChoiceT]]] = {}

    @property
    def is_exhausted(self) -> bool:
        return self.root.is_exhausted

    def generate_novel_prefix(self, random: Random) -> Tuple[Any, ...]:
        assert not self.is_exhausted
        prefix: List[Any] = []

        def append_choice(choice_type: str, choice: Any) -> None:
            if choice_type == 'float':
                assert isinstance(choice, int)
                choice_converted = int_to_float(choice)
            else:
                choice_converted = choice
            prefix.append(choice_converted)

        current_node: TreeNode = self.root
        while True:
            assert not current_node.is_exhausted
            for i, (choice_type, kwargs, value) in enumerate(zip(current_node.choice_types, current_node.kwargs, current_node.values)):
                if i in current_node.forced:
                    append_choice(choice_type, value)
                else:
                    attempts = 0
                    while True:
                        if attempts <= 10:
                            try:
                                node_value: Any = self._draw(choice_type, kwargs, random=random)
                            except StopTest:
                                attempts += 1
                                continue
                        else:
                            node_value = self._draw_from_cache(choice_type, kwargs, key=id(current_node), random=random)
                        if node_value != value:
                            append_choice(choice_type, node_value)
                            break
                        attempts += 1
                        self._reject_child(choice_type, kwargs, child=node_value, key=id(current_node))
                    return tuple(prefix)
            else:
                assert not isinstance(current_node.transition, (Conclusion, Killed))
                if current_node.transition is None:
                    return tuple(prefix)
                branch = current_node.transition
                assert isinstance(branch, Branch)
                attempts = 0
                while True:
                    if attempts <= 10:
                        try:
                            node_value = self._draw(branch.choice_type, branch.kwargs, random=random)
                        except StopTest:
                            attempts += 1
                            continue
                    else:
                        node_value = self._draw_from_cache(branch.choice_type, branch.kwargs, key=id(branch), random=random)
                    try:
                        child = branch.children[node_value]
                    except KeyError:
                        append_choice(branch.choice_type, node_value)
                        return tuple(prefix)
                    if not child.is_exhausted:
                        append_choice(branch.choice_type, node_value)
                        current_node = child
                        break
                    attempts += 1
                    self._reject_child(branch.choice_type, branch.kwargs, child=node_value, key=id(branch))
                    assert attempts != 1000 or len(branch.children) < branch.max_children or any((not v.is_exhausted for v in branch.children.values()))

    def rewrite(self, choices: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], Optional[Status]]:
        data: ConjectureData = ConjectureData.for_choices(choices)
        try:
            self.simulate_test_function(data)
            return (data.choices, data.status)
        except PreviouslyUnseenBehaviour:
            return (choices, None)

    def simulate_test_function(self, data: ConjectureData) -> None:
        node: TreeNode = self.root

        def draw(choice_type: str, kwargs: Any, *, forced: Optional[Any] = None, convert_forced: bool = True) -> Any:
            if choice_type == 'float' and forced is not None and convert_forced:
                forced = int_to_float(forced)
            draw_func = getattr(data, f'draw_{choice_type}')
            value = draw_func(**kwargs, forced=forced)
            if choice_type == 'float':
                value = float_to_int(value)
            return value
        try:
            while True:
                for i, (choice_type, kwargs, previous) in enumerate(zip(node.choice_types, node.kwargs, node.values)):
                    v = draw(choice_type, kwargs, forced=previous if i in node.forced else None)
                    if v != previous:
                        raise PreviouslyUnseenBehaviour
                if isinstance(node.transition, Conclusion):
                    t = node.transition
                    data.conclude_test(t.status, t.interesting_origin)
                elif node.transition is None:
                    raise PreviouslyUnseenBehaviour
                elif isinstance(node.transition, Branch):
                    v = draw(node.transition.choice_type, node.transition.kwargs)
                    try:
                        node = node.transition.children[v]
                    except KeyError as err:
                        raise PreviouslyUnseenBehaviour from err
                else:
                    assert isinstance(node.transition, Killed)
                    data.observer.kill_branch()
                    node = node.transition.next_node
        except StopTest:
            pass

    def new_observer(self) -> TreeRecordingObserver:
        return TreeRecordingObserver(self)

    def _draw(self, choice_type: str, kwargs: Any, *, random: Random) -> ChoiceT:
        from hypothesis.internal.conjecture.data import draw_choice
        value: Any = draw_choice(choice_type, kwargs, random=random)
        if choice_type == 'float':
            value = float_to_int(value)
        return value

    def _get_children_cache(
        self, choice_type: str, kwargs: Any, *, key: int
    ) -> Tuple[Generator[ChoiceT, None, None], List[ChoiceT], Set[ChoiceT]]:
        if key not in self._children_cache:
            generator: Generator[ChoiceT, None, None] = all_children(choice_type, kwargs)
            children: List[ChoiceT] = []
            rejected: Set[ChoiceT] = set()
            self._children_cache[key] = (generator, children, rejected)
        return self._children_cache[key]

    def _draw_from_cache(self, choice_type: str, kwargs: Any, *, key: int, random: Random) -> ChoiceT:
        generator, children, rejected = self._get_children_cache(choice_type, kwargs, key=key)
        if len(children) < 100:
            for v in generator:
                if choice_type == 'float':
                    assert isinstance(v, float)
                    v = float_to_int(v)
                if v in rejected:
                    continue
                children.append(v)
                if len(children) >= 100:
                    break
        return random.choice(children)

    def _reject_child(self, choice_type: str, kwargs: Any, *, child: ChoiceT, key: int) -> None:
        _generator, children, rejected = self._get_children_cache(choice_type, kwargs, key=key)
        rejected.add(child)
        if child in children:
            children.remove(child)

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        assert cycle is False
        p.pretty(self.root)

class TreeRecordingObserver(DataObserver):
    def __init__(self, tree: DataTree) -> None:
        self.__current_node: TreeNode = tree.root
        self.__index_in_current_node: int = 0
        self.__trail: List[TreeNode] = [self.__current_node]
        self.killed: bool = False

    def draw_integer(self, value: Any, *, was_forced: bool, kwargs: Dict[str, Any]) -> None:
        self.draw_value('integer', value, was_forced=was_forced, kwargs=kwargs)

    def draw_float(self, value: Any, *, was_forced: bool, kwargs: Dict[str, Any]) -> None:
        self.draw_value('float', value, was_forced=was_forced, kwargs=kwargs)

    def draw_string(self, value: Any, *, was_forced: bool, kwargs: Dict[str, Any]) -> None:
        self.draw_value('string', value, was_forced=was_forced, kwargs=kwargs)

    def draw_bytes(self, value: Any, *, was_forced: bool, kwargs: Dict[str, Any]) -> None:
        self.draw_value('bytes', value, was_forced=was_forced, kwargs=kwargs)

    def draw_boolean(self, value: Any, *, was_forced: bool, kwargs: Dict[str, Any]) -> None:
        self.draw_value('boolean', value, was_forced=was_forced, kwargs=kwargs)

    def draw_value(self, choice_type: str, value: Any, *, was_forced: bool, kwargs: Dict[str, Any]) -> None:
        i: int = self.__index_in_current_node
        self.__index_in_current_node += 1
        node: TreeNode = self.__current_node
        if isinstance(value, float):
            value = float_to_int(value)
        assert len(node.kwargs) == len(node.values) == len(node.choice_types)
        if i < len(node.values):
            if choice_type != node.choice_types[i] or kwargs != node.kwargs[i]:
                raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
            if was_forced and i not in node.forced:
                raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
            if value != node.values[i]:
                node.split_at(i)
                assert i == len(node.values)
                new_node = TreeNode()
                assert isinstance(node.transition, Branch)
                node.transition.children[value] = new_node
                self.__current_node = new_node
                self.__index_in_current_node = 0
        else:
            trans = node.transition
            if trans is None:
                node.choice_types.append(choice_type)
                node.kwargs.append(kwargs)
                node.values.append(value)
                if was_forced:
                    node.mark_forced(i)
                if compute_max_children(choice_type, kwargs) == 1 and (not was_forced):
                    node.split_at(i)
                    assert isinstance(node.transition, Branch)
                    self.__current_node = node.transition.children[value]
                    self.__index_in_current_node = 0
            elif isinstance(trans, Conclusion):
                assert trans.status != Status.OVERRUN
                raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
            else:
                assert isinstance(trans, Branch), trans
                if choice_type != trans.choice_type or kwargs != trans.kwargs:
                    raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
                try:
                    self.__current_node = trans.children[value]
                except KeyError:
                    self.__current_node = trans.children.setdefault(value, TreeNode())
                self.__index_in_current_node = 0
        if self.__trail[-1] is not self.__current_node:
            self.__trail.append(self.__current_node)

    def kill_branch(self) -> None:
        if self.killed:
            return
        self.killed = True
        if self.__index_in_current_node < len(self.__current_node.values) or (self.__current_node.transition is not None and (not isinstance(self.__current_node.transition, Killed))):
            raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
        if self.__current_node.transition is None:
            self.__current_node.transition = Killed(TreeNode())
            self.__update_exhausted()
        self.__current_node = self.__current_node.transition.next_node
        self.__index_in_current_node = 0
        self.__trail.append(self.__current_node)

    def conclude_test(self, status: Status, interesting_origin: Optional[InterestingOrigin]) -> None:
        if status == Status.OVERRUN:
            return
        i: int = self.__index_in_current_node
        node: TreeNode = self.__current_node
        if i < len(node.values) or isinstance(node.transition, Branch):
            raise FlakyStrategyDefinition(_FLAKY_STRAT_MSG)
        new_transition: Conclusion = Conclusion(status, interesting_origin)
        if node.transition is not None and node.transition != new_transition:
            if isinstance(node.transition, Conclusion) and (node.transition.status != Status.INTERESTING or new_transition.status != Status.VALID):
                old_origin = node.transition.interesting_origin
                new_origin = new_transition.interesting_origin
                raise FlakyReplay(f'Inconsistent results from replaying a test case!\n  last: {node.transition.status.name} from {old_origin}\n  this: {new_transition.status.name} from {new_origin}', (old_origin, new_origin))
        else:
            node.transition = new_transition
        assert node is self.__trail[-1]
        node.check_exhausted()
        assert len(node.values) > 0 or node.check_exhausted()
        if not self.killed:
            self.__update_exhausted()

    def __update_exhausted(self) -> None:
        for t in reversed(self.__trail):
            if not t.check_exhausted():
                break
