import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pytest
from hypothesis import HealthCheck, assume, example, given, settings, strategies as st
from hypothesis.internal.conjecture.data import ChoiceNode, ConjectureData
from hypothesis.internal.conjecture.datatree import compute_max_children
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.shrinker import Shrinker, ShrinkPass, StopShrinking, node_program
from hypothesis.internal.conjecture.shrinking.common import Shrinker as ShrinkerPass
from hypothesis.internal.conjecture.utils import Sampler
from hypothesis.internal.floats import MAX_PRECISE_INTEGER
from tests.conjecture.common import SOME_LABEL, float_kw, interesting_origin, ir, nodes, run_to_nodes, shrinking_from


@pytest.mark.parametrize('n', [1, 5, 8, 15])
def func_p3wk4n6h(n: int) -> None:

    @shrinking_from((n,) + (0,) * (n - 1) + (1,))
    def func_j5w0r7gm(data: ConjectureData) -> None:
        n = data.draw_integer(0, 2 ** 4 - 1)
        b = [data.draw_integer(0, 2 ** 8 - 1) for _ in range(n)]
        if any(b):
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_individual_nodes'])
    assert shrinker.choices == (1, 1)


def func_r7eas21i(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Shrinker, 'shrink', lambda self: self.
        fixate_shrink_passes(['minimize_individual_nodes']))
    monkeypatch.setattr(ConjectureRunner, 'generate_new_examples', lambda
        runner: runner.cached_test_function_ir((b'\x00',) * 10))

    @run_to_nodes
    def func_m6djgb9l(data: ConjectureData) -> None:
        for _ in range(10):
            data.draw_bytes(1, 1)
        data.mark_interesting()
    assert tuple(n.value for n in nodes) == (b'\x00',) * 10


def func_ysmfp6s9() -> None:

    @shrinking_from((1234567, 1234567) + (b'\x01',) * (1234567 & 255))
    def func_j5w0r7gm(data: ConjectureData) -> None:
        x = data.draw_integer(min_value=0)
        y = data.draw_integer(min_value=0)
        if x != y:
            data.mark_invalid()
        b = [data.draw_bytes(1, 1) for _ in range(x & 255)]
        if len(set(b)) <= 1:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_duplicated_nodes'])
    assert shrinker.shrink_target.choices == (0, 0)


def func_i6ssjenc() -> None:

    @shrinking_from((12, 12) + (b'\x02',) * 12)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        x = data.draw_integer(0, 2 ** 8 - 1)
        y = data.draw_integer(0, 2 ** 8 - 1)
        if x != y:
            data.mark_invalid()
        if x < 5:
            data.mark_invalid()
        b = [data.draw_bytes(1, 1) for _ in range(x)]
        if len(set(b)) == 1:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_duplicated_nodes'])
    print(shrinker.choices)
    assert shrinker.choices == (5, 5, *([b'\x00'] * 5))


def func_xn3c9zoa() -> None:

    @shrinking_from((3, 0, 0, 0, 1) * 10)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        for _ in range(10):
            data.start_example(SOME_LABEL)
            n = data.draw_integer(0, 2 ** 8 - 1)
            for _ in range(n):
                data.draw_integer(0, 2 ** 8 - 1)
            data.stop_example()
            if data.draw_integer(0, 2 ** 8 - 1) != 1:
                return
        data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (0, 1) * 10


def func_44ufqhv2() -> None:

    def func_4yhskf6m(data: ConjectureData) -> None:
        data.start_example(label=1)
        n = data.draw_integer(0, 1)
        data.draw_integer(0, 2 ** 8 - 1)
        if n:
            func_4yhskf6m(data)
            func_4yhskf6m(data)
        data.stop_example(discard=True)
    initial = (1, 10, 0, 0, 1, 0, 0, 10, 0, 0)
    target = (0, 10)
    good = {initial, target}

    @shrinking_from(initial)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        func_4yhskf6m(data)
        if data.choices in good:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['pass_to_descendant'])
    assert shrinker.choices == target


def func_kozsyz90() -> None:

    @shrinking_from((11, 10))
    def func_j5w0r7gm(data: ConjectureData) -> None:
        m = data.draw_integer(0, 2 ** 8 - 1)
        n = data.draw_integer(0, 2 ** 8 - 1)
        if abs(m - n) <= 1 and max(m, n) > 0:
            data.mark_interesting()
    func_j5w0r7gm.mark_changed(0)
    func_j5w0r7gm.mark_changed(1)
    func_j5w0r7gm.lower_common_node_offset()
    assert shrinker.choices in {(0, 1), (1, 0)}


def func_dbe11ejh() -> None:

    @run_to_nodes
    def func_m6djgb9l(data: ConjectureData) -> None:
        while True:
            data.start_example(SOME_LABEL)
            n = data.draw_integer(0, 1)
            data.start_example(SOME_LABEL)
            data.stop_example()
            data.stop_example(discard=n > 0)
            if not n:
                break
        data.mark_interesting()
    assert tuple(n.value for n in nodes) == (0,)


def func_6flb4rel() -> None:

    @shrinking_from((1, 1, 1, 1, 0, 0, 0))
    def func_j5w0r7gm(data: ConjectureData) -> None:
        total = 0
        for _ in range(5):
            data.start_example(label=0)
            if data.draw_integer(0, 2 ** 8 - 1):
                total += data.draw_integer(0, 2 ** 9 - 1)
            data.stop_example()
        if total == 2:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['reorder_examples'])
    assert shrinker.choices == (0, 0, 0, 1, 1, 1, 1)


def func_mpeuuge1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ConjectureRunner, 'generate_new_examples', lambda
        runner: runner.cached_test_function_ir((1,)))
    monkeypatch.setattr(Shrinker, 'shrink', lambda self: self.
        consider_new_nodes(ir(2)))

    @run_to_nodes
    def func_m6djgb9l(data: ConjectureData) -> None:
        data.draw_integer(0, 3)
        data.mark_interesting()
    assert tuple(n.value for n in nodes) == (1,)


def func_1u7fmu0w() -> None:

    @shrinking_from([v for i in range(5) for _ in range(i + 1) for v in [i]])
    def func_j5w0r7gm(data: ConjectureData) -> None:
        while True:
            n = data.draw_integer(0, 2 ** 16 - 1)
            for _ in range(n):
                if data.draw_integer(0, 2 ** 16 - 1) != n:
                    data.mark_invalid()
            if n == 4:
                data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes([node_program('X' * i) for i in
        range(1, 5)])
    assert shrinker.choices == (4,) * 5


def func_uy16n4no() -> None:
    distribution = Sampler([4.0, 8.0, 1.0, 1.0, 0.5])
    sizes = [8, 16, 32, 64, 128]

    @shrinking_from((3, True, 65538, 1))
    def func_j5w0r7gm(data: ConjectureData) -> None:
        size = sizes[distribution.sample(data)]
        result = data.draw_integer(0, 2 ** size - 1)
        sign = (-1) ** (result & 1)
        result = (result >> 1) * sign
        cap = data.draw_integer(0, 2 ** 8 - 1)
        if result >= 32768 and cap == 1:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_individual_nodes'])
    assert shrinker.choices == (1, True, 65536, 1)


def func_go73y8ul() -> None:

    def func_4yhskf6m(data: ConjectureData) -> Tuple[int, bool]:
        data.start_example(label=0)
        if not data.draw_boolean():
            result = (1, True)
        else:
            h1, b1 = func_4yhskf6m(data)
            h2, b2 = func_4yhskf6m(data)
            result = (1 + max(h1, h2), b1 and b2 and abs(h1 - h2) <= 1)
        data.stop_example()
        return result

    @shrinking_from((True,) * 5 + (False,) * 6)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        _, b = func_4yhskf6m(data)
        if not b:
            data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (True, False, True, False, True, False, False)


def func_5s7sa3e3() -> None:

    @shrinking_from((False,) * 1000 + (True,))
    def func_j5w0r7gm(data: ConjectureData) -> None:
        while not data.draw_boolean():
            pass
        data.mark_interesting()
    p = func_j5w0r7gm.add_new_pass(node_program('X'))
    func_j5w0r7gm.fixate_shrink_passes([p.name])
    assert len(shrinker.choices) == 1
    assert shrinker.calls <= 60


def func_c3cuesoo() -> None:

    @shrinking_from((255,) * 100)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        any_nonzero = False
        for i in range(1, 10):
            any_nonzero |= data.draw_integer(0, 2 ** i - 1) > 0
        if not any_nonzero:
            data.mark_invalid()
        data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (0,) * 8 + (1,)


def func_ikxiwsli() -> None:

    @shrinking_from((1,) * 8)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        for _ in range(4):
            data.start_example(1)
            if data.draw_integer(0, 2 ** 8 - 1) == 0:
                data.mark_invalid()
            data.start_example(1)
            data.draw_integer(0, 2 ** 8 - 1)
            data.stop_example()
            data.stop_example()
        data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (1, 0) * 4


def func_1hj495c2() -> None:

    @shrinking_from((255,) * 4)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        m = data.draw_integer(0, 2 ** 16 - 1)
        n = data.draw_integer(0, 2 ** 16 - 1)
        if m == 0 or n == 0:
            data.mark_invalid()
        if abs(m - n) <= 1:
            data.mark_interesting(interesting_origin(0))
        if abs(m - n) <= 10:
            data.mark_interesting(interesting_origin(1))
    func_j5w0r7gm.fixate_shrink_passes(['minimize_individual_nodes'])
    assert shrinker.engine.valid_examples <= 100
    assert shrinker.choices == (1, 1)


@pytest.mark.parametrize(
    'min_value, max_value, forced, shrink_towards, expected', [(-100, 0, -
    100, 0, (-10, -10)), (-100, 0, -100, -35, (-25, -25)), (0, 100, 100, 0,
    (10, 10)), (0, 100, 100, 65, (75, 75))])
def func_uvksw3tt(min_value: int, max_value: int, forced: int, shrink_towards: int, expected: Tuple[int, int]) -> None:

    @shrinking_from((forced,) * 2)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        m = data.draw_integer(min_value, max_value, shrink_towards=
            shrink_towards)
        n = data.draw_integer(min_value, max_value, shrink_towards=
            shrink_towards)
        if abs(m - shrink_towards) < 10 or abs(n - shrink_towards) < 10:
            data.mark_invalid()
        if abs(m - n) <= 1:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_individual_nodes'])
    assert shrinker.engine.valid_examples <= 40
    assert shrinker.choices == expected


def func_tyv1lyp2() -> None:

    @shrinking_from((255,) * 6)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        data.start_example(1)
        data.draw_integer(0, 2 ** 8 - 1)
        data.draw_integer(0, 2 ** 16 - 1)
        data.stop_example()
        data.start_example(1)
        interesting = data.draw_integer(0, 2 ** 8 - 1
            ) > 0 and data.draw_integer(0, 2 ** 16 - 1) > 0
        data.stop_example()
        if interesting:
            data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (0,) * 2 + (1, 1)


def func_64u37npi() -> None:

    @shrinking_from((1, 2, 3, 4, 5, 6, 0))
    def func_j5w0r7gm(data: ConjectureData) -> None:
        interesting = False
        while True:
            n = data.draw_integer(0, 2 ** 8 - 1)
            if n == 6:
                interesting = True
            if n == 0:
                break
        if interesting:
            data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (6, 0)


def func_tl1e8wci() -> None:

    @shrinking_from((255,) * 5)
    def func_j5w0r7gm(data: ConjectureData) -> None:
        seen_non_zero = False
        for _ in range(5):
            if data.draw_integer(0, 2 ** 8 - 1) == 0:
                if seen_non_zero:
                    data.mark_invalid()
            else:
                seen_non_zero = True
        data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (0,) * 5


def func_w1kblbss() -> None:

    @shrinking_from((1, 2, 3, 4, 0, 0))
    def func_j5w0r7gm(data: ConjectureData) -> None:

        def func_igtxweb1() -> Tuple