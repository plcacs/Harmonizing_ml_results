import time
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
def func_p3wk4n6h(n):

    @shrinking_from((n,) + (0,) * (n - 1) + (1,))
    def func_j5w0r7gm(data):
        n = data.draw_integer(0, 2 ** 4 - 1)
        b = [data.draw_integer(0, 2 ** 8 - 1) for _ in range(n)]
        if any(b):
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_individual_nodes'])
    assert shrinker.choices == (1, 1)


def func_r7eas21i(monkeypatch):
    monkeypatch.setattr(Shrinker, 'shrink', lambda self: self.
        fixate_shrink_passes(['minimize_individual_nodes']))
    monkeypatch.setattr(ConjectureRunner, 'generate_new_examples', lambda
        runner: runner.cached_test_function_ir((b'\x00',) * 10))

    @run_to_nodes
    def func_m6djgb9l(data):
        for _ in range(10):
            data.draw_bytes(1, 1)
        data.mark_interesting()
    assert tuple(n.value for n in nodes) == (b'\x00',) * 10


def func_ysmfp6s9():

    @shrinking_from((1234567, 1234567) + (b'\x01',) * (1234567 & 255))
    def func_j5w0r7gm(data):
        x = data.draw_integer(min_value=0)
        y = data.draw_integer(min_value=0)
        if x != y:
            data.mark_invalid()
        b = [data.draw_bytes(1, 1) for _ in range(x & 255)]
        if len(set(b)) <= 1:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_duplicated_nodes'])
    assert shrinker.shrink_target.choices == (0, 0)


def func_i6ssjenc():

    @shrinking_from((12, 12) + (b'\x02',) * 12)
    def func_j5w0r7gm(data):
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


def func_xn3c9zoa():

    @shrinking_from((3, 0, 0, 0, 1) * 10)
    def func_j5w0r7gm(data):
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


def func_44ufqhv2():

    def func_4yhskf6m(data):
        data.start_example(label=1)
        n = data.draw_integer(0, 1)
        data.draw_integer(0, 2 ** 8 - 1)
        if n:
            func_4yhskf6m(data)
            func_4yhskf6m(data)
        data.stop_example(discard=True)
    initial = 1, 10, 0, 0, 1, 0, 0, 10, 0, 0
    target = 0, 10
    good = {initial, target}

    @shrinking_from(initial)
    def func_j5w0r7gm(data):
        func_4yhskf6m(data)
        if data.choices in good:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['pass_to_descendant'])
    assert shrinker.choices == target


def func_kozsyz90():

    @shrinking_from((11, 10))
    def func_j5w0r7gm(data):
        m = data.draw_integer(0, 2 ** 8 - 1)
        n = data.draw_integer(0, 2 ** 8 - 1)
        if abs(m - n) <= 1 and max(m, n) > 0:
            data.mark_interesting()
    func_j5w0r7gm.mark_changed(0)
    func_j5w0r7gm.mark_changed(1)
    func_j5w0r7gm.lower_common_node_offset()
    assert shrinker.choices in {(0, 1), (1, 0)}


def func_dbe11ejh():

    @run_to_nodes
    def func_m6djgb9l(data):
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


def func_6flb4rel():

    @shrinking_from((1, 1, 1, 1, 0, 0, 0))
    def func_j5w0r7gm(data):
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


def func_mpeuuge1(monkeypatch):
    monkeypatch.setattr(ConjectureRunner, 'generate_new_examples', lambda
        runner: runner.cached_test_function_ir((1,)))
    monkeypatch.setattr(Shrinker, 'shrink', lambda self: self.
        consider_new_nodes(ir(2)))

    @run_to_nodes
    def func_m6djgb9l(data):
        data.draw_integer(0, 3)
        data.mark_interesting()
    assert tuple(n.value for n in nodes) == (1,)


def func_1u7fmu0w():

    @shrinking_from([v for i in range(5) for _ in range(i + 1) for v in [i]])
    def func_j5w0r7gm(data):
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


def func_uy16n4no():
    distribution = Sampler([4.0, 8.0, 1.0, 1.0, 0.5])
    sizes = [8, 16, 32, 64, 128]

    @shrinking_from((3, True, 65538, 1))
    def func_j5w0r7gm(data):
        size = sizes[distribution.sample(data)]
        result = data.draw_integer(0, 2 ** size - 1)
        sign = (-1) ** (result & 1)
        result = (result >> 1) * sign
        cap = data.draw_integer(0, 2 ** 8 - 1)
        if result >= 32768 and cap == 1:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_individual_nodes'])
    assert shrinker.choices == (1, True, 65536, 1)


def func_go73y8ul():

    def func_4yhskf6m(data):
        data.start_example(label=0)
        if not data.draw_boolean():
            result = 1, True
        else:
            h1, b1 = func_4yhskf6m(data)
            h2, b2 = func_4yhskf6m(data)
            result = 1 + max(h1, h2), b1 and b2 and abs(h1 - h2) <= 1
        data.stop_example()
        return result

    @shrinking_from((True,) * 5 + (False,) * 6)
    def func_j5w0r7gm(data):
        _, b = func_4yhskf6m(data)
        if not b:
            data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (True, False, True, False, True, False, False)


def func_5s7sa3e3():

    @shrinking_from((False,) * 1000 + (True,))
    def func_j5w0r7gm(data):
        while not data.draw_boolean():
            pass
        data.mark_interesting()
    p = func_j5w0r7gm.add_new_pass(node_program('X'))
    func_j5w0r7gm.fixate_shrink_passes([p.name])
    assert len(shrinker.choices) == 1
    assert shrinker.calls <= 60


def func_c3cuesoo():

    @shrinking_from((255,) * 100)
    def func_j5w0r7gm(data):
        any_nonzero = False
        for i in range(1, 10):
            any_nonzero |= data.draw_integer(0, 2 ** i - 1) > 0
        if not any_nonzero:
            data.mark_invalid()
        data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (0,) * 8 + (1,)


def func_ikxiwsli():

    @shrinking_from((1,) * 8)
    def func_j5w0r7gm(data):
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


def func_1hj495c2():

    @shrinking_from((255,) * 4)
    def func_j5w0r7gm(data):
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
def func_uvksw3tt(min_value, max_value, forced, shrink_towards, expected):

    @shrinking_from((forced,) * 2)
    def func_j5w0r7gm(data):
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


def func_tyv1lyp2():

    @shrinking_from((255,) * 6)
    def func_j5w0r7gm(data):
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


def func_64u37npi():

    @shrinking_from((1, 2, 3, 4, 5, 6, 0))
    def func_j5w0r7gm(data):
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


def func_tl1e8wci():

    @shrinking_from((255,) * 5)
    def func_j5w0r7gm(data):
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


def func_w1kblbss():

    @shrinking_from((1, 2, 3, 4, 0, 0))
    def func_j5w0r7gm(data):

        def func_igtxweb1():
            data.start_example(1)
            data.start_example(1)
            m = data.draw_integer(0, 2 ** 8 - 1)
            data.stop_example()
            data.start_example(1)
            n = data.draw_integer(0, 2 ** 8 - 1)
            data.stop_example()
            data.stop_example()
            return m, n
        v1 = func_igtxweb1()
        if v1 == (1, 2):
            if func_igtxweb1() != (3, 4):
                data.mark_invalid()
        if v1 == (0, 0) or func_igtxweb1() == (0, 0):
            data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.choices == (0, 0)


def func_t92qeuce():

    @shrinking_from((255,))
    def func_j5w0r7gm(data):
        data.draw_integer(0, 2 ** 8 - 1)
        data.mark_interesting()
    sp = func_j5w0r7gm.shrink_pass(node_program('X'))
    assert isinstance(sp, ShrinkPass)
    assert func_j5w0r7gm.shrink_pass(sp) is sp


def func_rdg9nqx4():
    time.freeze()

    @shrinking_from((255,) * 100)
    def func_j5w0r7gm(data):
        count = 0
        for _ in range(100):
            if data.draw_integer(0, 2 ** 8 - 1) != 255:
                count += 1
                if count >= 10:
                    return
        data.mark_interesting()
    func_j5w0r7gm.shrink()
    assert shrinker.calls <= 1 + 2 * shrinker.max_stall


def func_xi3ff43q():

    @shrinking_from(list(range(50)))
    def func_j5w0r7gm(data):
        for i in range(50):
            if data.draw_integer(0, 2 ** 8 - 1) != i:
                data.mark_invalid()
        data.mark_interesting()
    shrinker.max_stall = 5
    passes = [node_program('X' * i) for i in range(1, 11)]
    with pytest.raises(StopShrinking):
        func_j5w0r7gm.fixate_shrink_passes(passes)
    assert func_j5w0r7gm.shrink_pass(passes[-1]).calls > 0


@pytest.mark.parametrize('n_gap', [0, 1, 2])
def func_sl409ury(n_gap):

    @shrinking_from((1, 1) + (0,) * n_gap + (2,))
    def func_j5w0r7gm(data):
        if data.draw_integer(0, 2 ** 1 - 1) == 0:
            data.mark_invalid()
        m = data.draw_integer(0, 2 ** 8 - 1)
        for _ in range(n_gap):
            data.draw_integer(0, 2 ** 8 - 1)
        n = data.draw_integer(0, 2 ** 16 - 1)
        if n == m + 1:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['lower_integers_together'])
    assert shrinker.choices == (1, 0) + (0,) * n_gap + (1,)


def func_8ltsvmmz():

    @shrinking_from((15, 10))
    def func_j5w0r7gm(data):
        n1 = data.draw_integer(0, 100)
        n2 = data.draw_integer(0, 100, forced=10)
        if n1 + n2 > 20:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['redistribute_numeric_pairs'])
    assert shrinker.choices == (15, 10)


@pytest.mark.parametrize('n', [10, 50, 100, 200])
def func_qqgcdv4y(n):

    @shrinking_from([b'\x01' * n])
    def func_j5w0r7gm(data):
        b = data.draw_bytes()
        if len(b) >= n:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['minimize_individual_nodes'])
    assert shrinker.choices == (b'\x00' * n,)
    assert shrinker.calls < 10


def func_a4c8hu2u():
    seen_int = None

    @shrinking_from((1, b'hello world'))
    def func_j5w0r7gm(data):
        nonlocal seen_int
        i = data.draw_integer(min_value=0, max_value=1)
        if i == 1:
            if data.draw_bytes():
                data.mark_interesting()
        else:
            n = data.draw_integer(0, 100)
            if n == 0:
                return
            if seen_int is None:
                seen_int = n
            elif n != seen_int:
                data.mark_interesting()
    func_j5w0r7gm.initial_coarse_reduction()
    assert shrinker.choices[0] == 0


class BadShrinker(ShrinkerPass):
    """
    A shrinker that really doesn't do anything at all. This is mostly a covering
    test for the shrinker interface methods.
    """

    def func_3w12ocqs(self):
        return


def func_1mtys1ed():
    assert BadShrinker.shrink(10, lambda _: True) == 10


numeric_nodes = func_m6djgb9l(choice_types=['integer', 'float'])


@given(numeric_nodes, numeric_nodes, st.integers() | st.floats(allow_nan=False)
    )
@example(ChoiceNode(type='float', value=float(MAX_PRECISE_INTEGER - 1),
    kwargs=float_kw(), was_forced=False), ChoiceNode(type='float', value=
    float(MAX_PRECISE_INTEGER - 1), kwargs=float_kw(), was_forced=False), 0)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def func_cnkcaa6g(node1, node2, stop):
    assume(node1.value + node2.value > stop)
    assume(compute_max_children(node1.type, node1.kwargs) +
        compute_max_children(node2.type, node2.kwargs) > 2)

    @shrinking_from([node1.value, node2.value])
    def func_j5w0r7gm(data):
        v1 = getattr(data, f'draw_{node1.type}')(**node1.kwargs)
        v2 = getattr(data, f'draw_{node2.type}')(**node2.kwargs)
        if v1 + v2 > stop:
            data.mark_interesting()
    func_j5w0r7gm.fixate_shrink_passes(['redistribute_numeric_pairs'])
    assert len(shrinker.choices) == 2
    assert shrinker.choices[0] <= node1.value
    assert shrinker.choices[1] >= node2.value
