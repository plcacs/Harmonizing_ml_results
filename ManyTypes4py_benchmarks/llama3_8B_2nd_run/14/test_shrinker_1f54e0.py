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
def test_can_shrink_variable_draws_with_just_deletion(n: int) -> None:
    # ... (rest of the function)

def test_deletion_and_lowering_fails_to_shrink(monkeypatch: pytest.MonkeyPatch) -> None:
    # ... (rest of the function)

def test_duplicate_nodes_that_go_away() -> None:
    # ... (rest of the function)

# ... (rest of the functions)

class BadShrinker(ShrinkerPass):
    """A shrinker that really doesn't do anything at all. This is mostly a covering test for the shrinker interface methods."""
    def run_step(self) -> None:
        return

@given(numeric_nodes: nodes, numeric_nodes: nodes, st.integers() | st.floats(allow_nan=False))
@example(ChoiceNode(type='float', value=float(MAX_PRECISE_INTEGER - 1), kwargs=float_kw(), was_forced=False), ChoiceNode(type='float', value=float(MAX_PRECISE_INTEGER - 1), kwargs=float_kw(), was_forced=False), 0)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_redistribute_numeric_pairs(node1: ChoiceNode, node2: ChoiceNode, stop: int) -> None:
    # ... (rest of the function)
