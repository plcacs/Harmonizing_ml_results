import pytest
from hypothesis import HealthCheck, settings
from hypothesis.internal.conjecture.data import ChoiceNode, ConjectureData
from hypothesis.internal.conjecture.shrinker import Shrinker, ShrinkPass, StopShrinking
from hypothesis.internal.conjecture.utils import Sampler
from tests.conjecture.common import float_kw, nodes, run_to_nodes, shrinking_from

@pytest.mark.parametrize('n', [1, 5, 8, 15])
def test_can_shrink_variable_draws_with_just_deletion(n: int) -> None:
    ...

def test_deletion_and_lowering_fails_to_shrink(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

def test_duplicate_nodes_that_go_away() -> None:
    ...

def test_accidental_duplication() -> None:
    ...

def test_can_zero_subintervals() -> None:
    ...

def test_can_pass_to_an_indirect_descendant() -> None:
    ...

def test_shrinking_blocks_from_common_offset() -> None:
    ...

def test_handle_empty_draws() -> None:
    ...

def test_can_reorder_examples() -> None:
    ...

def test_permits_but_ignores_raising_order(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

def test_block_deletion_can_delete_short_ranges() -> None:
    ...

def test_dependent_block_pairs_is_up_to_shrinking_integers() -> None:
    ...

def test_finding_a_minimal_balanced_binary_tree() -> None:
    ...

def test_node_programs_are_adaptive() -> None:
    ...

def test_zero_examples_with_variable_min_size() -> None:
    ...

def test_zero_contained_examples() -> None:
    ...

def test_zig_zags_quickly() -> None:
    ...

@pytest.mark.parametrize('min_value, max_value, forced, shrink_towards, expected', [(-100, 0, -100, 0, (-10, -10)), (-100, 0, -100, -35, (-25, -25)), (0, 100, 100, 0, (10, 10)), (0, 100, 100, 65, (75, 75))])
def test_zig_zags_quickly_with_shrink_towards(min_value: int, max_value: int, forced: int, shrink_towards: int, expected: tuple[int, int]) -> None:
    ...

def test_zero_irregular_examples() -> None:
    ...

def test_retain_end_of_buffer() -> None:
    ...

def test_can_expand_zeroed_region() -> None:
    ...

def test_can_expand_deleted_region() -> None:
    ...

def test_shrink_pass_method_is_idempotent() -> None:
    ...

def test_will_terminate_stalled_shrinks() -> None:
    ...

def test_will_let_fixate_shrink_passes_do_a_full_run_through() -> None:
    ...

@pytest.mark.parametrize('n_gap', [0, 1, 2])
def test_can_simultaneously_lower_non_duplicated_nearby_integers(n_gap: int) -> None:
    ...

def test_redistribute_with_forced_node_integer() -> None:
    ...

@pytest.mark.parametrize('n', [10, 50, 100, 200])
def test_can_quickly_shrink_to_trivial_collection(n: int) -> None:
    ...

def test_alternative_shrinking_will_lower_to_alternate_value() -> None:
    ...

class BadShrinker(ShrinkerPass):
    def run_step(self) -> None:
        ...

@given(numeric_nodes, numeric_nodes, st.integers() | st.floats(allow_nan=False))
@example(ChoiceNode(type='float', value=float(MAX_PRECISE_INTEGER - 1), kwargs=float_kw(), was_forced=False), ChoiceNode(type='float', value=float(MAX_PRECISE_INTEGER - 1), kwargs=float_kw(), was_forced=False), 0)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_redistribute_numeric_pairs(node1: ChoiceNode, node2: ChoiceNode, stop: Union[int, float]) -> None:
    ...