from copy import deepcopy
from typing import Any, Generator, Tuple
from eth2spec.test.context import spec_state_test, expect_assertion_error, with_all_phases
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.execution_payload import compute_el_block_hash_for_block
from eth2spec.test.helpers.forks import is_post_bellatrix
from eth2spec.test.helpers.state import next_slot

def prepare_state_for_header_processing(spec: Any, state: Any) -> None:
    spec.process_slots(state, state.slot + 1)

def run_block_header_processing(spec: Any, state: Any, block: Any, prepare_state: bool = True, valid: bool = True) -> Generator[Tuple[str, Any], None, None]:
    """
    Run ``process_block_header``, yielding:
      - pre-state ('pre')
      - block ('block')
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    if prepare_state:
        prepare_state_for_header_processing(spec, state)
    yield ('pre', state)
    yield ('block', block)
    if not valid:
        expect_assertion_error(lambda: spec.process_block_header(state, block))
        yield ('post', None)
        return
    spec.process_block_header(state, block)
    yield ('post', state)

@with_all_phases
@spec_state_test
def test_basic_block_header(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    yield from run_block_header_processing(spec, state, block)

@with_all_phases
@spec_state_test
def test_invalid_slot_block_header(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    block.slot = state.slot + 2
    yield from run_block_header_processing(spec, state, block, valid=False)

@with_all_phases
@spec_state_test
def test_invalid_proposer_index(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    active_indices = spec.get_active_validator_indices(state, spec.get_current_epoch(state))
    active_indices = [i for i in active_indices if i != block.proposer_index]
    block.proposer_index = active_indices[0]
    yield from run_block_header_processing(spec, state, block, valid=False)

@with_all_phases
@spec_state_test
def test_invalid_parent_root(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    block.parent_root = b'\n' * 32
    if is_post_bellatrix(spec):
        block.body.execution_payload.block_hash = compute_el_block_hash_for_block(spec, block)
    yield from run_block_header_processing(spec, state, block, valid=False)

@with_all_phases
@spec_state_test
def test_invalid_multiple_blocks_single_slot(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    prepare_state_for_header_processing(spec, state)
    spec.process_block_header(state, block)
    assert state.latest_block_header.slot == state.slot
    child_block = block.copy()
    child_block.parent_root = block.hash_tree_root()
    if is_post_bellatrix(spec):
        child_block.body.execution_payload.block_hash = compute_el_block_hash_for_block(spec, child_block)
    yield from run_block_header_processing(spec, state, child_block, prepare_state=False, valid=False)

@with_all_phases
@spec_state_test
def test_invalid_proposer_slashed(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    stub_state = deepcopy(state)
    next_slot(spec, stub_state)
    proposer_index = spec.get_beacon_proposer_index(stub_state)
    state.validators[proposer_index].slashed = True
    block = build_empty_block_for_next_slot(spec, state)
    yield from run_block_header_processing(spec, state, block, valid=False)