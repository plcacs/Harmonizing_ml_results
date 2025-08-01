from typing import Any

from eth2spec.test.context import with_all_phases, spec_state_test
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.attestations import get_valid_attestation, sign_attestation
from eth2spec.test.helpers.constants import ALL_PHASES
from eth2spec.test.helpers.forks import is_post_electra
from eth2spec.test.helpers.state import transition_to, state_transition_and_sign_block, next_epoch, next_slot
from eth2spec.test.helpers.fork_choice import get_genesis_forkchoice_store

def run_on_attestation(spec: Any, state: Any, store: Any, attestation: Any, valid: bool = True) -> None:
    if not valid:
        try:
            spec.on_attestation(store, attestation)
        except AssertionError:
            return
        else:
            assert False
    indexed_attestation: Any = spec.get_indexed_attestation(state, attestation)
    spec.on_attestation(store, attestation)
    sample_index: Any = indexed_attestation.attesting_indices[0]
    if spec.fork in ALL_PHASES:
        latest_message = spec.LatestMessage(epoch=attestation.data.target.epoch,
                                              root=attestation.data.beacon_block_root)
    assert store.latest_messages[sample_index] == latest_message

@with_all_phases
@spec_state_test
def test_on_attestation_current_epoch(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(store, store.time + spec.config.SECONDS_PER_SLOT * 2)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation: Any = get_valid_attestation(spec, state, slot=block.slot, signed=True)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH
    assert spec.compute_epoch_at_slot(spec.get_current_slot(store)) == spec.GENESIS_EPOCH
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_previous_epoch(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(store, store.time + spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation: Any = get_valid_attestation(spec, state, slot=block.slot, signed=True)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH
    assert spec.compute_epoch_at_slot(spec.get_current_slot(store)) == spec.GENESIS_EPOCH + 1
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_past_epoch(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + 2 * spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH
    spec.on_tick(store, time)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation: Any = get_valid_attestation(spec, state, slot=state.slot, signed=True)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH
    assert spec.compute_epoch_at_slot(spec.get_current_slot(store)) == spec.GENESIS_EPOCH + 2
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_mismatched_target_and_slot(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(store, store.time + spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation: Any = get_valid_attestation(spec, state, slot=block.slot)
    attestation.data.target.epoch += 1
    sign_attestation(spec, state, attestation)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH + 1
    assert spec.compute_epoch_at_slot(attestation.data.slot) == spec.GENESIS_EPOCH
    assert spec.compute_epoch_at_slot(spec.get_current_slot(store)) == spec.GENESIS_EPOCH + 1
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_inconsistent_target_and_head(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(store, store.time + 2 * spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH)
    target_state_1: Any = state.copy()
    next_epoch(spec, target_state_1)
    target_state_2: Any = state.copy()
    diff_block: Any = build_empty_block_for_next_slot(spec, target_state_2)
    signed_diff_block: Any = state_transition_and_sign_block(spec, target_state_2, diff_block)
    spec.on_block(store, signed_diff_block)
    next_epoch(spec, target_state_2)
    next_slot(spec, target_state_2)
    head_block: Any = build_empty_block_for_next_slot(spec, target_state_1)
    signed_head_block: Any = state_transition_and_sign_block(spec, target_state_1, head_block)
    spec.on_block(store, signed_head_block)
    attestation: Any = get_valid_attestation(spec, target_state_1, slot=head_block.slot, signed=False)
    epoch: int = spec.compute_epoch_at_slot(attestation.data.slot)
    attestation.data.target = spec.Checkpoint(epoch=epoch, root=spec.get_block_root(target_state_2, epoch))
    sign_attestation(spec, state, attestation)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH + 1
    assert spec.compute_epoch_at_slot(attestation.data.slot) == spec.GENESIS_EPOCH + 1
    assert spec.get_block_root(target_state_1, epoch) != attestation.data.target.root
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_target_block_not_in_store(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch_value: int = spec.get_current_epoch(state) + 1
    transition_to(spec, state, spec.compute_start_slot_at_epoch(next_epoch_value) - 1)
    target_block: Any = build_empty_block_for_next_slot(spec, state)
    state_transition_and_sign_block(spec, state, target_block)
    attestation: Any = get_valid_attestation(spec, state, slot=target_block.slot, signed=True)
    assert attestation.data.target.root == target_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_target_checkpoint_not_in_store(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch_value: int = spec.get_current_epoch(state) + 1
    transition_to(spec, state, spec.compute_start_slot_at_epoch(next_epoch_value) - 1)
    target_block: Any = build_empty_block_for_next_slot(spec, state)
    signed_target_block: Any = state_transition_and_sign_block(spec, state, target_block)
    spec.on_block(store, signed_target_block)
    attestation: Any = get_valid_attestation(spec, state, slot=target_block.slot, signed=True)
    assert attestation.data.target.root == target_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_target_checkpoint_not_in_store_diff_slot(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch_value: int = spec.get_current_epoch(state) + 1
    transition_to(spec, state, spec.compute_start_slot_at_epoch(next_epoch_value) - 2)
    target_block: Any = build_empty_block_for_next_slot(spec, state)
    signed_target_block: Any = state_transition_and_sign_block(spec, state, target_block)
    spec.on_block(store, signed_target_block)
    attestation_slot: int = target_block.slot + 1
    transition_to(spec, state, attestation_slot)
    attestation: Any = get_valid_attestation(spec, state, slot=attestation_slot, signed=True)
    assert attestation.data.target.root == target_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_beacon_block_not_in_store(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch_value: int = spec.get_current_epoch(state) + 1
    transition_to(spec, state, spec.compute_start_slot_at_epoch(next_epoch_value) - 1)
    target_block: Any = build_empty_block_for_next_slot(spec, state)
    signed_target_block: Any = state_transition_and_sign_block(spec, state, target_block)
    spec.on_block(store, signed_target_block)
    head_block: Any = build_empty_block_for_next_slot(spec, state)
    state_transition_and_sign_block(spec, state, head_block)
    attestation: Any = get_valid_attestation(spec, state, slot=head_block.slot, signed=True)
    assert attestation.data.target.root == target_block.hash_tree_root()
    assert attestation.data.beacon_block_root == head_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_future_epoch(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + 3 * spec.config.SECONDS_PER_SLOT
    spec.on_tick(store, time)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    next_epoch(spec, state)
    attestation: Any = get_valid_attestation(spec, state, slot=state.slot, signed=True)
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_future_block(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + spec.config.SECONDS_PER_SLOT * 5
    spec.on_tick(store, time)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation: Any = get_valid_attestation(spec, state, slot=block.slot - 1, signed=False)
    attestation.data.beacon_block_root = block.hash_tree_root()
    sign_attestation(spec, state, attestation)
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_same_slot(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + spec.config.SECONDS_PER_SLOT
    spec.on_tick(store, time)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation: Any = get_valid_attestation(spec, state, slot=block.slot, signed=True)
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_invalid_attestation(spec: Any, state: Any) -> None:
    store: Any = get_genesis_forkchoice_store(spec, state)
    time: int = store.time + 3 * spec.config.SECONDS_PER_SLOT
    spec.on_tick(store, time)
    block: Any = build_empty_block_for_next_slot(spec, state)
    signed_block: Any = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation: Any = get_valid_attestation(spec, state, slot=block.slot, signed=True)
    if is_post_electra(spec):
        attestation.committee_bits = spec.Bitvector[spec.MAX_COMMITTEES_PER_SLOT]()
    else:
        attestation.data.index = spec.MAX_COMMITTEES_PER_SLOT * spec.SLOTS_PER_EPOCH
    run_on_attestation(spec, state, store, attestation, False)