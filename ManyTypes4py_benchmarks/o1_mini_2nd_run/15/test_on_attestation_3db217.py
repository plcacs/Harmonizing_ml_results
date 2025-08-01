from eth2spec.test.context import with_all_phases, spec_state_test
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.attestations import get_valid_attestation, sign_attestation
from eth2spec.test.helpers.constants import ALL_PHASES
from eth2spec.test.helpers.forks import is_post_electra
from eth2spec.test.helpers.state import (
    transition_to,
    state_transition_and_sign_block,
    next_epoch,
    next_slot,
)
from eth2spec.test.helpers.fork_choice import get_genesis_forkchoice_store
from typing import Any
from eth2spec.phase0.spec import Spec
from eth2spec.phase0.state import BeaconState
from eth2spec.phase0.forkchoice import ForkChoiceStore
from eth2spec.phase0.attestation import Attestation

def run_on_attestation(
    spec: Spec,
    state: BeaconState,
    store: ForkChoiceStore,
    attestation: Attestation,
    valid: bool = True,
) -> None:
    if not valid:
        try:
            spec.on_attestation(store, attestation)
        except AssertionError:
            return
        else:
            assert False
    indexed_attestation = spec.get_indexed_attestation(state, attestation)
    spec.on_attestation(store, attestation)
    sample_index = indexed_attestation.attesting_indices[0]
    if spec.fork in ALL_PHASES:
        latest_message = spec.LatestMessage(
            epoch=attestation.data.target.epoch,
            root=attestation.data.beacon_block_root,
        )
    assert store.latest_messages[sample_index] == latest_message

@with_all_phases
@spec_state_test
def test_on_attestation_current_epoch(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(store, store.time + spec.config.SECONDS_PER_SLOT * 2)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation = get_valid_attestation(spec, state, slot=block.slot, signed=True)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH
    assert spec.compute_epoch_at_slot(spec.get_current_slot(store)) == spec.GENESIS_EPOCH
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_previous_epoch(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(
        store, store.time + spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH
    )
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation = get_valid_attestation(spec, state, slot=block.slot, signed=True)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH
    assert (
        spec.compute_epoch_at_slot(spec.get_current_slot(store))
        == spec.GENESIS_EPOCH + 1
    )
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_past_epoch(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + 2 * spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH
    spec.on_tick(store, time)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation = get_valid_attestation(spec, state, slot=state.slot, signed=True)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH
    assert (
        spec.compute_epoch_at_slot(spec.get_current_slot(store))
        == spec.GENESIS_EPOCH + 2
    )
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_mismatched_target_and_slot(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(
        store, store.time + spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH
    )
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation = get_valid_attestation(spec, state, slot=block.slot)
    attestation.data.target.epoch += 1
    sign_attestation(spec, state, attestation)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH + 1
    assert spec.compute_epoch_at_slot(attestation.data.slot) == spec.GENESIS_EPOCH
    assert (
        spec.compute_epoch_at_slot(spec.get_current_slot(store))
        == spec.GENESIS_EPOCH + 1
    )
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_inconsistent_target_and_head(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    spec.on_tick(
        store, store.time + 2 * spec.config.SECONDS_PER_SLOT * spec.SLOTS_PER_EPOCH
    )
    target_state_1 = state.copy()
    next_epoch(spec, target_state_1)
    target_state_2 = state.copy()
    diff_block = build_empty_block_for_next_slot(spec, target_state_2)
    signed_diff_block = state_transition_and_sign_block(spec, target_state_2, diff_block)
    spec.on_block(store, signed_diff_block)
    next_epoch(spec, target_state_2)
    next_slot(spec, target_state_2)
    head_block = build_empty_block_for_next_slot(spec, target_state_1)
    signed_head_block = state_transition_and_sign_block(spec, target_state_1, head_block)
    spec.on_block(store, signed_head_block)
    attestation = get_valid_attestation(
        spec, target_state_1, slot=head_block.slot, signed=False
    )
    epoch = spec.compute_epoch_at_slot(attestation.data.slot)
    attestation.data.target = spec.Checkpoint(
        epoch=epoch,
        root=spec.get_block_root(target_state_2, epoch),
    )
    sign_attestation(spec, state, attestation)
    assert attestation.data.target.epoch == spec.GENESIS_EPOCH + 1
    assert spec.compute_epoch_at_slot(attestation.data.slot) == spec.GENESIS_EPOCH + 1
    assert (
        spec.get_block_root(target_state_1, epoch)
        != attestation.data.target.root
    )
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_target_block_not_in_store(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch = spec.get_current_epoch(state) + 1
    transition_to(
        spec, state, spec.compute_start_slot_at_epoch(next_epoch) - 1
    )
    target_block = build_empty_block_for_next_slot(spec, state)
    state_transition_and_sign_block(spec, state, target_block)
    attestation = get_valid_attestation(
        spec, state, slot=target_block.slot, signed=True
    )
    assert attestation.data.target.root == target_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_target_checkpoint_not_in_store(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch = spec.get_current_epoch(state) + 1
    transition_to(
        spec, state, spec.compute_start_slot_at_epoch(next_epoch) - 1
    )
    target_block = build_empty_block_for_next_slot(spec, state)
    signed_target_block = state_transition_and_sign_block(spec, state, target_block)
    spec.on_block(store, signed_target_block)
    attestation = get_valid_attestation(
        spec, state, slot=target_block.slot, signed=True
    )
    assert attestation.data.target.root == target_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_target_checkpoint_not_in_store_diff_slot(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch = spec.get_current_epoch(state) + 1
    transition_to(
        spec, state, spec.compute_start_slot_at_epoch(next_epoch) - 2
    )
    target_block = build_empty_block_for_next_slot(spec, state)
    signed_target_block = state_transition_and_sign_block(spec, state, target_block)
    spec.on_block(store, signed_target_block)
    attestation_slot = target_block.slot + 1
    transition_to(spec, state, attestation_slot)
    attestation = get_valid_attestation(
        spec, state, slot=attestation_slot, signed=True
    )
    assert attestation.data.target.root == target_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation)

@with_all_phases
@spec_state_test
def test_on_attestation_beacon_block_not_in_store(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + spec.config.SECONDS_PER_SLOT * (spec.SLOTS_PER_EPOCH + 1)
    spec.on_tick(store, time)
    next_epoch = spec.get_current_epoch(state) + 1
    transition_to(
        spec, state, spec.compute_start_slot_at_epoch(next_epoch) - 1
    )
    target_block = build_empty_block_for_next_slot(spec, state)
    signed_target_block = state_transition_and_sign_block(spec, state, target_block)
    spec.on_block(store, signed_target_block)
    head_block = build_empty_block_for_next_slot(spec, state)
    state_transition_and_sign_block(spec, state, head_block)
    attestation = get_valid_attestation(
        spec, state, slot=head_block.slot, signed=True
    )
    assert attestation.data.target.root == target_block.hash_tree_root()
    assert attestation.data.beacon_block_root == head_block.hash_tree_root()
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_future_epoch(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + 3 * spec.config.SECONDS_PER_SLOT
    spec.on_tick(store, time)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    next_epoch(spec, state)
    attestation = get_valid_attestation(
        spec, state, slot=state.slot, signed=True
    )
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_future_block(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + spec.config.SECONDS_PER_SLOT * 5
    spec.on_tick(store, time)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation = get_valid_attestation(
        spec, state, slot=block.slot - 1, signed=False
    )
    attestation.data.beacon_block_root = block.hash_tree_root()
    sign_attestation(spec, state, attestation)
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_same_slot(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + spec.config.SECONDS_PER_SLOT
    spec.on_tick(store, time)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation = get_valid_attestation(
        spec, state, slot=block.slot, signed=True
    )
    run_on_attestation(spec, state, store, attestation, False)

@with_all_phases
@spec_state_test
def test_on_attestation_invalid_attestation(spec: Spec, state: BeaconState) -> None:
    store = get_genesis_forkchoice_store(spec, state)
    time = store.time + 3 * spec.config.SECONDS_PER_SLOT
    spec.on_tick(store, time)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    spec.on_block(store, signed_block)
    attestation = get_valid_attestation(
        spec, state, slot=block.slot, signed=True
    )
    if is_post_electra(spec):
        attestation.committee_bits = spec.Bitvector[spec.MAX_COMMITTEES_PER_SLOT]()
    else:
        attestation.data.index = spec.MAX_COMMITTEES_PER_SLOT * spec.SLOTS_PER_EPOCH
    run_on_attestation(spec, state, store, attestation, False)
