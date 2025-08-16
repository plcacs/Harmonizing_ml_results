from eth2spec import Spec  # type: ignore
from eth2spec.test.context import with_all_phases, spec_state_test
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.attestations import get_valid_attestation, sign_attestation
from eth2spec.test.helpers.constants import ALL_PHASES
from eth2spec.test.helpers.forks import is_post_electra
from eth2spec.test.helpers.state import transition_to, state_transition_and_sign_block, next_epoch, next_slot
from eth2spec.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth2spec.test.helpers.typing import Store

def run_on_attestation(spec: Spec, state: BeaconState, store: Store, attestation: Attestation, valid: bool = True) -> None:
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
        latest_message = spec.LatestMessage(epoch=attestation.data.target.epoch, root=attestation.data.beacon_block_root)
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

# Add type annotations for the remaining test functions as well
