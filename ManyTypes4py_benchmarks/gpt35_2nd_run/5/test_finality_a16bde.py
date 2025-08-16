from eth2spec.test.helpers.state import BeaconState
from eth2spec.test.helpers.attestations import next_epoch_with_attestations
from eth2spec.test.context import spec_state_test, with_all_phases
from eth2spec.test.helpers.state import next_epoch_via_block

def check_finality(spec, state: BeaconState, prev_state: BeaconState, current_justified_changed: bool, previous_justified_changed: bool, finalized_changed: bool) -> None:
    if current_justified_changed:
        assert state.current_justified_checkpoint.epoch > prev_state.current_justified_checkpoint.epoch
        assert state.current_justified_checkpoint.root != prev_state.current_justified_checkpoint.root
    else:
        assert state.current_justified_checkpoint == prev_state.current_justified_checkpoint
    if previous_justified_changed:
        assert state.previous_justified_checkpoint.epoch > prev_state.previous_justified_checkpoint.epoch
        assert state.previous_justified_checkpoint.root != prev_state.previous_justified_checkpoint.root
    else:
        assert state.previous_justified_checkpoint == prev_state.previous_justified_checkpoint
    if finalized_changed:
        assert state.finalized_checkpoint.epoch > prev_state.finalized_checkpoint.epoch
        assert state.finalized_checkpoint.root != prev_state.finalized_checkpoint.root
    else:
        assert state.finalized_checkpoint == prev_state.finalized_checkpoint

@with_all_phases
@spec_state_test
def test_finality_no_updates_at_genesis(spec, state: BeaconState) -> None:
    assert spec.get_current_epoch(state) == spec.GENESIS_EPOCH
    yield ('pre', state)
    blocks = []
    for epoch in range(2):
        prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, True, False)
        blocks += new_blocks
        if epoch == 0:
            check_finality(spec, state, prev_state, False, False, False)
        elif epoch == 1:
            check_finality(spec, state, prev_state, False, False, False)
    yield ('blocks', blocks)
    yield ('post', state)

@with_all_phases
@spec_state_test
def test_finality_rule_4(spec, state: BeaconState) -> None:
    next_epoch_via_block(spec, state)
    next_epoch_via_block(spec, state)
    yield ('pre', state)
    blocks = []
    for epoch in range(2):
        prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, True, False)
        blocks += new_blocks
        if epoch == 0:
            check_finality(spec, state, prev_state, True, False, False)
        elif epoch == 1:
            check_finality(spec, state, prev_state, True, True, True)
            assert state.finalized_checkpoint == prev_state.current_justified_checkpoint
    yield ('blocks', blocks)
    yield ('post', state)

@with_all_phases
@spec_state_test
def test_finality_rule_1(spec, state: BeaconState) -> None:
    next_epoch_via_block(spec, state)
    next_epoch_via_block(spec, state)
    yield ('pre', state)
    blocks = []
    for epoch in range(3):
        prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, False, True)
        blocks += new_blocks
        if epoch == 0:
            check_finality(spec, state, prev_state, True, False, False)
        elif epoch == 1:
            check_finality(spec, state, prev_state, True, True, False)
        elif epoch == 2:
            check_finality(spec, state, prev_state, True, True, True)
            assert state.finalized_checkpoint == prev_state.previous_justified_checkpoint
    yield ('blocks', blocks)
    yield ('post', state)

@with_all_phases
@spec_state_test
def test_finality_rule_2(spec, state: BeaconState) -> None:
    next_epoch_via_block(spec, state)
    next_epoch_via_block(spec, state)
    yield ('pre', state)
    blocks = []
    for epoch in range(3):
        if epoch == 0:
            prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, True, False)
            check_finality(spec, state, prev_state, True, False, False)
        elif epoch == 1:
            prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, False, False)
            check_finality(spec, state, prev_state, False, True, False)
        elif epoch == 2:
            prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, False, True)
            check_finality(spec, state, prev_state, True, False, True)
            assert state.finalized_checkpoint == prev_state.previous_justified_checkpoint
        blocks += new_blocks
    yield ('blocks', blocks)
    yield ('post', state)

@with_all_phases
@spec_state_test
def test_finality_rule_3(spec, state: BeaconState) -> None:
    """
    Test scenario described here
    https://github.com/ethereum/consensus-specs/issues/611#issuecomment-463612892
    """
    next_epoch_via_block(spec, state)
    next_epoch_via_block(spec, state)
    yield ('pre', state)
    blocks = []
    prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, True, False)
    blocks += new_blocks
    check_finality(spec, state, prev_state, True, False, False)
    prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, True, False)
    blocks += new_blocks
    check_finality(spec, state, prev_state, True, True, True)
    prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, False, False)
    blocks += new_blocks
    check_finality(spec, state, prev_state, False, True, False)
    prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, False, True)
    blocks += new_blocks
    check_finality(spec, state, prev_state, True, False, True)
    prev_state, new_blocks, state = next_epoch_with_attestations(spec, state, True, True)
    blocks += new_blocks
    check_finality(spec, state, prev_state, True, True, True)
    assert state.finalized_checkpoint == prev_state.current_justified_checkpoint
    yield ('blocks', blocks)
    yield ('post', state)
