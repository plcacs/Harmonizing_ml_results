from eth2spec.test.helpers.custody import get_valid_custody_slashing, get_custody_slashable_shard_transition
from eth2spec.test.helpers.attestations import get_valid_attestation
from eth2spec.test.helpers.constants import CUSTODY_GAME, MINIMAL
from eth2spec.test.helpers.keys import privkeys
from eth2spec.utils.ssz.ssz_typing import ByteList
from eth2spec.test.helpers.state import get_balance, transition_to
from eth2spec.test.context import with_phases, spec_state_test, expect_assertion_error, disable_process_reveal_deadlines, with_presets
from eth2spec.test.phase0.block_processing.test_process_attestation import run_attestation_processing
from eth2spec.phase0.spec import Spec, BeaconState, CustodySlashing

def run_custody_slashing_processing(spec: Spec, state: BeaconState, custody_slashing: CustodySlashing, valid: bool = True, correct: bool = True) -> None:
    ...

def run_standard_custody_slashing_test(spec: Spec, state: BeaconState, shard_lateness: int = None, shard: int = None, validator_index: int = None, block_lengths: List[int] = None, slashing_message_data: ByteList = None, correct: bool = True, valid: bool = True) -> None:
    ...

@with_phases([CUSTODY_GAME])
@spec_state_test
@disable_process_reveal_deadlines
@with_presets([MINIMAL], reason='too slow')
def test_custody_slashing(spec: Spec, state: BeaconState) -> None:
    ...

@with_phases([CUSTODY_GAME])
@spec_state_test
@disable_process_reveal_deadlines
@with_presets([MINIMAL], reason='too slow')
def test_incorrect_custody_slashing(spec: Spec, state: BeaconState) -> None:
    ...

@with_phases([CUSTODY_GAME])
@spec_state_test
@disable_process_reveal_deadlines
@with_presets([MINIMAL], reason='too slow')
def test_multiple_epochs_custody(spec: Spec, state: BeaconState) -> None:
    ...

@with_phases([CUSTODY_GAME])
@spec_state_test
@disable_process_reveal_deadlines
@with_presets([MINIMAL], reason='too slow')
def test_many_epochs_custody(spec: Spec, state: BeaconState) -> None:
    ...

@with_phases([CUSTODY_GAME])
@spec_state_test
@disable_process_reveal_deadlines
@with_presets([MINIMAL], reason='too slow')
def test_invalid_custody_slashing(spec: Spec, state: BeaconState) -> None:
    ...
