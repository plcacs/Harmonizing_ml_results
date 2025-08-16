from random import Random
from eth2spec.utils import bls
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.forks import is_post_deneb
from eth2spec.test.helpers.keys import privkeys
from eth2spec.phase0.spec import (
    Spec, VoluntaryExit, SignedVoluntaryExit, compute_domain, get_current_epoch,
    DOMAIN_VOLUNTARY_EXIT, FAR_FUTURE_EPOCH, initiate_validator_exit, process_voluntary_exit
)
from eth2spec.phase0.spec import Validator, BeaconState

def prepare_signed_exits(spec: Spec, state: BeaconState, indices: List[int], fork_version: Optional[int] = None) -> List[SignedVoluntaryExit]:

def create_signed_exit(index: int) -> SignedVoluntaryExit:

def sign_voluntary_exit(spec: Spec, state: BeaconState, voluntary_exit: VoluntaryExit, privkey: int, fork_version: Optional[int] = None) -> SignedVoluntaryExit:

def get_exited_validators(spec: Spec, state: BeaconState) -> List[int]:

def get_unslashed_exited_validators(spec: Spec, state: BeaconState) -> List[int]:

def exit_validators(spec: Spec, state: BeaconState, validator_count: int, rng: Optional[Random] = None) -> List[int]:

def run_voluntary_exit_processing(spec: Spec, state: BeaconState, signed_voluntary_exit: SignedVoluntaryExit, valid: bool = True) -> Generator[Tuple[str, Union[BeaconState, SignedVoluntaryExit, None]], None, None]:
