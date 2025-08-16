from random import Random
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.forks import is_post_altair, is_post_electra
from eth2spec.test.helpers.keys import pubkeys, privkeys
from eth2spec.test.helpers.state import get_balance
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_to
from eth2spec.utils import bls
from eth2spec.utils.merkle_minimal import calc_merkle_tree_from_leaves, get_merkle_proof
from eth2spec.utils.ssz.ssz_impl import hash_tree_root
from eth2spec.utils.ssz.ssz_typing import List
from eth2spec.phase0.spec import Spec

def mock_deposit(spec: Spec, state, index: int):
    """
    Mock validator at ``index`` as having just made a deposit
    """

def build_deposit_data(spec: Spec, pubkey: bytes, privkey: bytes, amount: int, withdrawal_credentials: bytes, fork_version=None, signed: bool=False):
    
def sign_deposit_data(spec: Spec, deposit_data, privkey: bytes, fork_version=None):
    
def build_deposit(spec: Spec, deposit_data_list: List[spec.DepositData], pubkey: bytes, privkey: bytes, amount: int, withdrawal_credentials: bytes, signed: bool):
    
def deposit_from_context(spec: Spec, deposit_data_list: List[spec.DepositData], index: int):
    
def prepare_full_genesis_deposits(spec: Spec, amount: int, deposit_count: int, min_pubkey_index: int=0, signed: bool=False, deposit_data_list=None):
    
def prepare_random_genesis_deposits(spec: Spec, deposit_count: int, max_pubkey_index: int, min_pubkey_index: int=0, max_amount=None, min_amount=None, deposit_data_list=None, rng=Random(3131)):
    
def prepare_state_and_deposit(spec: Spec, state, validator_index: int, amount: int, pubkey=None, privkey=None, withdrawal_credentials=None, signed: bool=False):
    
def prepare_deposit_request(spec: Spec, validator_index: int, amount: int, index=None, pubkey=None, privkey=None, withdrawal_credentials=None, signed: bool=False):
    
def prepare_pending_deposit(spec: Spec, validator_index: int, amount: int, pubkey=None, privkey=None, withdrawal_credentials=None, fork_version=None, signed: bool=False, slot=None):
    
def run_deposit_processing(spec: Spec, state, deposit, validator_index: int, valid: bool=True, effective: bool=True):
    
def run_deposit_processing_with_specific_fork_version(spec: Spec, state, fork_version, valid: bool=True, effective: bool=True):
    
def run_deposit_request_processing(spec: Spec, state, deposit_request, validator_index: int, effective: bool=True):
    
def run_pending_deposit_applying(spec: Spec, state, pending_deposit, validator_index: int, effective: bool=True):
