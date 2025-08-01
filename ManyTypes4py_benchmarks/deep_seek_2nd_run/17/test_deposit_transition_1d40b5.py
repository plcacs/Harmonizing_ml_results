from typing import List, Tuple, Dict, Any, Generator, Optional, Union
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.context import spec_state_test, with_phases, ELECTRA
from eth2spec.test.helpers.deposits import build_deposit_data, deposit_from_context, prepare_deposit_request
from eth2spec.test.helpers.execution_payload import compute_el_block_hash_for_block
from eth2spec.test.helpers.keys import privkeys, pubkeys
from eth2spec.test.helpers.state import state_transition_and_sign_block
from eth2spec.typing import SpecObject

def run_deposit_transition_block(spec: SpecObject, state: SpecObject, block: SpecObject, top_up_keys: List[bytes] = [], valid: bool = True) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    """
    Run ``process_block``, yielding:
      - pre-state ('pre')
      - block ('block')
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    yield ('pre', state)
    pre_pending_deposits_len = len(state.pending_deposits)
    pre_validators_len = len(state.validators)
    signed_block = state_transition_and_sign_block(spec, state, block, not valid)
    yield ('blocks', [signed_block])
    yield ('post', state if valid else None)
    if valid:
        for i, deposit in enumerate(block.body.deposits):
            validator = state.validators[pre_validators_len + i]
            assert validator.pubkey == deposit.data.pubkey
            assert validator.withdrawal_credentials == deposit.data.withdrawal_credentials
            assert validator.effective_balance == spec.Gwei(0)
            assert state.balances[pre_validators_len + i] == spec.Gwei(0)
            pending_deposit = state.pending_deposits[pre_pending_deposits_len + i]
            assert pending_deposit.pubkey == deposit.data.pubkey
            assert pending_deposit.withdrawal_credentials == deposit.data.withdrawal_credentials
            assert pending_deposit.amount == deposit.data.amount
            assert pending_deposit.signature == deposit.data.signature
            assert pending_deposit.slot == spec.GENESIS_SLOT
        assert len(state.validators) == pre_validators_len + len(block.body.deposits)
        for i, deposit_request in enumerate(block.body.execution_requests.deposits):
            pending_deposit = state.pending_deposits[pre_pending_deposits_len + len(block.body.deposits) + i]
            assert pending_deposit.pubkey == deposit_request.pubkey
            assert pending_deposit.withdrawal_credentials == deposit_request.withdrawal_credentials
            assert pending_deposit.amount == deposit_request.amount
            assert pending_deposit.signature == deposit_request.signature
            assert pending_deposit.slot == signed_block.message.slot
        assert len(state.pending_deposits) == pre_pending_deposits_len + len(block.body.deposits) + len(block.body.execution_requests.deposits)

def prepare_state_and_block(spec: SpecObject, state: SpecObject, deposit_cnt: int, deposit_request_cnt: int, first_deposit_request_index: int = 0, deposit_requests_start_index: Optional[int] = None, eth1_data_deposit_count: Optional[int] = None) -> Tuple[SpecObject, SpecObject]:
    deposits: List[SpecObject] = []
    deposit_requests: List[SpecObject] = []
    keypair_index = len(state.validators)
    deposit_data_list: List[SpecObject] = []
    for index in range(deposit_cnt):
        deposit_data = build_deposit_data(spec, pubkeys[keypair_index], privkeys[keypair_index], spec.MIN_ACTIVATION_BALANCE, spec.BLS_WITHDRAWAL_PREFIX + spec.hash(pubkeys[keypair_index])[1:], signed=True)
        deposit_data_list.append(deposit_data)
        keypair_index += 1
    deposit_root = None
    for index in range(deposit_cnt):
        deposit, deposit_root, _ = deposit_from_context(spec, deposit_data_list, index)
        deposits.append(deposit)
    if deposit_root:
        state.eth1_deposit_index = 0
        if not eth1_data_deposit_count:
            eth1_data_deposit_count = deposit_cnt
        state.eth1_data = spec.Eth1Data(deposit_root=deposit_root, deposit_count=eth1_data_deposit_count, block_hash=state.eth1_data.block_hash)
    for offset in range(deposit_request_cnt):
        deposit_request = prepare_deposit_request(spec, keypair_index, spec.MIN_ACTIVATION_BALANCE, first_deposit_request_index + offset, signed=True)
        deposit_requests.append(deposit_request)
        keypair_index += 1
    if deposit_requests_start_index:
        state.deposit_requests_start_index = deposit_requests_start_index
    block = build_empty_block_for_next_slot(spec, state)
    block.body.deposits = deposits
    block.body.execution_requests.deposits = deposit_requests
    block.body.execution_payload.block_hash = compute_el_block_hash_for_block(spec, block)
    return (state, block)

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__start_index_is_set(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=0, deposit_request_cnt=2, first_deposit_request_index=state.eth1_data.deposit_count + 11)
    yield from run_deposit_transition_block(spec, state, block)
    assert state.deposit_requests_start_index == block.body.execution_requests.deposits[0].index

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__process_eth1_deposits(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=3, deposit_request_cnt=1, first_deposit_request_index=11, deposit_requests_start_index=7)
    yield from run_deposit_transition_block(spec, state, block)

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__process_max_eth1_deposits(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=spec.MAX_DEPOSITS, deposit_request_cnt=1, first_deposit_request_index=spec.MAX_DEPOSITS + 1, deposit_requests_start_index=spec.MAX_DEPOSITS, eth1_data_deposit_count=23)
    yield from run_deposit_transition_block(spec, state, block)

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__process_eth1_deposits_up_to_start_index(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=3, deposit_request_cnt=1, first_deposit_request_index=7, deposit_requests_start_index=3)
    yield from run_deposit_transition_block(spec, state, block)

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__invalid_not_enough_eth1_deposits(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=3, deposit_request_cnt=1, first_deposit_request_index=29, deposit_requests_start_index=23, eth1_data_deposit_count=17)
    yield from run_deposit_transition_block(spec, state, block, valid=False)

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__invalid_too_many_eth1_deposits(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=3, deposit_request_cnt=1, first_deposit_request_index=11, deposit_requests_start_index=7, eth1_data_deposit_count=2)
    yield from run_deposit_transition_block(spec, state, block, valid=False)

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__invalid_eth1_deposits_overlap_in_protocol_deposits(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=spec.MAX_DEPOSITS, deposit_request_cnt=1, first_deposit_request_index=spec.MAX_DEPOSITS, deposit_requests_start_index=spec.MAX_DEPOSITS - 1, eth1_data_deposit_count=23)
    yield from run_deposit_transition_block(spec, state, block, valid=False)

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__deposit_and_top_up_same_block(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    state, block = prepare_state_and_block(spec, state, deposit_cnt=1, deposit_request_cnt=1, first_deposit_request_index=11, deposit_requests_start_index=7)
    top_up_keys = [block.body.deposits[0].data.pubkey]
    block.body.execution_requests.deposits[0].pubkey = top_up_keys[0]
    block.body.execution_payload.block_hash = compute_el_block_hash_for_block(spec, block)
    pre_pending_deposits = len(state.pending_deposits)
    yield from run_deposit_transition_block(spec, state, block, top_up_keys=top_up_keys)
    assert len(state.pending_deposits) == pre_pending_deposits + 2
    assert state.pending_deposits[pre_pending_deposits].amount == block.body.deposits[0].data.amount
    amount_from_deposit = block.body.execution_requests.deposits[0].amount
    assert state.pending_deposits[pre_pending_deposits + 1].amount == amount_from_deposit

@with_phases([ELECTRA])
@spec_state_test
def test_deposit_transition__deposit_with_same_pubkey_different_withdrawal_credentials(spec: SpecObject, state: SpecObject) -> Generator[Tuple[str, Union[SpecObject, List[SpecObject], None]], None, None]:
    deposit_count = 1
    deposit_request_count = 4
    state, block = prepare_state_and_block(spec, state, deposit_cnt=deposit_count, deposit_request_cnt=deposit_request_count)
    indices_with_same_pubkey = [1, 3]
    for index in indices_with_same_pubkey:
        block.body.execution_requests.deposits[index].pubkey = block.body.deposits[0].data.pubkey
        assert block.body.execution_requests.deposits[index].withdrawal_credentials != block.body.deposits[0].data.withdrawal_credentials
    block.body.execution_payload.block_hash = compute_el_block_hash_for_block(spec, block)
    deposit_requests = block.body.execution_requests.deposits.copy()
    yield from run_deposit_transition_block(spec, state, block)
    assert len(state.pending_deposits) == deposit_request_count + deposit_count
    for index in indices_with_same_pubkey:
        assert state.pending_deposits[deposit_count + index].pubkey == deposit_requests[index].pubkey
        assert state.pending_deposits[deposit_count + index].withdrawal_credentials == deposit_requests[index].withdrawal_credentials
