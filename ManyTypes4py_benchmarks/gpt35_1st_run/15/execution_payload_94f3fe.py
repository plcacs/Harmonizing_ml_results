from eth2spec import Spec  # type: ignore
from eth2spec.test.helpers.keys import privkeys  # type: ignore
from eth2spec.utils.ssz.ssz_impl import hash_tree_root  # type: ignore
from eth2spec.debug.random_value import get_random_bytes_list  # type: ignore
from eth2spec.test.helpers.withdrawals import get_expected_withdrawals  # type: ignore
from eth2spec.test.helpers.forks import is_post_capella, is_post_deneb, is_post_electra, is_post_eip7732  # type: ignore
from typing import List, Tuple, Optional

def get_execution_payload_header(spec: Spec, execution_payload: ExecutionPayload) -> ExecutionPayloadHeader:
    ...

def compute_trie_root_from_indexed_data(data: List) -> bytes:
    ...

def compute_requests_hash(block_requests: List) -> bytes:
    ...

def compute_el_header_block_hash(spec: Spec, payload_header: ExecutionPayloadHeader, transactions_trie_root: bytes, withdrawals_trie_root: Optional[bytes] = None, parent_beacon_block_root: Optional[bytes] = None, requests_hash: Optional[bytes] = None) -> bytes:
    ...

def get_withdrawal_rlp(withdrawal: Withdrawal) -> bytes:
    ...

def get_deposit_request_rlp_bytes(deposit_request: DepositRequest) -> bytes:
    ...

def get_withdrawal_request_rlp_bytes(withdrawal_request: WithdrawalRequest) -> bytes:
    ...

def get_consolidation_request_rlp_bytes(consolidation_request: ConsolidationRequest) -> bytes:
    ...

def compute_el_block_hash_with_new_fields(spec: Spec, payload: ExecutionPayload, parent_beacon_block_root: Optional[bytes], requests_hash: Optional[bytes]) -> bytes:
    ...

def compute_el_block_hash(spec: Spec, payload: ExecutionPayload, pre_state: State) -> bytes:
    ...

def compute_el_block_hash_for_block(spec: Spec, block: Block) -> bytes:
    ...

def build_empty_post_eip7732_execution_payload_header(spec: Spec, state: State) -> Optional[ExecutionPayloadHeader]:
    ...

def build_empty_signed_execution_payload_header(spec: Spec, state: State) -> Optional[SignedExecutionPayloadHeader]:
    ...

def build_empty_execution_payload(spec: Spec, state: State, randao_mix: Optional[bytes] = None) -> ExecutionPayload:
    ...

def build_randomized_execution_payload(spec: Spec, state: State, rng: Random) -> ExecutionPayload:
    ...

def build_state_with_incomplete_transition(spec: Spec, state: State) -> State:
    ...

def build_state_with_complete_transition(spec: Spec, state: State) -> State:
    ...

def build_state_with_execution_payload_header(spec: Spec, state: State, execution_payload_header: ExecutionPayloadHeader) -> State:
    ...

def get_random_tx(rng: Random) -> bytes:
    ...
