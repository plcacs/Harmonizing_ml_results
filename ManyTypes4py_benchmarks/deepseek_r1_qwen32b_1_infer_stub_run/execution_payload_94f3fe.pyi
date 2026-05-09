from eth2spec.eth_types import (
    ExecutionPayload,
    ExecutionPayloadHeader,
    SignedExecutionPayloadHeader,
    ValidatorIndex,
    Slot,
    Gwei,
    Root,
    Hash32,
    Bytes32,
    ByteVector,
    ByteList,
    Transaction,
)
from typing import Any, List, Optional

def get_execution_payload_header(spec: Any, execution_payload: Any) -> ExecutionPayloadHeader: ...

def compute_trie_root_from_indexed_data(data: List[Any]) -> bytes: ...

def compute_requests_hash(block_requests: List[bytes]) -> bytes: ...

def compute_el_header_block_hash(
    spec: Any,
    payload_header: ExecutionPayloadHeader,
    transactions_trie_root: bytes,
    withdrawals_trie_root: Optional[bytes] = ...,
    parent_beacon_block_root: Optional[Root] = ...,
    requests_hash: Optional[bytes] = ...,
) -> Hash32: ...

def get_withdrawal_rlp(withdrawal: Any) -> bytes: ...

def get_deposit_request_rlp_bytes(deposit_request: Any) -> bytes: ...

def get_withdrawal_request_rlp_bytes(withdrawal_request: Any) -> bytes: ...

def get_consolidation_request_rlp_bytes(consolidation_request: Any) -> bytes: ...

def compute_el_block_hash_with_new_fields(
    spec: Any,
    payload: ExecutionPayload,
    parent_beacon_block_root: Optional[Root],
    requests_hash: Optional[bytes],
) -> Hash32: ...

def compute_el_block_hash(spec: Any, payload: ExecutionPayload, pre_state: Any) -> Hash32: ...

def compute_el_block_hash_for_block(spec: Any, block: Any) -> Hash32: ...

def build_empty_post_eip7732_execution_payload_header(spec: Any, state: Any) -> Optional[ExecutionPayloadHeader]: ...

def build_empty_signed_execution_payload_header(spec: Any, state: Any) -> Optional[SignedExecutionPayloadHeader]: ...

def build_empty_execution_payload(spec: Any, state: Any, randao_mix: Optional[Any] = ...) -> ExecutionPayload: ...

def build_randomized_execution_payload(spec: Any, state: Any, rng: Any) -> ExecutionPayload: ...

def build_state_with_incomplete_transition(spec: Any, state: Any) -> Any: ...

def build_state_with_complete_transition(spec: Any, state: Any) -> Any: ...

def build_state_with_execution_payload_header(spec: Any, state: Any, execution_payload_header: ExecutionPayloadHeader) -> Any: ...

def get_random_tx(rng: Any) -> bytes: ...