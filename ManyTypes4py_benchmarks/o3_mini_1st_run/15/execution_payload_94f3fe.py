from typing import Any, Optional, List
from eth_hash.auto import keccak
from hashlib import sha256
from trie import HexaryTrie
from rlp import encode
from rlp.sedes import big_endian_int, Binary, List as RLPList
from eth2spec.test.helpers.keys import privkeys
from eth2spec.utils.ssz.ssz_impl import hash_tree_root
from eth2spec.debug.random_value import get_random_bytes_list
from eth2spec.test.helpers.withdrawals import get_expected_withdrawals
from eth2spec.test.helpers.forks import is_post_capella, is_post_deneb, is_post_electra, is_post_eip7732
from random import Random

def get_execution_payload_header(spec: Any, execution_payload: Any) -> Any:
    if is_post_eip7732(spec):
        return spec.ExecutionPayloadHeader(
            parent_block_hash=execution_payload.parent_hash,
            parent_block_root=spec.Root(),
            block_hash=execution_payload.block_hash,
            gas_limit=execution_payload.gas_limit,
            builder_index=spec.ValidatorIndex(0),
            slot=spec.Slot(0),
            value=spec.Gwei(0),
            blob_kzg_commitments_root=spec.Root()
        )
    payload_header = spec.ExecutionPayloadHeader(
        parent_hash=execution_payload.parent_hash,
        fee_recipient=execution_payload.fee_recipient,
        state_root=execution_payload.state_root,
        receipts_root=execution_payload.receipts_root,
        logs_bloom=execution_payload.logs_bloom,
        prev_randao=execution_payload.prev_randao,
        block_number=execution_payload.block_number,
        gas_limit=execution_payload.gas_limit,
        gas_used=execution_payload.gas_used,
        timestamp=execution_payload.timestamp,
        extra_data=execution_payload.extra_data,
        base_fee_per_gas=execution_payload.base_fee_per_gas,
        block_hash=execution_payload.block_hash,
        transactions_root=spec.hash_tree_root(execution_payload.transactions)
    )
    if is_post_capella(spec):
        payload_header.withdrawals_root = spec.hash_tree_root(execution_payload.withdrawals)
    if is_post_deneb(spec):
        payload_header.blob_gas_used = execution_payload.blob_gas_used
        payload_header.excess_blob_gas = execution_payload.excess_blob_gas
    return payload_header

def compute_trie_root_from_indexed_data(data: List[Any]) -> bytes:
    """
    Computes the root hash of `patriciaTrie(rlp(Index) => Data)` for a data array.
    """
    t = HexaryTrie(db={})
    for i, obj in enumerate(data):
        k = encode(i, big_endian_int)
        t.set(k, obj)
    return t.root_hash

def compute_requests_hash(block_requests: List[bytes]) -> bytes:
    m = sha256()
    for r in block_requests:
        if len(r) > 1:
            m.update(sha256(r).digest())
    return m.digest()

def compute_el_header_block_hash(
    spec: Any,
    payload_header: Any,
    transactions_trie_root: bytes,
    withdrawals_trie_root: Optional[bytes] = None,
    parent_beacon_block_root: Optional[bytes] = None,
    requests_hash: Optional[bytes] = None
) -> bytes:
    """
    Computes the RLP execution block hash described by an `ExecutionPayloadHeader`.
    """
    if is_post_eip7732(spec):
        return spec.Hash32()
    execution_payload_header_rlp = [
        (Binary(32, 32), payload_header.parent_hash),
        (Binary(32, 32), bytes.fromhex('1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347')),
        (Binary(20, 20), payload_header.fee_recipient),
        (Binary(32, 32), payload_header.state_root),
        (Binary(32, 32), transactions_trie_root),
        (Binary(32, 32), payload_header.receipts_root),
        (Binary(256, 256), payload_header.logs_bloom),
        (big_endian_int, 0),
        (big_endian_int, payload_header.block_number),
        (big_endian_int, payload_header.gas_limit),
        (big_endian_int, payload_header.gas_used),
        (big_endian_int, payload_header.timestamp),
        (Binary(0, 32), payload_header.extra_data),
        (Binary(32, 32), payload_header.prev_randao),
        (Binary(8, 8), bytes.fromhex('0000000000000000')),
        (big_endian_int, payload_header.base_fee_per_gas)
    ]
    if is_post_capella(spec):
        execution_payload_header_rlp.append((Binary(32, 32), withdrawals_trie_root))
    if is_post_deneb(spec):
        execution_payload_header_rlp.append((big_endian_int, payload_header.blob_gas_used))
        execution_payload_header_rlp.append((big_endian_int, payload_header.excess_blob_gas))
        execution_payload_header_rlp.append((Binary(32, 32), parent_beacon_block_root))
    if is_post_electra(spec):
        execution_payload_header_rlp.append((Binary(32, 32), requests_hash))
    sedes = RLPList([schema for schema, _ in execution_payload_header_rlp])
    values = [value for _, value in execution_payload_header_rlp]
    encoded = encode(values, sedes)
    return spec.Hash32(keccak(encoded))

def get_withdrawal_rlp(withdrawal: Any) -> bytes:
    withdrawal_rlp = [
        (big_endian_int, withdrawal.index),
        (big_endian_int, withdrawal.validator_index),
        (Binary(20, 20), withdrawal.address),
        (big_endian_int, withdrawal.amount)
    ]
    sedes = RLPList([schema for schema, _ in withdrawal_rlp])
    values = [value for _, value in withdrawal_rlp]
    return encode(values, sedes)

def get_deposit_request_rlp_bytes(deposit_request: Any) -> bytes:
    deposit_request_rlp = [
        (Binary(48, 48), deposit_request.pubkey),
        (Binary(32, 32), deposit_request.withdrawal_credentials),
        (big_endian_int, deposit_request.amount),
        (Binary(96, 96), deposit_request.signature),
        (big_endian_int, deposit_request.index)
    ]
    sedes = RLPList([schema for schema, _ in deposit_request_rlp])
    values = [value for _, value in deposit_request_rlp]
    return b'\x00' + encode(values, sedes)

def get_withdrawal_request_rlp_bytes(withdrawal_request: Any) -> bytes:
    withdrawal_request_rlp = [
        (Binary(20, 20), withdrawal_request.source_address),
        (Binary(48, 48), withdrawal_request.validator_pubkey)
    ]
    sedes = RLPList([schema for schema, _ in withdrawal_request_rlp])
    values = [value for _, value in withdrawal_request_rlp]
    return b'\x01' + encode(values, sedes)

def get_consolidation_request_rlp_bytes(consolidation_request: Any) -> bytes:
    consolidation_request_rlp = [
        (Binary(20, 20), consolidation_request.source_address),
        (Binary(48, 48), consolidation_request.source_pubkey),
        (Binary(48, 48), consolidation_request.target_pubkey)
    ]
    sedes = RLPList([schema for schema, _ in consolidation_request_rlp])
    values = [value for _, value in consolidation_request_rlp]
    return b'\x02' + encode(values, sedes)

def compute_el_block_hash_with_new_fields(
    spec: Any,
    payload: Any,
    parent_beacon_block_root: Optional[bytes],
    requests_hash: Optional[bytes]
) -> bytes:
    if payload == spec.ExecutionPayload():
        return spec.Hash32()
    transactions_trie_root: bytes = compute_trie_root_from_indexed_data(payload.transactions)
    withdrawals_trie_root: Optional[bytes] = None
    if is_post_capella(spec):
        withdrawals_encoded: List[bytes] = [get_withdrawal_rlp(withdrawal) for withdrawal in payload.withdrawals]
        withdrawals_trie_root = compute_trie_root_from_indexed_data(withdrawals_encoded)
    if not is_post_deneb(spec):
        parent_beacon_block_root = None
    payload_header: Any = get_execution_payload_header(spec, payload)
    return compute_el_header_block_hash(
        spec,
        payload_header,
        transactions_trie_root,
        withdrawals_trie_root,
        parent_beacon_block_root,
        requests_hash
    )

def compute_el_block_hash(spec: Any, payload: Any, pre_state: Any) -> bytes:
    parent_beacon_block_root: Optional[bytes] = None
    requests_hash: Optional[bytes] = None
    if is_post_deneb(spec):
        previous_block_header = pre_state.latest_block_header.copy()
        if previous_block_header.state_root == spec.Root():
            previous_block_header.state_root = pre_state.hash_tree_root()
        parent_beacon_block_root = previous_block_header.hash_tree_root()
    if is_post_electra(spec):
        requests_hash = compute_requests_hash([])
    return compute_el_block_hash_with_new_fields(spec, payload, parent_beacon_block_root, requests_hash)

def compute_el_block_hash_for_block(spec: Any, block: Any) -> bytes:
    requests_hash: Optional[bytes] = None
    if is_post_electra(spec):
        requests_list: List[bytes] = spec.get_execution_requests_list(block.body.execution_requests)
        requests_hash = compute_requests_hash(requests_list)
    return compute_el_block_hash_with_new_fields(spec, block.body.execution_payload, block.parent_root, requests_hash)

def build_empty_post_eip7732_execution_payload_header(spec: Any, state: Any) -> Optional[Any]:
    if not is_post_eip7732(spec):
        return None
    parent_block_root: bytes = hash_tree_root(state.latest_block_header)
    return spec.ExecutionPayloadHeader(
        parent_block_hash=state.latest_block_hash,
        parent_block_root=parent_block_root,
        block_hash=spec.Hash32(),
        gas_limit=spec.uint64(0),
        builder_index=spec.ValidatorIndex(0),
        slot=state.slot,
        value=spec.Gwei(0),
        blob_kzg_commitments_root=spec.Root()
    )

def build_empty_signed_execution_payload_header(spec: Any, state: Any) -> Optional[Any]:
    if not is_post_eip7732(spec):
        return None
    message: Any = build_empty_post_eip7732_execution_payload_header(spec, state)
    privkey = privkeys[0]
    signature = spec.get_execution_payload_header_signature(state, message, privkey)
    return spec.SignedExecutionPayloadHeader(message=message, signature=signature)

def build_empty_execution_payload(spec: Any, state: Any, randao_mix: Optional[Any] = None) -> Any:
    """
    Assuming a pre-state of the same slot, build a valid ExecutionPayload without any transactions.
    """
    latest = state.latest_execution_payload_header
    timestamp: int = spec.compute_timestamp_at_slot(state, state.slot)
    empty_txs: Any = spec.List[spec.Transaction, spec.MAX_TRANSACTIONS_PER_PAYLOAD]()
    if randao_mix is None:
        randao_mix = spec.get_randao_mix(state, spec.get_current_epoch(state))
    payload = spec.ExecutionPayload(
        parent_hash=latest.block_hash,
        fee_recipient=spec.ExecutionAddress(),
        receipts_root=spec.Bytes32(bytes.fromhex('1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347')),
        logs_bloom=spec.ByteVector[spec.BYTES_PER_LOGS_BLOOM](),
        prev_randao=randao_mix,
        gas_used=0,
        timestamp=timestamp,
        extra_data=spec.ByteList[spec.MAX_EXTRA_DATA_BYTES](),
        transactions=empty_txs
    )
    if not is_post_eip7732(spec):
        payload.state_root = latest.state_root
        payload.block_number = latest.block_number + 1
        payload.gas_limit = latest.gas_limit
        payload.base_fee_per_gas = latest.base_fee_per_gas
    if is_post_capella(spec):
        payload.withdrawals = get_expected_withdrawals(spec, state)
    if is_post_deneb(spec):
        payload.blob_gas_used = 0
        payload.excess_blob_gas = 0
    payload.block_hash = compute_el_block_hash(spec, payload, state)
    return payload

def build_randomized_execution_payload(spec: Any, state: Any, rng: Random) -> Any:
    execution_payload = build_empty_execution_payload(spec, state)
    execution_payload.fee_recipient = spec.ExecutionAddress(get_random_bytes_list(rng, 20))
    execution_payload.state_root = spec.Bytes32(get_random_bytes_list(rng, 32))
    execution_payload.receipts_root = spec.Bytes32(get_random_bytes_list(rng, 32))
    execution_payload.logs_bloom = spec.ByteVector[spec.BYTES_PER_LOGS_BLOOM](get_random_bytes_list(rng, spec.BYTES_PER_LOGS_BLOOM))
    execution_payload.block_number = rng.randint(0, int(100000000000.0))
    execution_payload.gas_limit = rng.randint(0, int(100000000000.0))
    execution_payload.gas_used = rng.randint(0, int(100000000000.0))
    extra_data_length: int = rng.randint(0, spec.MAX_EXTRA_DATA_BYTES)
    execution_payload.extra_data = spec.ByteList[spec.MAX_EXTRA_DATA_BYTES](get_random_bytes_list(rng, extra_data_length))
    execution_payload.base_fee_per_gas = rng.randint(0, 2 ** 256 - 1)
    num_transactions: int = rng.randint(0, 100)
    execution_payload.transactions = [get_random_tx(rng) for _ in range(num_transactions)]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    return execution_payload

def build_state_with_incomplete_transition(spec: Any, state: Any) -> Any:
    state = build_state_with_execution_payload_header(spec, state, spec.ExecutionPayloadHeader())
    assert not spec.is_merge_transition_complete(state)
    return state

def build_state_with_complete_transition(spec: Any, state: Any) -> Any:
    pre_state_payload: Any = build_empty_execution_payload(spec, state)
    payload_header: Any = get_execution_payload_header(spec, pre_state_payload)
    state = build_state_with_execution_payload_header(spec, state, payload_header)
    assert spec.is_merge_transition_complete(state)
    return state

def build_state_with_execution_payload_header(spec: Any, state: Any, execution_payload_header: Any) -> Any:
    pre_state = state.copy()
    pre_state.latest_execution_payload_header = execution_payload_header
    return pre_state

def get_random_tx(rng: Random) -> bytes:
    return get_random_bytes_list(rng, rng.randint(1, 1000))