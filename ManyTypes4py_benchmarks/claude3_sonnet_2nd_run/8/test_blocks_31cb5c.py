import random
from typing import Iterator, List, Optional, Tuple, Any
from eth2spec.test.helpers.state import state_transition_and_sign_block
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.context import spec_state_test, with_deneb_and_later
from eth2spec.test.helpers.execution_payload import compute_el_block_hash, get_random_tx
from eth2spec.test.helpers.blob import get_sample_blob_tx, get_max_blob_count

def run_block_with_blobs(
    spec: Any,
    state: Any,
    blob_count: int,
    tx_count: int = 1,
    blob_gas_used: int = 1,
    excess_blob_gas: int = 1,
    non_blob_tx_count: int = 0,
    rng: random.Random = random.Random(7777),
    valid: bool = True
) -> Iterator[Tuple[str, Any]]:
    yield ('pre', state)
    block = build_empty_block_for_next_slot(spec, state)
    txs: List[bytes] = []
    blob_kzg_commitments: List[bytes] = []
    for _ in range(tx_count):
        opaque_tx, _, commits, _ = get_sample_blob_tx(spec, blob_count=blob_count)
        txs.append(opaque_tx)
        blob_kzg_commitments += commits
    for _ in range(non_blob_tx_count):
        txs.append(get_random_tx(rng))
    rng.shuffle(txs)
    block.body.blob_kzg_commitments = blob_kzg_commitments
    block.body.execution_payload.transactions = txs
    block.body.execution_payload.blob_gas_used = blob_gas_used
    block.body.execution_payload.excess_blob_gas = excess_blob_gas
    block.body.execution_payload.block_hash = compute_el_block_hash(spec, block.body.execution_payload, state)
    if valid:
        signed_block = state_transition_and_sign_block(spec, state, block)
    else:
        signed_block = state_transition_and_sign_block(spec, state, block, expect_fail=True)
    yield ('blocks', [signed_block])
    yield ('post', state if valid else None)

@with_deneb_and_later
@spec_state_test
def test_zero_blob(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=0)

@with_deneb_and_later
@spec_state_test
def test_one_blob(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=1)

@with_deneb_and_later
@spec_state_test
def test_one_blob_two_txs(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=1, tx_count=2)

@with_deneb_and_later
@spec_state_test
def test_one_blob_max_txs(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=1, tx_count=get_max_blob_count(spec))

@with_deneb_and_later
@spec_state_test
def test_invalid_one_blob_max_plus_one_txs(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=1, tx_count=get_max_blob_count(spec) + 1, valid=False)

@with_deneb_and_later
@spec_state_test
def test_max_blobs_per_block(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=get_max_blob_count(spec))

@with_deneb_and_later
@spec_state_test
def test_invalid_max_blobs_per_block_two_txs(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=get_max_blob_count(spec), tx_count=2, valid=False)

@with_deneb_and_later
@spec_state_test
def test_invalid_exceed_max_blobs_per_block(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=get_max_blob_count(spec) + 1, valid=False)

@with_deneb_and_later
@spec_state_test
def test_mix_blob_tx_and_non_blob_tx(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    yield from run_block_with_blobs(spec, state, blob_count=1, tx_count=1, non_blob_tx_count=1)
