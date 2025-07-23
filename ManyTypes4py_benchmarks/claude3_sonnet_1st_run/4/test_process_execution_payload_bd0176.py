from random import Random
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any, Callable

from eth2spec.test.helpers.execution_payload import build_empty_execution_payload, compute_el_block_hash, get_execution_payload_header
from eth2spec.test.context import spec_state_test, expect_assertion_error, with_deneb_and_later
from eth2spec.test.helpers.blob import get_sample_blob_tx, get_max_blob_count

def run_execution_payload_processing(
    spec: Any, 
    state: Any, 
    execution_payload: Any, 
    blob_kzg_commitments: List[bytes], 
    valid: bool = True, 
    execution_valid: bool = True
) -> Iterator[Tuple[str, Any]]:
    """
    Run ``process_execution_payload``, yielding:
      - pre-state ('pre')
      - execution payload ('execution_payload')
      - execution details, to mock EVM execution ('execution.yml', a dict with 'execution_valid' key and boolean value)
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    body = spec.BeaconBlockBody(blob_kzg_commitments=blob_kzg_commitments, execution_payload=execution_payload)
    yield ('pre', state)
    yield ('execution', {'execution_valid': execution_valid})
    yield ('body', body)
    called_new_block: bool = False

    class TestEngine(spec.NoopExecutionEngine):

        def verify_and_notify_new_payload(self, new_payload_request: Any) -> bool:
            nonlocal called_new_block, execution_valid
            called_new_block = True
            assert new_payload_request.execution_payload == body.execution_payload
            return execution_valid
    if not valid:
        expect_assertion_error(lambda: spec.process_execution_payload(state, body, TestEngine()))
        yield ('post', None)
        return
    spec.process_execution_payload(state, body, TestEngine())
    assert called_new_block
    yield ('post', state)
    assert state.latest_execution_payload_header == get_execution_payload_header(spec, body.execution_payload)
'\nTests with incorrect blob transactions in the execution payload, but the execution client returns\nVALID, and the purpose of these tests is that the beacon client must not reject the block by\nattempting to do a validation of its own.\n'

@with_deneb_and_later
@spec_state_test
def test_incorrect_blob_tx_type(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = b'\x04' + opaque_tx[1:]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_1_extra_byte(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx + b'\x12'
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_1_byte_short(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx[:-1]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_empty(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx[0:0]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_32_extra_bytes(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx + b'\x12' * 32
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_no_transactions_with_commitments(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    _, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    execution_payload.transactions = []
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_commitment(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    blob_kzg_commitments[0] = b'\x12' * 48
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_no_commitments_for_transactions(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=2, rng=Random(1111))
    blob_kzg_commitments = []
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_commitments_order(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=2, rng=Random(1111))
    blob_kzg_commitments = [blob_kzg_commitments[1], blob_kzg_commitments[0]]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_no_blobs_but_with_commitments(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, _, _ = get_sample_blob_tx(spec, blob_count=0, rng=Random(1111))
    _, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=2, rng=Random(1112))
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_block_hash(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = b'\x12' * 32
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_zeroed_commitment(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The blob is invalid, but the commitment is in correct form.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=1, is_valid_blob=False)
    assert all((commitment == b'\x00' * 48 for commitment in blob_kzg_commitments))
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_invalid_correct_input__execution_invalid(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments, valid=False, execution_valid=False)

@with_deneb_and_later
@spec_state_test
def test_invalid_exceed_max_blobs_per_block(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=get_max_blob_count(spec) + 1)
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments, valid=False)
