"""
Utility code to generate randomized block tests
"""
import sys
import warnings
from random import Random
from typing import Callable, Dict, List, Any, Tuple, Iterator, Union

import eth2spec.test.helpers.execution_payload
import eth2spec.test.helpers.multi_operations
import eth2spec.test.helpers.inactivity_scores
import eth2spec.test.helpers.random
import eth2spec.test.helpers.blob
import eth2spec.test.helpers.state

def _randomize_deposit_state(spec, state, stats) -> Dict[str, List[Any]]:
    rng = Random(999)
    block_count = stats.get('block_count', 0)
    deposits = []
    if block_count > 0:
        num_deposits = rng.randrange(1, block_count * spec.MAX_DEPOSITS)
        deposits = eth2spec.test.helpers.random.prepare_state_and_get_random_deposits(spec, state, rng, num_deposits=num_deposits)
    return {'deposits': deposits}

def randomize_state(spec, state, stats, exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    eth2spec.test.helpers.random.randomize_state_helper(spec, state, exit_fraction=exit_fraction, slash_fraction=slash_fraction)
    scenario_state = _randomize_deposit_state(spec, state, stats)
    return scenario_state

def randomize_state_altair(spec, state, stats, exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    scenario_state = randomize_state(spec, state, stats, exit_fraction=exit_fraction, slash_fraction=slash_fraction)
    eth2spec.test.helpers.inactivity_scores.randomize_inactivity_scores(spec, state)
    return scenario_state

def randomize_state_bellatrix(spec, state, stats, exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    scenario_state = randomize_state_altair(spec, state, stats, exit_fraction=exit_fraction, slash_fraction=slash_fraction)
    return scenario_state

def randomize_state_capella(spec, state, stats, exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    scenario_state = randomize_state_bellatrix(spec, state, stats, exit_fraction=exit_fraction, slash_fraction=slash_fraction)
    return scenario_state

def randomize_state_deneb(spec, state, stats, exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    scenario_state = randomize_state_capella(spec, state, stats, exit_fraction=exit_fraction, slash_fraction=slash_fraction)
    return scenario_state

def randomize_state_electra(spec, state, stats, exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    scenario_state = randomize_state_deneb(spec, state, stats, exit_fraction=exit_fraction, slash_fraction=slash_fraction)
    return scenario_state

def randomize_state_fulu(spec, state, stats, exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    scenario_state = randomize_state_electra(spec, state, stats, exit_fraction=exit_fraction, slash_fraction=slash_fraction)
    return scenario_state

def epochs_until_leak(spec) -> int:
    return spec.MIN_EPOCHS_TO_INACTIVITY_PENALTY + 1

def epochs_for_shard_committee_period(spec) -> int:
    return spec.config.SHARD_COMMITTEE_PERIOD

def last_slot_in_epoch(spec) -> int:
    return spec.SLOTS_PER_EPOCH - 1

def random_slot_in_epoch(spec, rng: Random = Random(1336)) -> int:
    return rng.randrange(1, spec.SLOTS_PER_EPOCH - 2)

def penultimate_slot_in_epoch(spec) -> int:
    return spec.SLOTS_PER_EPOCH - 2

def no_block(_spec, _pre_state, _signed_blocks, _scenario_state) -> None:
    return None

BLOCK_ATTEMPTS = 32

def _warn_if_empty_operations(block) -> None:
    if len(block.body.proposer_slashings) == 0:
        warnings.warn(f'proposer slashings missing in block at slot {block.slot}')
    if len(block.body.attester_slashings) == 0:
        warnings.warn(f'attester slashings missing in block at slot {block.slot}')
    if len(block.body.attestations) == 0:
        warnings.warn(f'attestations missing in block at slot {block.slot}')
    if len(block.body.voluntary_exits) == 0:
        warnings.warn(f'voluntary exits missing in block at slot {block.slot}')

def _pull_deposits_from_scenario_state(spec, scenario_state, existing_block_count) -> List[Any]:
    all_deposits = scenario_state.get('deposits', [])
    start = existing_block_count * spec.MAX_DEPOSITS
    return all_deposits[start:start + spec.MAX_DEPOSITS]

def random_block(spec, state, signed_blocks, scenario_state) -> Any:
    temp_state = state.copy()
    eth2spec.test.helpers.state.next_slot(spec, temp_state)
    for _ in range(BLOCK_ATTEMPTS):
        proposer_index = spec.get_beacon_proposer_index(temp_state)
        proposer = state.validators[proposer_index]
        if proposer.slashed:
            eth2spec.test.helpers.state.next_slot(spec, state)
            eth2spec.test.helpers.state.next_slot(spec, temp_state)
        else:
            deposits_for_block = _pull_deposits_from_scenario_state(spec, scenario_state, len(signed_blocks))
            block = eth2spec.test.helpers.multi_operations.build_random_block_from_state_for_next_slot(spec, state, deposits=deposits_for_block)
            _warn_if_empty_operations(block)
            return block
    else:
        raise AssertionError('could not find a block with an unslashed proposer, check ``state`` input')

SYNC_AGGREGATE_PARTICIPATION_BUCKETS = 4

def random_block_altair_with_cycling_sync_committee_participation(spec, state, signed_blocks, scenario_state) -> Any:
    block = random_block(spec, state, signed_blocks, scenario_state)
    block_index = len(signed_blocks) % SYNC_AGGREGATE_PARTICIPATION_BUCKETS
    fraction_missed = block_index * (1 / SYNC_AGGREGATE_PARTICIPATION_BUCKETS)
    fraction_participated = 1.0 - fraction_missed
    previous_root = block.parent_root
    block.body.sync_aggregate = eth2spec.test.helpers.multi_operations.get_random_sync_aggregate(spec, state, block.slot - 1, block_root=previous_root, fraction_participated=fraction_participated)
    return block

def random_block_bellatrix(spec, state, signed_blocks, scenario_state, rng: Random = Random(3456)) -> Any:
    block = random_block_altair_with_cycling_sync_committee_participation(spec, state, signed_blocks, scenario_state)
    state = state.copy()
    eth2spec.test.helpers.state.next_slot(spec, state)
    block.body.execution_payload = eth2spec.test.helpers.execution_payload.build_randomized_execution_payload(spec, state, rng=rng)
    return block

def random_block_capella(spec, state, signed_blocks, scenario_state, rng: Random = Random(3456)) -> Any:
    block = random_block_bellatrix(spec, state, signed_blocks, scenario_state, rng=rng)
    block.body.bls_to_execution_changes = eth2spec.test.helpers.multi_operations.get_random_bls_to_execution_changes(spec, state, num_address_changes=rng.randint(1, spec.MAX_BLS_TO_EXECUTION_CHANGES))
    return block

def random_block_deneb(spec, state, signed_blocks, scenario_state, rng: Random = Random(3456)) -> Any:
    block = random_block_capella(spec, state, signed_blocks, scenario_state, rng=rng)
    opaque_tx, _, blob_kzg_commitments, _ = eth2spec.test.helpers.blob.get_sample_blob_tx(spec, blob_count=rng.randint(0, spec.config.MAX_BLOBS_PER_BLOCK), rng=rng)
    block.body.execution_payload.transactions.append(opaque_tx)
    block.body.execution_payload.block_hash = eth2spec.test.helpers.execution_payload.compute_el_block_hash_for_block(spec, block)
    block.body.blob_kzg_commitments = blob_kzg_commitments
    return block

def random_block_electra(spec, state, signed_blocks, scenario_state, rng: Random = Random(3456)) -> Any:
    block = random_block_deneb(spec, state, signed_blocks, scenario_state, rng=rng)
    block.body.execution_requests = eth2spec.test.helpers.multi_operations.get_random_execution_requests(spec, state, rng=rng)
    block.body.execution_payload.block_hash = eth2spec.test.helpers.execution_payload.compute_el_block_hash_for_block(spec, block)
    return block

def random_block_fulu(spec, state, signed_blocks, scenario_state, rng: Random = Random(3456)) -> Any:
    block = random_block_electra(spec, state, signed_blocks, scenario_state, rng=rng)
    return block

def no_op_validation(_spec, _state) -> bool:
    return True

def validate_is_leaking(spec, state) -> bool:
    return eth2spec.spec.is_in_inactivity_leak(state)

def validate_is_not_leaking(spec, state) -> bool:
    return not validate_is_leaking(spec, state)

def with_validation(transition: Callable, validation: Callable) -> Dict[str, Callable]:
    if isinstance(transition, Callable):
        transition = transition()
    transition['validation'] = validation
    return transition

def no_op_transition() -> Dict:
    return {}

def epoch_transition(n: int = 0) -> Dict[str, int]:
    return {'epochs_to_skip': n}

def slot_transition(n: int = 0) -> Dict[str, int]:
    return {'slots_to_skip': n}

def transition_to_leaking() -> Dict[str, Union[Callable, str]]:
    return {'epochs_to_skip': epochs_until_leak, 'validation': validate_is_leaking}

transition_without_leak = with_validation(no_op_transition, validate_is_not_leaking)

def transition_with_random_block(block_randomizer: Callable) -> Dict[str, Callable]:
    return {'block_producer': block_randomizer}

def _randomized_scenario_setup(state_randomizer: Callable) -> Tuple[Tuple[Callable, Callable], ...]:
    def _skip_epochs(epoch_producer: Callable) -> Callable:
        def f(spec, state, _stats) -> None:
            epochs_to_skip = epoch_producer(spec)
            slots_to_skip = epochs_to_skip * spec.SLOTS_PER_EPOCH
            state.slot += slots_to_skip
        return f

    def _simulate_honest_execution(spec, state, _stats) -> None:
        eth2spec.test.helpers.random.patch_state_to_non_leaking(spec, state)
    return ((_skip_epochs(epochs_for_shard_committee_period), no_op_validation), (_simulate_honest_execution, no_op_validation), (state_randomizer, eth2spec.test.helpers.state.ensure_state_has_validators_across_lifecycle))

_this_module = sys.modules[__name__]

def _resolve_ref(ref: Union[str, Callable]) -> Callable:
    if isinstance(ref, str):
        return getattr(_this_module, ref)
    return ref

def _iter_temporal(spec, description: Union[int, Callable]) -> Iterator[int]:
    numeric = _resolve_ref(description)
    if isinstance(numeric, Callable):
        numeric = numeric(spec)
    for i in range(numeric):
        yield i

def _compute_statistics(scenario: Dict) -> Dict[str, int]:
    block_count = 0
    for transition in scenario['transitions']:
        block_producer = _resolve_ref(transition.get('block_producer', None))
        if block_producer and block_producer != no_block:
            block_count += 1
    return {'block_count': block_count}

def run_generated_randomized_test(spec, state, scenario: Dict) -> Iterator[Tuple[str, Union[Any, List[Any], Any]]]:
    stats = _compute_statistics(scenario)
    if 'setup' not in scenario:
        state_randomizer = _resolve_ref(scenario.get('state_randomizer', randomize_state))
        scenario['setup'] = _randomized_scenario_setup(state_randomizer)
    scenario_state = {}
    for mutation, validation in scenario['setup']:
        additional_state = mutation(spec, state, stats)
        validation(spec, state)
        if additional_state:
            scenario_state.update(additional_state)
    yield ('pre', state)
    blocks = []
    for transition in scenario['transitions']:
        epochs_to_skip = _iter_temporal(spec, transition['epochs_to_skip'])
        for _ in epochs_to_skip:
            eth2spec.test.helpers.state.next_epoch(spec, state)
        slots_to_skip = _iter_temporal(spec, transition['slots_to_skip'])
        for _ in slots_to_skip:
            eth2spec.test.helpers.state.next_slot(spec, state)
        block_producer = _resolve_ref(transition['block_producer'])
        block = block_producer(spec, state, blocks, scenario_state)
        if block:
            signed_block = eth2spec.test.helpers.state.state_transition_and_sign_block(spec, state, block)
            blocks.append(signed_block)
        validation = _resolve_ref(transition['validation'])
        assert validation(spec, state)
    yield ('blocks', blocks)
    yield ('post', state)