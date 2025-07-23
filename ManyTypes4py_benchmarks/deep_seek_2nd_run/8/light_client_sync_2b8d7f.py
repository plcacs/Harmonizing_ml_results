from typing import Any, Dict, List, Optional, Tuple, Union, Generator, Iterator
from eth_utils import encode_hex
from eth2spec.test.helpers.attestations import next_slots_with_attestations, state_transition_with_full_block
from eth2spec.test.helpers.fork_transition import do_fork, transition_across_forks
from eth2spec.test.helpers.forks import get_spec_for_fork_version, is_post_capella, is_post_deneb, is_post_electra
from eth2spec.test.helpers.light_client import get_sync_aggregate, upgrade_lc_bootstrap_to_new_spec, upgrade_lc_update_to_new_spec, upgrade_lc_store_to_new_spec
from eth2spec.test.helpers.state import next_slots, transition_to
from eth2spec.phase0 import spec as spec_phase0
from eth2spec.altair import spec as spec_altair
from eth2spec.bellatrix import spec as spec_bellatrix
from eth2spec.capella import spec as spec_capella
from eth2spec.deneb import spec as spec_deneb
from eth2spec.electra import spec as spec_electra

class LightClientSyncTest(object):
    steps: List[Dict[str, Any]]
    s_spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]
    genesis_validators_root: bytes
    store: Any

def _get_store_fork_version(s_spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]) -> bytes:
    if is_post_electra(s_spec):
        return s_spec.config.ELECTRA_FORK_VERSION
    if is_post_deneb(s_spec):
        return s_spec.config.DENEB_FORK_VERSION
    if is_post_capella(s_spec):
        return s_spec.config.CAPELLA_FORK_VERSION
    return s_spec.config.ALTAIR_FORK_VERSION

def setup_lc_sync_test(spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], state: Any, s_spec: Optional[Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]] = None, phases: Optional[Dict[str, Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]]] = None) -> Generator[Tuple[str, str, str], None, LightClientSyncTest]:
    test = LightClientSyncTest()
    test.steps = []
    if s_spec is None:
        s_spec = spec
    if phases is None:
        phases = {spec.fork: spec, s_spec.fork: s_spec}
    test.s_spec = s_spec
    yield ('genesis_validators_root', 'meta', '0x' + state.genesis_validators_root.hex())
    test.genesis_validators_root = state.genesis_validators_root
    next_slots(spec, state, spec.SLOTS_PER_EPOCH * 2 - 1)
    trusted_block = state_transition_with_full_block(spec, state, True, True)
    trusted_block_root = trusted_block.message.hash_tree_root()
    yield ('trusted_block_root', 'meta', '0x' + trusted_block_root.hex())
    data_fork_version = spec.compute_fork_version(spec.compute_epoch_at_slot(trusted_block.message.slot))
    data_fork_digest = spec.compute_fork_digest(data_fork_version, test.genesis_validators_root)
    d_spec = get_spec_for_fork_version(spec, data_fork_version, phases)
    data = d_spec.create_light_client_bootstrap(state, trusted_block)
    yield ('bootstrap_fork_digest', 'meta', encode_hex(data_fork_digest))
    yield ('bootstrap', data)
    upgraded = upgrade_lc_bootstrap_to_new_spec(d_spec, test.s_spec, data, phases)
    test.store = test.s_spec.initialize_light_client_store(trusted_block_root, upgraded)
    store_fork_version = _get_store_fork_version(test.s_spec)
    store_fork_digest = test.s_spec.compute_fork_digest(store_fork_version, test.genesis_validators_root)
    yield ('store_fork_digest', 'meta', encode_hex(store_fork_digest))
    return test

def finish_lc_sync_test(test: LightClientSyncTest) -> Generator[Tuple[str, List[Dict[str, Any]]], None, None]:
    yield ('steps', test.steps)

def _get_update_file_name(d_spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], update: Any) -> str:
    if d_spec.is_sync_committee_update(update):
        suffix1 = 's'
    else:
        suffix1 = 'x'
    if d_spec.is_finality_update(update):
        suffix2 = 'f'
    else:
        suffix2 = 'x'
    return f'update_{encode_hex(update.attested_header.beacon.hash_tree_root())}_{suffix1}{suffix2}'

def _get_checks(s_spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], store: Any) -> Dict[str, Dict[str, Union[int, str]]]:
    if is_post_capella(s_spec):
        return {'finalized_header': {'slot': int(store.finalized_header.beacon.slot), 'beacon_root': encode_hex(store.finalized_header.beacon.hash_tree_root()), 'execution_root': encode_hex(s_spec.get_lc_execution_root(store.finalized_header))}, 'optimistic_header': {'slot': int(store.optimistic_header.beacon.slot), 'beacon_root': encode_hex(store.optimistic_header.beacon.hash_tree_root()), 'execution_root': encode_hex(s_spec.get_lc_execution_root(store.optimistic_header))}}
    return {'finalized_header': {'slot': int(store.finalized_header.beacon.slot), 'beacon_root': encode_hex(store.finalized_header.beacon.hash_tree_root())}, 'optimistic_header': {'slot': int(store.optimistic_header.beacon.slot), 'beacon_root': encode_hex(store.optimistic_header.beacon.hash_tree_root())}}

def emit_force_update(test: LightClientSyncTest, spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], state: Any) -> Iterator[None]:
    current_slot = state.slot
    test.s_spec.process_light_client_store_force_update(test.store, current_slot)
    yield from []
    test.steps.append({'force_update': {'current_slot': int(current_slot), 'checks': _get_checks(test.s_spec, test.store)}})

def emit_update(test: LightClientSyncTest, spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], state: Any, block: Any, attested_state: Any, attested_block: Any, finalized_block: Any, with_next: bool = True, phases: Optional[Dict[str, Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]]] = None) -> Generator[Union[Tuple[str, Any], None], None, Any]:
    data_fork_version = spec.compute_fork_version(spec.compute_epoch_at_slot(attested_block.message.slot))
    data_fork_digest = spec.compute_fork_digest(data_fork_version, test.genesis_validators_root)
    d_spec = get_spec_for_fork_version(spec, data_fork_version, phases)
    data = d_spec.create_light_client_update(state, block, attested_state, attested_block, finalized_block)
    if not with_next:
        data.next_sync_committee = spec.SyncCommittee()
        data.next_sync_committee_branch = spec.NextSyncCommitteeBranch()
    current_slot = state.slot
    upgraded = upgrade_lc_update_to_new_spec(d_spec, test.s_spec, data, phases)
    test.s_spec.process_light_client_update(test.store, upgraded, current_slot, test.genesis_validators_root)
    yield (_get_update_file_name(d_spec, data), data)
    test.steps.append({'process_update': {'update_fork_digest': encode_hex(data_fork_digest), 'update': _get_update_file_name(d_spec, data), 'current_slot': int(current_slot), 'checks': _get_checks(test.s_spec, test.store)}})
    return upgraded

def _emit_upgrade_store(test: LightClientSyncTest, new_s_spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], phases: Optional[Dict[str, Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]]] = None) -> Iterator[None]:
    test.store = upgrade_lc_store_to_new_spec(test.s_spec, new_s_spec, test.store, phases)
    test.s_spec = new_s_spec
    store_fork_version = _get_store_fork_version(test.s_spec)
    store_fork_digest = test.s_spec.compute_fork_digest(store_fork_version, test.genesis_validators_root)
    yield from []
    test.steps.append({'upgrade_store': {'store_fork_digest': encode_hex(store_fork_digest), 'checks': _get_checks(test.s_spec, test.store)}})

def run_lc_sync_test_single_fork(spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], phases: Dict[str, Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]], state: Any, fork: str) -> Generator[Union[Tuple[str, str, str], Tuple[str, Any], None], None, None]:
    test = (yield from setup_lc_sync_test(spec, state, phases=phases))
    finalized_block = spec.SignedBeaconBlock()
    finalized_block.message.state_root = state.hash_tree_root()
    finalized_state = state.copy()
    attested_block = state_transition_with_full_block(spec, state, True, True)
    attested_state = state.copy()
    sync_aggregate, _ = get_sync_aggregate(spec, state, phases=phases)
    block = state_transition_with_full_block(spec, state, True, True, sync_aggregate=sync_aggregate)
    yield from emit_update(test, spec, state, block, attested_state, attested_block, finalized_block, phases=phases)
    assert test.store.finalized_header.beacon.slot == finalized_state.slot
    assert test.store.next_sync_committee == finalized_state.next_sync_committee
    assert test.store.best_valid_update is None
    assert test.store.optimistic_header.beacon.slot == attested_state.slot
    fork_epoch = getattr(phases[fork].config, fork.upper() + '_FORK_EPOCH')
    transition_to(spec, state, spec.compute_start_slot_at_epoch(fork_epoch) - 4)
    attested_block = state_transition_with_full_block(spec, state, True, True)
    attested_state = state.copy()
    sync_aggregate, _ = get_sync_aggregate(spec, state, phases=phases)
    block = state_transition_with_full_block(spec, state, True, True, sync_aggregate=sync_aggregate)
    update = (yield from emit_update(test, spec, state, block, attested_state, attested_block, finalized_block, phases=phases))
    assert test.store.finalized_header.beacon.slot == finalized_state.slot
    assert test.store.next_sync_committee == finalized_state.next_sync_committee
    assert test.store.best_valid_update == update
    assert test.store.optimistic_header.beacon.slot == attested_state.slot
    yield from _emit_upgrade_store(test, phases[fork], phases=phases)
    update = test.store.best_valid_update
    attested_block = block.copy()
    attested_state = state.copy()
    sync_aggregate, _ = get_sync_aggregate(spec, state, phases=phases)
    block = state_transition_with_full_block(spec, state, True, True, sync_aggregate=sync_aggregate)
    yield from emit_update(test, spec, state, block, attested_state, attested_block, finalized_block, phases=phases)
    assert test.store.finalized_header.beacon.slot == finalized_state.slot
    assert test.store.next_sync_committee == finalized_state.next_sync_committee
    assert test.store.best_valid_update == update
    assert test.store.optimistic_header.beacon.slot == attested_state.slot
    attested_block = block.copy()
    attested_state = state.copy()
    sync_aggregate, _ = get_sync_aggregate(spec, state, phases=phases)
    state, block = do_fork(state, spec, phases[fork], fork_epoch, sync_aggregate=sync_aggregate)
    spec = phases[fork]
    yield from emit_update(test, spec, state, block, attested_state, attested_block, finalized_block, phases=phases)
    assert test.store.finalized_header.beacon.slot == finalized_state.slot
    assert test.store.next_sync_committee == finalized_state.next_sync_committee
    assert test.store.best_valid_update == update
    assert test.store.optimistic_header.beacon.slot == attested_state.slot
    attested_block = block.copy()
    attested_state = state.copy()
    sync_aggregate, _ = get_sync_aggregate(spec, state, phases=phases)
    block = state_transition_with_full_block(spec, state, True, True, sync_aggregate=sync_aggregate)
    yield from emit_update(test, spec, state, block, attested_state, attested_block, finalized_block, phases=phases)
    assert test.store.finalized_header.beacon.slot == finalized_state.slot
    assert test.store.next_sync_committee == finalized_state.next_sync_committee
    assert test.store.best_valid_update == update
    assert test.store.optimistic_header.beacon.slot == attested_state.slot
    transition_to(spec, state, spec.compute_start_slot_at_epoch(fork_epoch + 1) - 2)
    attested_block = state_transition_with_full_block(spec, state, True, True)
    attested_state = state.copy()
    sync_aggregate, _ = get_sync_aggregate(spec, state, phases=phases)
    block = state_transition_with_full_block(spec, state, True, True, sync_aggregate=sync_aggregate)
    yield from emit_update(test, spec, state, block, attested_state, attested_block, finalized_block, phases=phases)
    assert test.store.finalized_header.beacon.slot == finalized_state.slot
    assert test.store.next_sync_committee == finalized_state.next_sync_committee
    assert test.store.best_valid_update == update
    assert test.store.optimistic_header.beacon.slot == attested_state.slot
    finalized_block = block.copy()
    finalized_state = state.copy()
    _, _, state = next_slots_with_attestations(spec, state, 2 * spec.SLOTS_PER_EPOCH - 1, True, True)
    attested_block = state_transition_with_full_block(spec, state, True, True)
    attested_state = state.copy()
    sync_aggregate, _ = get_sync_aggregate(spec, state, phases=phases)
    block = state_transition_with_full_block(spec, state, True, True, sync_aggregate=sync_aggregate)
    yield from emit_update(test, spec, state, block, attested_state, attested_block, finalized_block, phases=phases)
    assert test.store.finalized_header.beacon.slot == finalized_state.slot
    assert test.store.next_sync_committee == finalized_state.next_sync_committee
    assert test.store.best_valid_update is None
    assert test.store.optimistic_header.beacon.slot == attested_state.slot
    yield from finish_lc_sync_test(test)

def run_lc_sync_test_multi_fork(spec: Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec], phases: Dict[str, Union[spec_phase0.Spec, spec_altair.Spec, spec_bellatrix.Spec, spec_capella.Spec, spec_deneb.Spec, spec_electra.Spec]], state: Any, fork_1: str