from eth2spec.test.helpers.constants import CAPELLA, DENEB, ELECTRA
from eth2spec.test.helpers.fork_transition import transition_across_forks
from eth2spec.test.helpers.forks import is_post_capella, is_post_deneb, is_post_electra
from eth2spec.test.helpers.sync_committee import compute_aggregate_sync_committee_signature, compute_committee_indices
from math import floor

def latest_finalized_root_gindex(spec: typing.Callable):
    if hasattr(spec, 'FINALIZED_ROOT_GINDEX_ELECTRA'):
        return spec.FINALIZED_ROOT_GINDEX_ELECTRA
    return spec.FINALIZED_ROOT_GINDEX

def latest_current_sync_committee_gindex(spec: Union[typing.Callable, dict[str, typing.Any]]):
    if hasattr(spec, 'CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA'):
        return spec.CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA
    return spec.CURRENT_SYNC_COMMITTEE_GINDEX

def latest_next_sync_committee_gindex(spec: Union[Atom, dict]):
    if hasattr(spec, 'NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA'):
        return spec.NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA
    return spec.NEXT_SYNC_COMMITTEE_GINDEX

def latest_normalize_merkle_branch(spec: Union[str, dict[str, bool], None, typing.Sequence[str]], branch: str, gindex: str) -> str:
    if hasattr(spec, 'normalize_merkle_branch'):
        return spec.normalize_merkle_branch(branch, gindex)
    return branch

def compute_start_slot_at_sync_committee_period(spec: int, sync_committee_period: int):
    return spec.compute_start_slot_at_epoch(sync_committee_period * spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD)

def compute_start_slot_at_next_sync_committee_period(spec: float, state: Union[dict[str, "Outcome"], int]):
    sync_committee_period = spec.compute_sync_committee_period_at_slot(state.slot)
    return compute_start_slot_at_sync_committee_period(spec, sync_committee_period + 1)

def get_sync_aggregate(spec: Union[str, tuple[int], typing.Callable], state: Any, num_participants: Union[None, int, str, tuple[float]]=None, signature_slot: None=None, phases: Union[None, str, tuple[int], typing.Callable]=None) -> tuple[tuple]:
    if signature_slot is None:
        signature_slot = state.slot + 1
    assert signature_slot > state.slot
    signature_spec, signature_state, _ = transition_across_forks(spec, state, signature_slot, phases)
    committee_indices = compute_committee_indices(signature_state)
    committee_size = len(committee_indices)
    if num_participants is None:
        num_participants = committee_size
    assert committee_size >= num_participants >= 0
    sync_committee_bits = [True] * num_participants + [False] * (committee_size - num_participants)
    sync_committee_signature = compute_aggregate_sync_committee_signature(signature_spec, signature_state, max(signature_slot, 1) - 1, committee_indices[:num_participants])
    sync_aggregate = signature_spec.SyncAggregate(sync_committee_bits=sync_committee_bits, sync_committee_signature=sync_committee_signature)
    return (sync_aggregate, signature_slot)

def create_update(spec: Union[bool, list['ValidatorRecord'], dict[str, typing.Any]], attested_state: Union[bool, bytes, str], attested_block: Union[bool, typing.Any, None, typing.Callable], finalized_block: Union[bool, typing.Any, None, list['ValidatorRecord']], with_next: Union[bool, typing.Mapping, typing.Iterable[str]], with_finality: Union[bool, typing.Mapping, typing.Iterable[str]], participation_rate: bool, signature_slot: Union[None, bool, typing.Any, list['ValidatorRecord']]=None) -> Union[str, dict, dict[str, dict[str, str]]]:
    num_participants = floor(spec.SYNC_COMMITTEE_SIZE * participation_rate)
    update = spec.LightClientUpdate()
    update.attested_header = spec.block_to_light_client_header(attested_block)
    if with_next:
        update.next_sync_committee = attested_state.next_sync_committee
        update.next_sync_committee_branch = spec.compute_merkle_proof(attested_state, latest_next_sync_committee_gindex(spec))
    if with_finality:
        update.finalized_header = spec.block_to_light_client_header(finalized_block)
        update.finality_branch = spec.compute_merkle_proof(attested_state, latest_finalized_root_gindex(spec))
    update.sync_aggregate, update.signature_slot = get_sync_aggregate(spec, attested_state, num_participants, signature_slot=signature_slot)
    return update

def needs_upgrade_to_capella(spec: Union[dict[str, typing.Any], str, typing.Mapping], new_spec: Union[dict[str, typing.Any], str, typing.Mapping]) -> bool:
    return is_post_capella(new_spec) and (not is_post_capella(spec))

def needs_upgrade_to_deneb(spec: Union[dict[str, typing.Any], str, typing.Mapping], new_spec: Union[dict[str, typing.Any], str, typing.Mapping]) -> bool:
    return is_post_deneb(new_spec) and (not is_post_deneb(spec))

def needs_upgrade_to_electra(spec: Union[dict[str, typing.Any], str, typing.Mapping], new_spec: Union[dict[str, typing.Any], str, typing.Mapping]) -> bool:
    return is_post_electra(new_spec) and (not is_post_electra(spec))

def check_merkle_branch_equal(spec: Union[str, None, bool], new_spec: Union[dict, typing.Mapping, dict[str, typing.Any]], data: Union[dict, dict[str, typing.Any], str, typing.Any, None], upgraded: Union[dict, dict[str, typing.Any], str, typing.Any, None], gindex: Union[dict, typing.Type, typing.Sequence[typing.Type]]) -> None:
    if is_post_electra(new_spec):
        assert new_spec.normalize_merkle_branch(upgraded, gindex) == new_spec.normalize_merkle_branch(data, gindex)
    else:
        assert upgraded == data

def check_lc_header_equal(spec: Union[dict, dict[str, typing.Any], str], new_spec: Union[dict[str, typing.Any], dict, typing.Mapping], data: Union[str, dict, typing.Mapping], upgraded: Union[str, dict, typing.Mapping]) -> None:
    assert upgraded.beacon.slot == data.beacon.slot
    assert upgraded.beacon.hash_tree_root() == data.beacon.hash_tree_root()
    if is_post_capella(new_spec):
        if is_post_capella(spec):
            assert new_spec.get_lc_execution_root(upgraded) == spec.get_lc_execution_root(data)
        else:
            assert new_spec.get_lc_execution_root(upgraded) == new_spec.Root()

def upgrade_lc_header_to_new_spec(spec: Union[str, dict, dict[str, dict[str, typing.Any]]], new_spec: Union[str, dict, dict[str, dict[str, typing.Any]]], data: Union[dict, str, float], phases: Union[tuple[list[typing.Any]], dict, list[str]]) -> Union[dict, str, float, dict[typing.Any, dict[typing.Any, str]]]:
    upgraded = data
    if needs_upgrade_to_capella(spec, new_spec):
        upgraded = phases[CAPELLA].upgrade_lc_header_to_capella(upgraded)
        check_lc_header_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_deneb(spec, new_spec):
        upgraded = phases[DENEB].upgrade_lc_header_to_deneb(upgraded)
        check_lc_header_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_electra(spec, new_spec):
        upgraded = phases[ELECTRA].upgrade_lc_header_to_electra(upgraded)
        check_lc_header_equal(spec, new_spec, data, upgraded)
    return upgraded

def check_lc_bootstrap_equal(spec: Union[dict, bytes, typing.Mapping], new_spec: Union[dict, bytes, typing.Mapping], data: Union[str, dict[str, typing.Any], dict, None], upgraded: Union[str, dict[str, typing.Any], dict, None]) -> None:
    check_lc_header_equal(spec, new_spec, data.header, upgraded.header)
    assert upgraded.current_sync_committee == data.current_sync_committee
    check_merkle_branch_equal(spec, new_spec, data.current_sync_committee_branch, upgraded.current_sync_committee_branch, latest_current_sync_committee_gindex(new_spec))

def upgrade_lc_bootstrap_to_new_spec(spec: Union[str, typing.Mapping, dict], new_spec: Union[str, typing.Mapping, dict], data: Union[dict, str, typing.Type], phases: Union[dict, list[str]]) -> Union[dict, str, typing.Type, dict[str, str], dict[typing.Any, dict[typing.Any, str]]]:
    upgraded = data
    if needs_upgrade_to_capella(spec, new_spec):
        upgraded = phases[CAPELLA].upgrade_lc_bootstrap_to_capella(upgraded)
        check_lc_bootstrap_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_deneb(spec, new_spec):
        upgraded = phases[DENEB].upgrade_lc_bootstrap_to_deneb(upgraded)
        check_lc_bootstrap_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_electra(spec, new_spec):
        upgraded = phases[ELECTRA].upgrade_lc_bootstrap_to_electra(upgraded)
        check_lc_bootstrap_equal(spec, new_spec, data, upgraded)
    return upgraded

def check_lc_update_equal(spec: Union[dict, dict[str, typing.Any], typing.Sequence], new_spec: Union[dict, dict[str, typing.Any], typing.Sequence], data: Union[typing.Sequence, dict, None], upgraded: Union[typing.Sequence, dict, None]) -> None:
    check_lc_header_equal(spec, new_spec, data.attested_header, upgraded.attested_header)
    assert upgraded.next_sync_committee == data.next_sync_committee
    check_merkle_branch_equal(spec, new_spec, data.next_sync_committee_branch, upgraded.next_sync_committee_branch, latest_next_sync_committee_gindex(new_spec))
    check_lc_header_equal(spec, new_spec, data.finalized_header, upgraded.finalized_header)
    check_merkle_branch_equal(spec, new_spec, data.finality_branch, upgraded.finality_branch, latest_finalized_root_gindex(new_spec))
    assert upgraded.sync_aggregate == data.sync_aggregate
    assert upgraded.signature_slot == data.signature_slot

def upgrade_lc_update_to_new_spec(spec: Union[str, dict, dict[str, typing.Any]], new_spec: Union[str, dict, dict[str, typing.Any]], data: Union[str, dict, int], phases: Union[str, bool, dict[str, typing.Any]]) -> Union[str, dict, int, dict[str, str]]:
    upgraded = data
    if needs_upgrade_to_capella(spec, new_spec):
        upgraded = phases[CAPELLA].upgrade_lc_update_to_capella(upgraded)
        check_lc_update_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_deneb(spec, new_spec):
        upgraded = phases[DENEB].upgrade_lc_update_to_deneb(upgraded)
        check_lc_update_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_electra(spec, new_spec):
        upgraded = phases[ELECTRA].upgrade_lc_update_to_electra(upgraded)
        check_lc_update_equal(spec, new_spec, data, upgraded)
    return upgraded

def check_lc_finality_update_equal(spec: Union[dict, dict[str, typing.Any], typing.Sequence], new_spec: Union[dict, dict[str, typing.Any], typing.Sequence], data: Union[dict, dict[str, typing.Any], typing.Sequence], upgraded: Union[dict, dict[str, typing.Any], typing.Sequence]) -> None:
    check_lc_header_equal(spec, new_spec, data.attested_header, upgraded.attested_header)
    check_lc_header_equal(spec, new_spec, data.finalized_header, upgraded.finalized_header)
    check_merkle_branch_equal(spec, new_spec, data.finality_branch, upgraded.finality_branch, latest_finalized_root_gindex(new_spec))
    assert upgraded.sync_aggregate == data.sync_aggregate
    assert upgraded.signature_slot == data.signature_slot

def upgrade_lc_finality_update_to_new_spec(spec: Union[str, dict, dict[str, typing.Any]], new_spec: Union[str, dict, dict[str, typing.Any]], data: Union[str, dict, int], phases: Union[str, int]) -> Union[str, dict, int, dict[str, str]]:
    upgraded = data
    if needs_upgrade_to_capella(spec, new_spec):
        upgraded = phases[CAPELLA].upgrade_lc_finality_update_to_capella(upgraded)
        check_lc_finality_update_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_deneb(spec, new_spec):
        upgraded = phases[DENEB].upgrade_lc_finality_update_to_deneb(upgraded)
        check_lc_finality_update_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_electra(spec, new_spec):
        upgraded = phases[ELECTRA].upgrade_lc_finality_update_to_electra(upgraded)
        check_lc_finality_update_equal(spec, new_spec, data, upgraded)
    return upgraded

def check_lc_store_equal(spec: Union[dict, typing.Mapping, bytes], new_spec: Union[dict, typing.Mapping, bytes], data: Union[dict, bytes, dict[str, typing.Any]], upgraded: Union[dict, bytes, dict[str, typing.Any]]) -> None:
    check_lc_header_equal(spec, new_spec, data.finalized_header, upgraded.finalized_header)
    assert upgraded.current_sync_committee == data.current_sync_committee
    assert upgraded.next_sync_committee == data.next_sync_committee
    if upgraded.best_valid_update is None:
        assert data.best_valid_update is None
    else:
        check_lc_update_equal(spec, new_spec, data.best_valid_update, upgraded.best_valid_update)
    check_lc_header_equal(spec, new_spec, data.optimistic_header, upgraded.optimistic_header)
    assert upgraded.previous_max_active_participants == data.previous_max_active_participants
    assert upgraded.current_max_active_participants == data.current_max_active_participants

def upgrade_lc_store_to_new_spec(spec: Union[str, dict, typing.Mapping], new_spec: Union[str, dict, typing.Mapping], data: Union[dict, float, typing.Callable[T, typing.Any]], phases: Union[dict, typing.Callable[str,str, float]]) -> Union[dict, float, typing.Callable[T, typing.Any]]:
    upgraded = data
    if needs_upgrade_to_capella(spec, new_spec):
        upgraded = phases[CAPELLA].upgrade_lc_store_to_capella(upgraded)
        check_lc_store_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_deneb(spec, new_spec):
        upgraded = phases[DENEB].upgrade_lc_store_to_deneb(upgraded)
        check_lc_store_equal(spec, new_spec, data, upgraded)
    if needs_upgrade_to_electra(spec, new_spec):
        upgraded = phases[ELECTRA].upgrade_lc_store_to_electra(upgraded)
        check_lc_store_equal(spec, new_spec, data, upgraded)
    return upgraded