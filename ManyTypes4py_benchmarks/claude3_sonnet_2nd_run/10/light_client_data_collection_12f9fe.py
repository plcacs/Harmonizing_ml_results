from typing import Any, Dict, List, Set, Tuple, Iterator, Optional, Union
from dataclasses import dataclass
from eth_utils import encode_hex
from eth2spec.test.helpers.constants import ALTAIR
from eth2spec.test.helpers.fork_transition import transition_across_forks
from eth2spec.test.helpers.forks import is_post_altair
from eth2spec.test.helpers.light_client import compute_start_slot_at_sync_committee_period, get_sync_aggregate, latest_current_sync_committee_gindex, latest_finalized_root_gindex, latest_next_sync_committee_gindex, latest_normalize_merkle_branch, upgrade_lc_header_to_new_spec, upgrade_lc_update_to_new_spec

def _next_epoch_boundary_slot(spec: Any, slot: int) -> int:
    epoch = spec.compute_epoch_at_slot(slot + spec.SLOTS_PER_EPOCH - 1)
    return spec.compute_start_slot_at_epoch(epoch)

@dataclass(frozen=True)
class BlockID(object):
    slot: int
    root: bytes

def _block_to_block_id(block: Any) -> BlockID:
    return BlockID(slot=block.message.slot, root=block.message.hash_tree_root())

def _state_to_block_id(state: Any) -> BlockID:
    parent_header = state.latest_block_header.copy()
    parent_header.state_root = state.hash_tree_root()
    return BlockID(slot=parent_header.slot, root=parent_header.hash_tree_root())

def get_lc_bootstrap_block_id(bootstrap: Any) -> BlockID:
    return BlockID(slot=bootstrap.header.beacon.slot, root=bootstrap.header.beacon.hash_tree_root())

def get_lc_update_attested_block_id(update: Any) -> BlockID:
    return BlockID(slot=update.attested_header.beacon.slot, root=update.attested_header.beacon.hash_tree_root())

@dataclass
class ForkedBeaconState(object):
    spec: Any
    data: Any

@dataclass
class ForkedSignedBeaconBlock(object):
    spec: Any
    data: Any

@dataclass
class ForkedLightClientHeader(object):
    spec: Any
    data: Any

@dataclass
class ForkedLightClientBootstrap(object):
    spec: Optional[Any]
    data: Optional[Any]

@dataclass
class ForkedLightClientUpdate(object):
    spec: Optional[Any]
    data: Optional[Any]

@dataclass
class ForkedLightClientFinalityUpdate(object):
    spec: Optional[Any]
    data: Optional[Any]

@dataclass
class ForkedLightClientOptimisticUpdate(object):
    spec: Optional[Any]
    data: Optional[Any]

@dataclass
class CachedLightClientData(object):
    current_sync_committee_branch: List[bytes]
    next_sync_committee_branch: List[bytes]
    finalized_slot: int
    finality_branch: List[bytes]
    current_period_best_update: ForkedLightClientUpdate
    latest_signature_slot: int

@dataclass
class LightClientDataCache(object):
    data: Dict[BlockID, CachedLightClientData]
    latest: ForkedLightClientFinalityUpdate
    tail_slot: int

@dataclass
class LightClientDataDB(object):
    headers: Dict[bytes, ForkedLightClientHeader]
    current_branches: Dict[int, List[bytes]]
    sync_committees: Dict[int, Any]
    best_updates: Dict[int, ForkedLightClientUpdate]

@dataclass
class LightClientDataStore(object):
    spec: Any
    cache: LightClientDataCache
    db: LightClientDataDB

@dataclass
class LightClientDataCollectionTest(object):
    steps: List[Dict[str, Any]]
    files: Set[str]
    phases: Dict[str, Any]
    blocks: Dict[bytes, ForkedSignedBeaconBlock]
    finalized_block_roots: Dict[int, bytes]
    states: Dict[bytes, ForkedBeaconState]
    finalized_checkpoint_states: Dict[bytes, ForkedBeaconState]
    latest_finalized_epoch: int
    latest_finalized_bid: BlockID
    historical_tail_slot: int
    lc_data_store: LightClientDataStore

def get_ancestor_of_block_id(test: LightClientDataCollectionTest, bid: BlockID, slot: int) -> Optional[BlockID]:
    try:
        block = test.blocks[bid.root]
        while True:
            if block.data.message.slot <= slot:
                return _block_to_block_id(block.data)
            block = test.blocks[block.data.message.parent_root]
    except KeyError:
        return None

def _block_id_at_finalized_slot(test: LightClientDataCollectionTest, slot: int) -> Optional[BlockID]:
    while slot >= test.historical_tail_slot:
        try:
            return BlockID(slot=slot, root=test.finalized_block_roots[slot])
        except KeyError:
            slot = slot - 1
    return None

def _get_current_sync_committee_for_finalized_period(test: LightClientDataCollectionTest, period: int) -> Optional[Any]:
    low_slot = max(test.historical_tail_slot, test.lc_data_store.spec.compute_start_slot_at_epoch(test.lc_data_store.spec.config.ALTAIR_FORK_EPOCH))
    if period < test.lc_data_store.spec.compute_sync_committee_period_at_slot(low_slot):
        return None
    period_start_slot = compute_start_slot_at_sync_committee_period(test.lc_data_store.spec, period)
    sync_committee_slot = max(period_start_slot, low_slot)
    bid = _block_id_at_finalized_slot(test, sync_committee_slot)
    if bid is None:
        return None
    block = test.blocks[bid.root]
    state = test.finalized_checkpoint_states[block.data.message.state_root]
    if sync_committee_slot > state.data.slot:
        state.spec, state.data, _ = transition_across_forks(state.spec, state.data, sync_committee_slot, phases=test.phases)
    assert is_post_altair(state.spec)
    return state.data.current_sync_committee

def _light_client_header_for_block(test: LightClientDataCollectionTest, block: Any) -> ForkedLightClientHeader:
    if not is_post_altair(block.spec):
        spec = test.phases[ALTAIR]
    else:
        spec = block.spec
    return ForkedLightClientHeader(spec=spec, data=spec.block_to_light_client_header(block.data))

def _light_client_header_for_block_id(test: LightClientDataCollectionTest, bid: BlockID) -> ForkedLightClientHeader:
    block = test.blocks[bid.root]
    if not is_post_altair(block.spec):
        spec = test.phases[ALTAIR]
    else:
        spec = block.spec
    return ForkedLightClientHeader(spec=spec, data=spec.block_to_light_client_header(block.data))

def _sync_aggregate_for_block_id(test: LightClientDataCollectionTest, bid: BlockID) -> Optional[Any]:
    block = test.blocks[bid.root]
    if not is_post_altair(block.spec):
        return None
    return block.data.message.body.sync_aggregate

def _get_light_client_data(lc_data_store: LightClientDataStore, bid: BlockID) -> CachedLightClientData:
    try:
        return lc_data_store.cache.data[bid]
    except KeyError:
        raise ValueError('Trying to get light client data that was not cached')

def _cache_lc_data(lc_data_store: LightClientDataStore, spec: Any, state: Any, bid: BlockID, current_period_best_update: ForkedLightClientUpdate, latest_signature_slot: int) -> None:
    cached_data = CachedLightClientData(current_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.current_sync_committee_gindex_at_slot(state.slot)), latest_current_sync_committee_gindex(lc_data_store.spec)), next_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.next_sync_committee_gindex_at_slot(state.slot)), latest_next_sync_committee_gindex(lc_data_store.spec)), finalized_slot=spec.compute_start_slot_at_epoch(state.finalized_checkpoint.epoch), finality_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.finalized_root_gindex_at_slot(state.slot)), latest_finalized_root_gindex(lc_data_store.spec)), current_period_best_update=current_period_best_update, latest_signature_slot=latest_signature_slot)
    if bid in lc_data_store.cache.data:
        raise ValueError('Redundant `_cache_lc_data` call')
    lc_data_store.cache.data[bid] = cached_data

def _delete_light_client_data(lc_data_store: LightClientDataStore, bid: BlockID) -> None:
    del lc_data_store.cache.data[bid]

def _create_lc_finality_update_from_lc_data(test: LightClientDataCollectionTest, attested_bid: BlockID, signature_slot: int, sync_aggregate: Any) -> ForkedLightClientFinalityUpdate:
    attested_header = _light_client_header_for_block_id(test, attested_bid)
    attested_data = _get_light_client_data(test.lc_data_store, attested_bid)
    finalized_bid = _block_id_at_finalized_slot(test, attested_data.finalized_slot)
    if finalized_bid is not None:
        if finalized_bid.slot != attested_data.finalized_slot:
            attested_data.finalized_slot = finalized_bid.slot
        if finalized_bid.slot == attested_header.spec.GENESIS_SLOT:
            finalized_header = ForkedLightClientHeader(spec=attested_header.spec, data=attested_header.spec.LightClientHeader())
        else:
            finalized_header = _light_client_header_for_block_id(test, finalized_bid)
            finalized_header = ForkedLightClientHeader(spec=attested_header.spec, data=upgrade_lc_header_to_new_spec(finalized_header.spec, attested_header.spec, finalized_header.data))
        finality_branch = attested_data.finality_branch
    return ForkedLightClientFinalityUpdate(spec=attested_header.spec, data=attested_header.spec.LightClientFinalityUpdate(attested_header=attested_header.data, finalized_header=finalized_header.data, finality_branch=finality_branch, sync_aggregate=sync_aggregate, signature_slot=signature_slot))

def _create_lc_update_from_lc_data(test: LightClientDataCollectionTest, attested_bid: BlockID, signature_slot: int, sync_aggregate: Any, next_sync_committee: Any) -> ForkedLightClientUpdate:
    finality_update = _create_lc_finality_update_from_lc_data(test, attested_bid, signature_slot, sync_aggregate)
    attested_data = _get_light_client_data(test.lc_data_store, attested_bid)
    return ForkedLightClientUpdate(spec=finality_update.spec, data=finality_update.spec.LightClientUpdate(attested_header=finality_update.data.attested_header, next_sync_committee=next_sync_committee, next_sync_committee_branch=attested_data.next_sync_committee_branch, finalized_header=finality_update.data.finalized_header, finality_branch=finality_update.data.finality_branch, sync_aggregate=finality_update.data.sync_aggregate, signature_slot=finality_update.data.signature_slot))

def _create_lc_update(test: LightClientDataCollectionTest, spec: Any, state: Any, block: Any, parent_bid: BlockID) -> None:
    attested_bid = parent_bid
    attested_slot = attested_bid.slot
    if attested_slot < test.lc_data_store.cache.tail_slot:
        _cache_lc_data(test.lc_data_store, spec, state, _block_to_block_id(block), current_period_best_update=ForkedLightClientUpdate(spec=None, data=None), latest_signature_slot=spec.GENESIS_SLOT)
        return
    attested_period = spec.compute_sync_committee_period_at_slot(attested_slot)
    signature_slot = block.message.slot
    signature_period = spec.compute_sync_committee_period_at_slot(signature_slot)
    attested_data = _get_light_client_data(test.lc_data_store, attested_bid)
    if attested_period != signature_period:
        best = ForkedLightClientUpdate(spec=None, data=None)
    else:
        best = attested_data.current_period_best_update
    sync_aggregate = block.message.body.sync_aggregate
    num_active_participants = sum(sync_aggregate.sync_committee_bits)
    if num_active_participants < spec.MIN_SYNC_COMMITTEE_PARTICIPANTS:
        latest_signature_slot = attested_data.latest_signature_slot
    else:
        latest_signature_slot = signature_slot
    if num_active_participants < spec.MIN_SYNC_COMMITTEE_PARTICIPANTS or attested_period != signature_period:
        _cache_lc_data(test.lc_data_store, spec, state, _block_to_block_id(block), current_period_best_update=best, latest_signature_slot=latest_signature_slot)
        return
    update = _create_lc_update_from_lc_data(test, attested_bid, signature_slot, sync_aggregate, state.next_sync_committee)
    is_better = best.spec is None or spec.is_better_update(update.data, upgrade_lc_update_to_new_spec(best.spec, update.spec, best.data, test.phases))
    if is_better:
        best = update
    _cache_lc_data(test.lc_data_store, spec, state, _block_to_block_id(block), current_period_best_update=best, latest_signature_slot=latest_signature_slot)

def _create_lc_bootstrap(test: LightClientDataCollectionTest, spec: Any, bid: BlockID) -> None:
    block = test.blocks[bid.root]
    period = spec.compute_sync_committee_period_at_slot(bid.slot)
    if period not in test.lc_data_store.db.sync_committees:
        test.lc_data_store.db.sync_committees[period] = _get_current_sync_committee_for_finalized_period(test, period)
    test.lc_data_store.db.headers[bid.root] = ForkedLightClientHeader(spec=block.spec, data=block.spec.block_to_light_client_header(block.data))
    test.lc_data_store.db.current_branches[bid.slot] = _get_light_client_data(test.lc_data_store, bid).current_sync_committee_branch

def _process_new_block_for_light_client(test: LightClientDataCollectionTest, spec: Any, state: Any, block: Any, parent_bid: BlockID) -> None:
    if block.message.slot < test.lc_data_store.cache.tail_slot:
        return
    if is_post_altair(spec):
        _create_lc_update(test, spec, state, block, parent_bid)
    else:
        raise ValueError('`tail_slot` cannot be before Altair')

def _process_head_change_for_light_client(test: LightClientDataCollectionTest, spec: Any, head_bid: BlockID, old_finalized_bid: BlockID) -> None:
    if head_bid.slot < test.lc_data_store.cache.tail_slot:
        return
    head_period = spec.compute_sync_committee_period_at_slot(head_bid.slot)
    low_slot = max(test.lc_data_store.cache.tail_slot, old_finalized_bid.slot)
    low_period = spec.compute_sync_committee_period_at_slot(low_slot)
    bid = head_bid
    for period in reversed(range(low_period, head_period + 1)):
        period_end_slot = compute_start_slot_at_sync_committee_period(spec, period + 1) - 1
        bid = get_ancestor_of_block_id(test, bid, period_end_slot)
        if bid is None or bid.slot < low_slot:
            break
        best = _get_light_client_data(test.lc_data_store, bid).current_period_best_update
        if best.spec is None or sum(best.data.sync_aggregate.sync_committee_bits) < spec.MIN_SYNC_COMMITTEE_PARTICIPANTS:
            test.lc_data_store.db.best_updates.pop(period, None)
        else:
            test.lc_data_store.db.best_updates[period] = best
    head_data = _get_light_client_data(test.lc_data_store, head_bid)
    signature_slot = head_data.latest_signature_slot
    if signature_slot <= low_slot:
        test.lc_data_store.cache.latest = ForkedLightClientFinalityUpdate(spec=None, data=None)
        return
    signature_bid = get_ancestor_of_block_id(test, head_bid, signature_slot)
    if signature_bid is None or signature_bid.slot <= low_slot:
        test.lc_data_store.cache.latest = ForkedLightClientFinalityUpdate(spec=None, data=None)
        return
    attested_bid = get_ancestor_of_block_id(test, signature_bid, signature_bid.slot - 1)
    if attested_bid is None or attested_bid.slot < low_slot:
        test.lc_data_store.cache.latest = ForkedLightClientFinalityUpdate(spec=None, data=None)
        return
    sync_aggregate = _sync_aggregate_for_block_id(test, signature_bid)
    assert sync_aggregate is not None
    test.lc_data_store.cache.latest = _create_lc_finality_update_from_lc_data(test, attested_bid, signature_slot, sync_aggregate)

def _process_finalization_for_light_client(test: LightClientDataCollectionTest, spec: Any, finalized_bid: BlockID, old_finalized_bid: BlockID) -> None:
    finalized_slot = finalized_bid.slot
    if finalized_slot < test.lc_data_store.cache.tail_slot:
        return
    first_new_slot = old_finalized_bid.slot + 1
    low_slot = max(first_new_slot, test.lc_data_store.cache.tail_slot)
    boundary_slot = finalized_slot
    while boundary_slot >= low_slot:
        bid = _block_id_at_finalized_slot(test, boundary_slot)
        if bid is None:
            break
        if bid.slot >= low_slot:
            _create_lc_bootstrap(test, spec, bid)
        boundary_slot = _next_epoch_boundary_slot(spec, bid.slot)
        if boundary_slot < spec.SLOTS_PER_EPOCH:
            break
        boundary_slot = boundary_slot - spec.SLOTS_PER_EPOCH
    bids_to_delete = []
    for bid in test.lc_data_store.cache.data:
        if bid.slot >= finalized_bid.slot:
            continue
        bids_to_delete.append(bid)
    for bid in bids_to_delete:
        _delete_light_client_data(test.lc_data_store, bid)

def get_light_client_bootstrap(test: LightClientDataCollectionTest, block_root: bytes) -> ForkedLightClientBootstrap:
    try:
        header = test.lc_data_store.db.headers[block_root]
    except KeyError:
        return ForkedLightClientBootstrap(spec=None, data=None)
    slot = header.data.beacon.slot
    period = header.spec.compute_sync_committee_period_at_slot(slot)
    return ForkedLightClientBootstrap(spec=header.spec, data=header.spec.LightClientBootstrap(header=header.data, current_sync_committee=test.lc_data_store.db.sync_committees[period], current_sync_committee_branch=test.lc_data_store.db.current_branches[slot]))

def get_light_client_update_for_period(test: LightClientDataCollectionTest, period: int) -> ForkedLightClientUpdate:
    try:
        return test.lc_data_store.db.best_updates[period]
    except KeyError:
        return ForkedLightClientUpdate(spec=None, data=None)

def get_light_client_finality_update(test: LightClientDataCollectionTest) -> ForkedLightClientFinalityUpdate:
    return test.lc_data_store.cache.latest

def get_light_client_optimistic_update(test: LightClientDataCollectionTest) -> ForkedLightClientOptimisticUpdate:
    finality_update = get_light_client_finality_update(test)
    if finality_update.spec is None:
        return ForkedLightClientOptimisticUpdate(spec=None, data=None)
    return ForkedLightClientOptimisticUpdate(spec=finality_update.spec, data=finality_update.spec.LightClientOptimisticUpdate(attested_header=finality_update.data.attested_header, sync_aggregate=finality_update.data.sync_aggregate, signature_slot=finality_update.data.signature_slot))

def setup_lc_data_collection_test(spec: Any, state: Any, phases: Optional[Dict[str, Any]] = None) -> LightClientDataCollectionTest:
    assert spec.compute_slots_since_epoch_start(state.slot) == 0
    test = LightClientDataCollectionTest(steps=[], files=set(), phases=phases, blocks={}, finalized_block_roots={}, states={}, finalized_checkpoint_states={}, latest_finalized_epoch=state.finalized_checkpoint.epoch, latest_finalized_bid=BlockID(slot=spec.compute_start_slot_at_epoch(state.finalized_checkpoint.epoch), root=state.finalized_checkpoint.root), historical_tail_slot=state.slot, lc_data_store=LightClientDataStore(spec=spec, cache=LightClientDataCache(data={}, latest=ForkedLightClientFinalityUpdate(spec=None, data=None), tail_slot=max(state.slot, spec.compute_start_slot_at_epoch(spec.config.ALTAIR_FORK_EPOCH))), db=LightClientDataDB(headers={}, current_branches={}, sync_committees={}, best_updates={})))
    bid = _state_to_block_id(state)
    yield ('initial_state', state)
    test.blocks[bid.root] = ForkedSignedBeaconBlock(spec=spec, data=spec.SignedBeaconBlock(message=spec.BeaconBlock(state_root=state.hash_tree_root())))
    test.finalized_block_roots[bid.slot] = bid.root
    test.states[state.hash_tree_root()] = ForkedBeaconState(spec=spec, data=state)
    test.finalized_checkpoint_states[state.hash_tree_root()] = ForkedBeaconState(spec=spec, data=state)
    _cache_lc_data(test.lc_data_store, spec, state, bid, current_period_best_update=ForkedLightClientUpdate(spec=None, data=None), latest_signature_slot=spec.GENESIS_SLOT)
    _create_lc_bootstrap(test, spec, bid)
    return test

def finish_lc_data_collection_test(test: LightClientDataCollectionTest) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    yield ('steps', test.steps)

def _encode_lc_object(test: LightClientDataCollectionTest, prefix: str, obj: Union[ForkedSignedBeaconBlock, ForkedLightClientBootstrap, ForkedLightClientUpdate, ForkedLightClientFinalityUpdate, ForkedLightClientOptimisticUpdate], slot: int, genesis_validators_root: bytes) -> Iterator[Tuple[str, Any]]:
    yield from []
    file_name = f'{prefix}_{slot}_{encode_hex(obj.data.hash_tree_root())}'
    if file_name not in test.files:
        test.files.add(file_name)
        yield (file_name, obj.data)
    return {'fork_digest': encode_hex(obj.spec.compute_fork_digest(obj.spec.compute_fork_version(obj.spec.compute_epoch_at_slot(slot)), genesis_validators_root)), 'data': file_name}

def add_new_block(test: LightClientDataCollectionTest, spec: Any, state: Any, slot: Optional[int] = None, num_sync_participants: int = 0) -> Iterator[Union[Tuple[str, Any], Tuple[Any, Any, BlockID]]]:
    if slot is None:
        slot = state.slot + 1
    assert slot > state.slot
    parent_bid = _state_to_block_id(state)
    if state.slot < slot - 1:
        spec, state, _ = transition_across_forks(spec, state, slot - 1, phases=test.phases)
    sync_aggregate, signature_slot = get_sync_aggregate(spec, state, num_participants=num_sync_participants, phases=test.phases)
    assert signature_slot == slot
    spec, state, block = transition_across_forks(spec, state, slot, phases=test.phases, with_block=True, sync_aggregate=sync_aggregate)
    bid = _block_to_block_id(block)
    test.blocks[bid.root] = ForkedSignedBeaconBlock(spec=spec, data=block)
    test.states[block.message.state_root] = ForkedBeaconState(spec=spec, data=state)
    _process_new_block_for_light_client(test, spec, state, block, parent_bid)
    block_obj = (yield from _encode_lc_object(test, 'block', ForkedSignedBeaconBlock(spec=spec, data=block), block.message.slot, state.genesis_validators_root))
    test.steps.append({'new_block': block_obj})
    return (spec, state, bid)

def select_new_head(test: LightClientDataCollectionTest, spec: Any, head_bid: BlockID) -> Iterator[Tuple[str, Any]]:
    old_finalized_bid = test.latest_finalized_bid
    _process_head_change_for_light_client(test, spec, head_bid, old_finalized_bid)
    block = test.blocks[head_bid.root]
    state = test.states[block.data.message.state_root]
    if state.data.finalized_checkpoint.epoch != spec.GENESIS_EPOCH:
        block = test.blocks[state.data.finalized_checkpoint.root]
        bid = _block_to_block_id(block.data)
        new_finalized_bid = bid
        if new_finalized_bid.slot > old_finalized_bid.slot:
            old_finalized_epoch = None
            new_finalized_epoch = state.data.finalized_checkpoint.epoch
            while bid.slot > test.latest_finalized_bid.slot:
                test.finalized_block_roots[bid.slot] = bid.root
                finalized_epoch = spec.compute_epoch_at_slot(bid.slot + spec.SLOTS_PER_EPOCH - 1)
                if finalized_epoch != old_finalized_epoch:
                    state = test.states[block.data.message.state_root]
                    test.finalized_checkpoint_states[block.data.message.state_root] = state
                    old_finalized_epoch = finalized_epoch
                block = test.blocks[block.data.message.parent_root]
                bid = _block_to_block_id(block.data)
            test.latest_finalized_epoch = new_finalized_epoch
            test.latest_finalized_bid = new_finalized_bid
            _process_finalization_for_light_client(test, spec, new_finalized_bid, old_finalized_bid)
            blocks_to_delete = []
            for block_root, block in test.blocks.items():
                if block.data.message.slot < new_finalized_bid.slot:
                    blocks_to_delete.append(block_root)
            for block_root in blocks_to_delete:
                del test.blocks[block_root]
            states_to_delete = []
            for state_root, state in test.states.items():
                if state.data.slot < new_finalized_bid.slot:
                    states_to_delete.append(state_root)
            for state_root in states_to_delete:
                del test.states[state_root]
    yield from []
    bootstraps = []
    for state in test.finalized_checkpoint_states.values():
        bid = _state_to_block_id(state.data)
        entry = {'block_root': encode_hex(bid.root)}
        bootstrap = get_light_client_bootstrap(test, bid.root)
        if bootstrap.spec is not None:
            bootstrap_obj = (yield from _encode_lc_object(test, 'bootstrap', bootstrap, bootstrap.data.header.beacon.slot, state.data.genesis_validators_root))
            entry['bootstrap'] = bootstrap_obj
        bootstraps.append(entry)
    best_updates = []
    low_period = spec.compute_sync_committee_period_at_slot(test.lc_data_store.cache.tail_slot)
    head_period = spec.compute_sync_committee_period_at_slot(head_bid.slot)
    for period in range(low_period, head_period + 1):
        entry = {'period': int(period)}
        update = get_light_client_update_for_period(test, period)
        if update.spec is not None:
            update_obj = (yield from _encode_lc_object(test, 'update', update, update.data.attested_header.beacon.slot, state.data.genesis_validators_root))
            entry['update'] = update_obj
        best_updates.append(entry)
    checks = {'latest_finalized_checkpoint': {'epoch': int(test.latest_finalized_epoch), 'root': encode_hex(test.latest_finalized_bid.root)}, 'bootstraps': bootstraps, 'best_updates': best_updates}
    finality_update = get_light_client_finality_update(test)
    if finality_update.spec is not None:
        finality_update_obj = (yield from _encode_lc_object(test, 'finality_update', finality_update, finality_update.data.attested_header.beacon.slot, state.data.genesis_validators_root))
        checks['latest_finality_update'] = finality_update_obj
    optimistic_update = get_light_client_optimistic_update(test)
    if optimistic_update.spec is not None:
        optimistic_update_obj = (yield from _encode_lc_object(test, 'optimistic_update', optimistic_update, optimistic_update.data.attested_header.beacon.slot, state.data.genesis_validators_root))
        checks['latest_optimistic_update'] = optimistic_update_obj
    test.steps.append({'new_head': {'head_block_root': encode_hex(head_bid.root), 'checks': checks}})

def run_lc_data_collection_test_multi_fork(spec: Any, phases: Dict[str, Any], state: Any, fork_1: str, fork_2: str) -> Iterator[Union[Tuple[str, Any], Tuple[str, List[Dict[str, Any]]]]]:
    test = (yield from setup_lc_data_collection_test(spec, state, phases=phases))
    genesis_bid = BlockID(slot=state.slot, root=spec.BeaconBlock(state_root=state.hash_tree_root()).hash_tree_root())
    assert get_lc_bootstrap_block_id(get_light_client_bootstrap(test, genesis_bid.root).data) == genesis_bid
    fork_1_epoch = getattr(phases[fork_1].config, fork_1.upper() + '_FORK_EPOCH')
    fork_1_period = spec.compute_sync_committee_period(fork_1_epoch)
    slot = compute_start_slot_at_sync_committee_period(spec, fork_1_period) - spec.SLOTS_PER_EPOCH
    spec, state, bid = (yield from add_new_block(test, spec, state, slot=slot, num_sync_participants=1))
    yield from select_new_head(test, spec, bid)
    assert get_light_client_bootstrap(test, bid.root).spec is None
    slot_period = spec.compute_sync_committee_period_at_slot(slot)
    if slot_period == 0:
        assert get_lc_update_attested_block_id(get_light_client_update_for_period(test, 0).data) == genesis_bid
    else:
        for period in range(0, slot_period):
            assert get_light_client_update_for_period(test, period).spec is None
    state_period = spec.compute_sync_committee_period_at_slot(state.slot)
    spec_a = spec
    state_a = state
    slot_a = state_a.slot
    bids_a: List[BlockID] = [bid]
    num_sync_participants_a = 1
    fork_2_epoch = getattr(phases[fork_2].config, fork_2.upper() + '_FORK_EPOCH')
    while spec_a.get_current_epoch(state_a) <= fork_2_epoch:
        attested_period = spec_a.compute_sync_committee_period_at_slot(slot_a)
        slot_a += 4
        signature_period = spec_a.compute_sync_committee_period_at_slot(slot_a)
        if signature_period != attested_period:
            num_sync_participants_a = 0
        num_sync_participants_a += 1
        spec_a, state_a, bid_a = (yield from add_new_block(test, spec_a, state_a, slot=slot_a, num_sync_participants=num_sync_participants_a))
        yield from select_new_head(test, spec_a, bid_a)
        for bid in bids_a:
            assert get_light_client_bootstrap(test, bid.root).spec is None
        if attested_period == signature_period:
            assert get_lc_update_attested_block_id(get_light_client_update_for_period(test, attested_period).data) == bids_a[-1]
        else:
            assert signature_period == attested_period + 1
            assert get_lc_update_attested_block_id(get_light_client_update_for_period(test, attested_period).data) == bids_a[-2]
            assert get_light_client_update_for_period(test, signature_period).spec is None
        assert get_lc_update_attested_block_id(get_light_client_finality_update(test).data) == bids_a[-1]
        assert get_lc_update_attested_block_id(get_light_client_optimistic_update(test).data) == bids_a[-1]
        bids_a.append(bid_a)
    spec_b = spec
    state_b = state
    slot_b = state_b.slot
    bids_b: List[BlockID] = [bid]
    while spec_b.get_current_epoch(state_b) <= fork_2_epoch:
        slot_b += 4
        signature_period = spec_b.compute_sync_committee_period_at_slot(slot_b)
        spec_b, state_b, bid_b = (yield from add_new_block(test, spec_b, state_b, slot=slot_b))
        for bid in bids_b:
            assert get_light_client_bootstrap(test, bid.root).spec is None
        bids_b.append(bid_b)
    attested_period = spec_b.compute_sync_committee_period_at_slot(slot_b)
    slot_b += 1
    signature_period = spec_b.compute_sync_committee_period_at_slot(slot_b)
    num_sync_participants_b = 1
    spec_b, state_b, bid_b = (yield from add_new_block(test, spec_b, state_b, slot=slot_b, num_sync_participants=num_sync_participants_b))
    yield from select_new_head(test, spec_b, bid_b)
    for bid in bids_b:
        assert get_light_client_bootstrap(test, bid.root).spec is None
    if attested_period == signature_period:
        assert get_lc_update_attested_block_id(get_light_client_update_for_period(test, attested_period).data) == bids_b[-1]
    else:
        assert signature_period == attested_period + 1
        assert get_lc_update_attested_block_id(get_light_client_update_for_period(test, attested_period).data) == bids_b[-2]
        assert get_light_client_update_for_period(test, signature_period).spec is None
    assert get_lc_update_attested_block_id(get_light_client_finality_update(test).data) == bids_b[-1]
    assert get_lc_update_attested_block_id(get_light_client_optimistic_update(test).data) == bids_b[-1]
    bids_b.append(bid_b)
    state_b_period = spec_b.compute_sync_committee_period_at_slot(state_b.slot)
    for period in range(state_period + 1, state_b_period):
        assert get_light_client_update_for_period(test, period).spec is None
    attested_period = spec_a.compute_sync_committee_period_at_slot(slot_a)
    slot_a = slot_b + 1
    signature_period = spec_a.compute_sync_committee_period_at_slot(slot_a)
    if signature_period != attested_period:
        num_sync_participants_a = 0
    num_sync_participants_a += 1
    spec_a, state_a, bid_a = (yield from add_new_block(test, spec_a, state_a, slot=slot_a, num_sync_participants=num_sync_participants_a))
    yield from select_new_head(test, spec_a, bid_a)
    for bid in bids_a:
        assert get_light_client_bootstrap(test, bid.root).spec is None
    if attested_period == signature_period:
        assert get_lc_update_attested_block_id(get_light_client_update_for_period(test, attested_period).data) == bids_a[-1]
    else:
        assert signature_period == attested_period + 1
        assert get_lc_update_attested_block_id(get_light_client_update_for_period(test, attested_period).data) == bids_a[-2]
        assert get_light_client_update_for_period(test, signature_period).spec is None
    assert get_lc_update_attested_block_id(get_light_client_finality_update(test).data) == bids_a[-1]
    assert get_lc_update_attested_block_id(get_light_client_optimistic_update(test).data) == bids_a[-1]
    bids_a.append(bid_a)
    state_a_period = spec_a.compute_sync_committee_period_at_slot(state_a.slot)
    for period in range(state_period + 1, state_a_period):
        assert get_light_client_update_for_period(test, period).spec is not None
    yield from finish_lc_data_collection_test(test)
