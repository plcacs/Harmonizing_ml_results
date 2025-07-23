from random import Random
from typing import Generator, Tuple, List, Optional
from eth2spec.test.helpers.keys import privkeys, pubkeys
from eth2spec.test.helpers.state import state_transition_and_sign_block
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.sync_committee import compute_committee_indices, compute_aggregate_sync_committee_signature
from eth2spec.test.helpers.proposer_slashings import get_valid_proposer_slashing
from eth2spec.test.helpers.attester_slashings import get_valid_attester_slashing_by_indices
from eth2spec.test.helpers.attestations import get_valid_attestation, get_max_attestations
from eth2spec.test.helpers.deposits import build_deposit, deposit_from_context
from eth2spec.test.helpers.voluntary_exits import prepare_signed_exits
from eth2spec.test.helpers.bls_to_execution_changes import get_signed_address_change

def run_slash_and_exit(
    spec: 'Spec', 
    state: 'BeaconState', 
    slash_index: int, 
    exit_index: int, 
    valid: bool = True
) -> Generator[Tuple[str, Optional['BeaconState'] or List['SignedBlock']], None, None]:
    """
    Helper function to run a test that slashes and exits two validators
    """
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    yield ('pre', state)
    block = build_empty_block_for_next_slot(spec, state)
    proposer_slashing = get_valid_proposer_slashing(
        spec, state, slashed_index=slash_index, signed_1=True, signed_2=True
    )
    signed_exit = prepare_signed_exits(spec, state, [exit_index])[0]
    block.body.proposer_slashings.append(proposer_slashing)
    block.body.voluntary_exits.append(signed_exit)
    signed_block = state_transition_and_sign_block(spec, state, block, expect_fail=not valid)
    yield ('blocks', [signed_block])
    if not valid:
        yield ('post', None)
        return
    yield ('post', state)

def get_random_proposer_slashings(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random
) -> List['ProposerSlashing']:
    num_slashings: int = rng.randrange(1, spec.MAX_PROPOSER_SLASHINGS)
    active_indices: List[int] = spec.get_active_validator_indices(state, spec.get_current_epoch(state)).copy()
    indices: List[int] = [index for index in active_indices if not state.validators[index].slashed]
    slashings: List['ProposerSlashing'] = [
        get_valid_proposer_slashing(
            spec, state, 
            slashed_index=indices.pop(rng.randrange(len(indices))), 
            signed_1=True, 
            signed_2=True
        ) for _ in range(num_slashings)
    ]
    return slashings

def get_random_attester_slashings(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random, 
    slashed_indices: Optional[List[int]] = []
) -> List['AttesterSlashing']:
    """
    Caller can supply ``slashed_indices`` if they are aware of other indices
    that will be slashed by other operations in the same block as the one that
    contains the output of this function.
    """
    num_slashings: int = rng.randrange(1, spec.MAX_ATTESTER_SLASHINGS)
    active_indices: List[int] = spec.get_active_validator_indices(state, spec.get_current_epoch(state)).copy()
    indices: List[int] = [
        index for index in active_indices 
        if not state.validators[index].slashed and index not in slashed_indices
    ]
    sample_upper_bound: int = 4
    max_slashed_count: int = num_slashings * sample_upper_bound - 1
    if len(indices) < max_slashed_count:
        return []
    slot_range: List[int] = list(range(state.slot - spec.SLOTS_PER_HISTORICAL_ROOT + 1, state.slot))
    slashings: List['AttesterSlashing'] = [
        get_valid_attester_slashing_by_indices(
            spec, state, 
            sorted([
                indices.pop(rng.randrange(len(indices))) 
                for _ in range(rng.randrange(1, sample_upper_bound))
            ]), 
            slot=slot_range.pop(rng.randrange(len(slot_range))), 
            signed_1=True, 
            signed_2=True
        ) for _ in range(num_slashings)
    ]
    return slashings

def get_random_attestations(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random
) -> List['Attestation']:
    num_attestations: int = rng.randrange(1, get_max_attestations(spec))
    attestations: List['Attestation'] = [
        get_valid_attestation(
            spec, state, 
            slot=rng.randrange(state.slot - spec.SLOTS_PER_EPOCH + 1, state.slot), 
            signed=True
        ) for _ in range(num_attestations)
    ]
    return attestations

def get_random_deposits(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random, 
    num_deposits: Optional[int] = None
) -> Tuple[List['Deposit'], bytes]:
    if not num_deposits:
        num_deposits = rng.randrange(1, spec.MAX_DEPOSITS)
    if num_deposits == 0:
        return ([], b'\x00' * 32)
    deposit_data_leaves: List['DepositData'] = [spec.DepositData() for _ in range(len(state.validators))]
    deposits: List['Deposit'] = []
    for i in range(num_deposits):
        index: int = len(state.validators) + i
        withdrawal_pubkey: bytes = pubkeys[-1 - index]
        withdrawal_credentials: bytes = spec.BLS_WITHDRAWAL_PREFIX + spec.hash(withdrawal_pubkey)[1:]
        _, root, deposit_data_leaves = build_deposit(
            spec, 
            deposit_data_leaves, 
            pubkeys[index], 
            privkeys[index], 
            spec.MAX_EFFECTIVE_BALANCE, 
            withdrawal_credentials=withdrawal_credentials, 
            signed=True
        )
    for i in range(num_deposits):
        index: int = len(state.validators) + i
        deposit, _, _ = deposit_from_context(spec, deposit_data_leaves, index)
        deposits.append(deposit)
    return (deposits, root)

def prepare_state_and_get_random_deposits(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random, 
    num_deposits: Optional[int] = None
) -> List['Deposit']:
    deposits, root = get_random_deposits(spec, state, rng, num_deposits=num_deposits)
    state.eth1_data.deposit_root = root
    state.eth1_data.deposit_count += len(deposits)
    return deposits

def _eligible_for_exit(
    spec: 'Spec', 
    state: 'BeaconState', 
    index: int
) -> bool:
    validator = state.validators[index]
    not_slashed: bool = not validator.slashed
    current_epoch: int = spec.get_current_epoch(state)
    activation_epoch: int = validator.activation_epoch
    active_for_long_enough: bool = current_epoch >= activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD
    not_exited: bool = validator.exit_epoch == spec.FAR_FUTURE_EPOCH
    return not_slashed and active_for_long_enough and not_exited

def get_random_voluntary_exits(
    spec: 'Spec', 
    state: 'BeaconState', 
    to_be_slashed_indices: List[int], 
    rng: Random
) -> List['VoluntaryExit']:
    num_exits: int = rng.randrange(1, spec.MAX_VOLUNTARY_EXITS)
    active_indices: set = set(spec.get_active_validator_indices(state, spec.get_current_epoch(state)).copy())
    indices: set = set(
        index for index in active_indices 
        if _eligible_for_exit(spec, state, index)
    )
    eligible_indices: set = indices - set(to_be_slashed_indices)
    indices_count: int = min(num_exits, len(eligible_indices))
    exit_indices: List[int] = [eligible_indices.pop() for _ in range(indices_count)]
    return prepare_signed_exits(spec, state, exit_indices)

def get_random_sync_aggregate(
    spec: 'Spec', 
    state: 'BeaconState', 
    slot: int, 
    block_root: Optional[bytes] = None, 
    fraction_participated: float = 1.0, 
    rng: Optional[Random] = Random(2099)
) -> 'SyncAggregate':
    committee_indices: List[int] = compute_committee_indices(state, state.current_sync_committee)
    participant_count: int = int(len(committee_indices) * fraction_participated)
    participant_indices: List[int] = rng.sample(range(len(committee_indices)), participant_count)
    participants: List[int] = [committee_indices[index] for index in participant_indices]
    signature: bytes = compute_aggregate_sync_committee_signature(
        spec, state, slot, participants, block_root=block_root
    )
    return spec.SyncAggregate(
        sync_committee_bits=[index in participant_indices for index in range(len(committee_indices))], 
        sync_committee_signature=signature
    )

def get_random_bls_to_execution_changes(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random = Random(2188), 
    num_address_changes: int = 0
) -> List['SignedAddressChange']:
    bls_indices: List[int] = [
        index for index, validator in enumerate(state.validators) 
        if validator.withdrawal_credentials[:1] == spec.BLS_WITHDRAWAL_PREFIX
    ]
    assert len(bls_indices) > 0
    return [
        get_signed_address_change(spec, state, validator_index=validator_index) 
        for validator_index in rng.sample(bls_indices, min(num_address_changes, len(bls_indices)))
    ]

def build_random_block_from_state_for_next_slot(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random = Random(2188), 
    deposits: Optional[List['Deposit']] = None
) -> 'Block':
    block: 'Block' = build_empty_block_for_next_slot(spec, state)
    proposer_slashings: List['ProposerSlashing'] = get_random_proposer_slashings(spec, state, rng)
    block.body.proposer_slashings = proposer_slashings
    slashed_indices: List[int] = [
        slashing.signed_header_1.message.proposer_index for slashing in proposer_slashings
    ]
    block.body.attester_slashings = get_random_attester_slashings(spec, state, rng, slashed_indices)
    block.body.attestations = get_random_attestations(spec, state, rng)
    if deposits:
        block.body.deposits = deposits
    slashed_indices_set: set = set(
        slashing.signed_header_1.message.proposer_index for slashing in block.body.proposer_slashings
    )
    for attester_slashing in block.body.attester_slashings:
        slashed_indices_set = slashed_indices_set.union(attester_slashing.attestation_1.attesting_indices)
        slashed_indices_set = slashed_indices_set.union(attester_slashing.attestation_2.attesting_indices)
    block.body.voluntary_exits = get_random_voluntary_exits(spec, state, list(slashed_indices_set), rng)
    return block

def run_test_full_random_operations(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random = Random(2080)
) -> Generator[Tuple[str, Optional['BeaconState'] or List['SignedBlock']], None, None]:
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    deposits: List['Deposit'] = prepare_state_and_get_random_deposits(spec, state, rng)
    block: 'Block' = build_random_block_from_state_for_next_slot(spec, state, rng, deposits=deposits)
    yield ('pre', state)
    signed_block: 'SignedBlock' = state_transition_and_sign_block(spec, state, block)
    yield ('blocks', [signed_block])
    yield ('post', state)

def get_random_execution_requests(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random
) -> 'ExecutionRequests':
    deposits: List['DepositRequest'] = get_random_deposit_requests(spec, state, rng)
    withdrawals: List['WithdrawalRequest'] = get_random_withdrawal_requests(spec, state, rng)
    consolidations: List['ConsolidationRequest'] = get_random_consolidation_requests(spec, state, rng)
    execution_requests: 'ExecutionRequests' = spec.ExecutionRequests(
        deposits=deposits, 
        withdrawals=withdrawals, 
        consolidations=consolidations
    )
    return execution_requests

def get_random_deposit_requests(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random, 
    num_deposits: Optional[int] = None
) -> List['DepositRequest']:
    if num_deposits is None:
        num_deposits = rng.randint(0, spec.MAX_DEPOSIT_REQUESTS_PER_PAYLOAD)
    deposit_data_leaves: List['DepositData'] = [spec.DepositData() for _ in range(len(state.validators))]
    deposit_requests: List['DepositRequest'] = []
    for _ in range(num_deposits):
        index: int = rng.randrange(0, num_deposits)
        withdrawal_pubkey: bytes = pubkeys[index]
        withdrawal_credentials: bytes = spec.BLS_WITHDRAWAL_PREFIX + spec.hash(withdrawal_pubkey)[1:]
        deposit, _, _ = build_deposit(
            spec, 
            deposit_data_leaves, 
            pubkeys[index], 
            privkeys[index], 
            rng.randint(spec.EFFECTIVE_BALANCE_INCREMENT, 2 * spec.MAX_EFFECTIVE_BALANCE_ELECTRA), 
            withdrawal_credentials=withdrawal_credentials, 
            signed=True
        )
        deposit_requests.append(
            spec.DepositRequest(
                pubkey=deposit.data.pubkey, 
                withdrawal_credentials=deposit.data.withdrawal_credentials, 
                amount=deposit.data.amount, 
                signature=deposit.data.signature, 
                index=rng.randrange(0, 2 ** 64)
            )
        )
    return deposit_requests

def get_random_withdrawal_requests(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random, 
    num_withdrawals: Optional[int] = None
) -> List['WithdrawalRequest']:
    if num_withdrawals is None:
        num_withdrawals = rng.randint(0, spec.MAX_WITHDRAWAL_REQUESTS_PER_PAYLOAD)
    current_epoch: int = spec.get_current_epoch(state)
    active_validator_indices: List[int] = spec.get_active_validator_indices(state, current_epoch)
    withdrawal_requests: List['WithdrawalRequest'] = []
    for _ in range(num_withdrawals):
        if not active_validator_indices:
            break
        address: bytes = rng.getrandbits(160).to_bytes(20, 'big')
        validator_index: int = rng.choice(active_validator_indices)
        validator = state.validators[validator_index]
        validator_balance: int = state.balances[validator_index]
        withdrawal_requests.append(
            spec.WithdrawalRequest(
                source_address=address, 
                validator_pubkey=validator.pubkey, 
                amount=rng.randint(0, validator_balance)
            )
        )
    return withdrawal_requests

def get_random_consolidation_requests(
    spec: 'Spec', 
    state: 'BeaconState', 
    rng: Random, 
    num_consolidations: Optional[int] = None
) -> List['ConsolidationRequest']:
    if num_consolidations is None:
        num_consolidations = rng.randint(0, spec.MAX_CONSOLIDATION_REQUESTS_PER_PAYLOAD)
    current_epoch: int = spec.get_current_epoch(state)
    active_validator_indices: List[int] = spec.get_active_validator_indices(state, current_epoch)
    consolidation_requests: List['ConsolidationRequest'] = []
    for _ in range(num_consolidations):
        source_address: bytes = rng.getrandbits(160).to_bytes(20, 'big')
        source_index: int = rng.choice(active_validator_indices)
        target_index: int = rng.choice(active_validator_indices)
        source_validator = state.validators[source_index]
        target_validator = state.validators[target_index]
        consolidation_requests.append(
            spec.ConsolidationRequest(
                source_address=source_address, 
                source_pubkey=source_validator.pubkey, 
                target_pubkey=target_validator.pubkey
            )
        )
    return consolidation_requests
