from random import Random
from eth2spec.test.context import spec_state_test, with_all_phases
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with
from eth2spec.test.helpers.forks import is_post_altair
from eth2spec.test.helpers.state import transition_to, next_epoch_via_block, next_slot
from eth2spec.test.helpers.voluntary_exits import get_unslashed_exited_validators
from eth2spec import spec

def run_process_just_and_fin(spec: spec.Spec, state: spec.BeaconState) -> None:
    yield from run_epoch_processing_with(spec, state, 'process_justification_and_finalization')

def add_mock_attestations(spec: spec.Spec, state: spec.BeaconState, epoch: int, source: spec.Checkpoint, target: spec.Checkpoint, sufficient_support: bool = False, messed_up_target: bool = False) -> None:
    assert (state.slot + 1) % spec.SLOTS_PER_EPOCH == 0
    previous_epoch = spec.get_previous_epoch(state)
    current_epoch = spec.get_current_epoch(state)
    if not is_post_altair(spec):
        if current_epoch == epoch:
            attestations = state.current_epoch_attestations
        elif previous_epoch == epoch:
            attestations = state.previous_epoch_attestations
        else:
            raise Exception(f'cannot include attestations in epoch ${epoch} from epoch ${current_epoch}')
    elif current_epoch == epoch:
        epoch_participation = state.current_epoch_participation
    elif previous_epoch == epoch:
        epoch_participation = state.previous_epoch_participation
    else:
        raise Exception(f'cannot include attestations in epoch ${epoch} from epoch ${current_epoch}')
    total_balance = spec.get_total_active_balance(state)
    remaining_balance = int(total_balance * 2 // 3)
    start_slot = spec.compute_start_slot_at_epoch(epoch)
    committees_per_slot = spec.get_committee_count_per_slot(state, epoch)
    for slot in range(start_slot, start_slot + spec.SLOTS_PER_EPOCH):
        for index in range(committees_per_slot):
            if remaining_balance < 0:
                return
            committee = spec.get_beacon_committee(state, slot, index)
            aggregation_bits = [0] * len(committee)
            for v in range(len(committee) * 2 // 3 + 1):
                if remaining_balance > 0:
                    remaining_balance -= int(state.validators[v].effective_balance)
                    aggregation_bits[v] = 1
                else:
                    break
            if not sufficient_support:
                for i in range(max(len(committee) // 5, 1)):
                    aggregation_bits[i] = 0
            if not is_post_altair(spec):
                attestations.append(spec.PendingAttestation(aggregation_bits=aggregation_bits, data=spec.AttestationData(slot=slot, beacon_block_root=b'\xff' * 32, source=source, target=target, index=index), inclusion_delay=1))
                if messed_up_target:
                    attestations[len(attestations) - 1].data.target.root = b'\x99' * 32
            else:
                for i, index in enumerate(committee):
                    if aggregation_bits[i]:
                        epoch_participation[index] |= spec.ParticipationFlags(2 ** spec.TIMELY_HEAD_FLAG_INDEX)
                        epoch_participation[index] |= spec.ParticipationFlags(2 ** spec.TIMELY_SOURCE_FLAG_INDEX)
                        if not messed_up_target:
                            epoch_participation[index] |= spec.ParticipationFlags(2 ** spec.TIMELY_TARGET_FLAG_INDEX)

def get_checkpoints(spec: spec.Spec, epoch: int) -> tuple:
    c1 = None if epoch < 1 else spec.Checkpoint(epoch=epoch - 1, root=b'\xaa' * 32)
    c2 = None if epoch < 2 else spec.Checkpoint(epoch=epoch - 2, root=b'\bb' * 32)
    c3 = None if epoch < 3 else spec.Checkpoint(epoch=epoch - 3, root=b'\cc' * 32)
    c4 = None if epoch < 4 else spec.Checkpoint(epoch=epoch - 4, root=b'\dd' * 32)
    c5 = None if epoch < 5 else spec.Checkpoint(epoch=epoch - 5, root=b'\ee' * 32)
    return (c1, c2, c3, c4, c5)

def put_checkpoints_in_block_roots(spec: spec.Spec, state: spec.BeaconState, checkpoints: list) -> None:
    for c in checkpoints:
        state.block_roots[spec.compute_start_slot_at_epoch(c.epoch) % spec.SLOTS_PER_HISTORICAL_ROOT] = c.root

def finalize_on_234(spec: spec.Spec, state: spec.BeaconState, epoch: int, sufficient_support: bool) -> None:
    assert epoch > 4
    transition_to(spec, state, spec.SLOTS_PER_EPOCH * epoch - 1)
    c1, c2, c3, c4, _ = get_checkpoints(spec, epoch)
    put_checkpoints_in_block_roots(spec, state, [c1, c2, c3, c4])
    old_finalized = state.finalized_checkpoint
    state.previous_justified_checkpoint = c4
    state.current_justified_checkpoint = c3
    state.justification_bits = spec.Bitvector[spec.JUSTIFICATION_BITS_LENGTH]()
    state.justification_bits[1:3] = [1, 1]
    add_mock_attestations(spec, state, epoch=epoch - 2, source=c4, target=c2, sufficient_support=sufficient_support)
    yield from run_process_just_and_fin(spec, state)
    assert state.previous_justified_checkpoint == c3
    if sufficient_support:
        assert state.current_justified_checkpoint == c2
        assert state.finalized_checkpoint == c4
    else:
        assert state.current_justified_checkpoint == c3
        assert state.finalized_checkpoint == old_finalized

# Remaining functions and tests have similar type annotations
