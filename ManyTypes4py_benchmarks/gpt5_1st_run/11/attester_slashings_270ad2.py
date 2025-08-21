from typing import Any, Callable, List, Optional, Sequence

from eth2spec.test.helpers.attestations import get_valid_attestation, sign_attestation, sign_indexed_attestation
from eth2spec.test.helpers.forks import is_post_electra


def get_valid_attester_slashing(
    spec: Any,
    state: Any,
    slot: Optional[int] = None,
    signed_1: bool = False,
    signed_2: bool = False,
    filter_participant_set: Optional[Callable[[Sequence[int]], Sequence[int]]] = None,
) -> Any:
    attestation_1: Any = get_valid_attestation(
        spec, state, slot=slot, signed=signed_1, filter_participant_set=filter_participant_set
    )
    attestation_2: Any = attestation_1.copy()
    attestation_2.data.target.root = b"\x01" * 32
    if signed_2:
        sign_attestation(spec, state, attestation_2)
    return spec.AttesterSlashing(
        attestation_1=spec.get_indexed_attestation(state, attestation_1),
        attestation_2=spec.get_indexed_attestation(state, attestation_2),
    )


def get_valid_attester_slashing_by_indices(
    spec: Any,
    state: Any,
    indices_1: Sequence[int],
    indices_2: Optional[Sequence[int]] = None,
    slot: Optional[int] = None,
    signed_1: bool = False,
    signed_2: bool = False,
) -> Any:
    if indices_2 is None:
        indices_2 = indices_1
    assert list(indices_1) == sorted(indices_1)
    assert list(indices_2) == sorted(indices_2)
    attester_slashing: Any = get_valid_attester_slashing(spec, state, slot=slot)
    attester_slashing.attestation_1.attesting_indices = indices_1
    attester_slashing.attestation_2.attesting_indices = indices_2
    if signed_1:
        sign_indexed_attestation(spec, state, attester_slashing.attestation_1)
    if signed_2:
        sign_indexed_attestation(spec, state, attester_slashing.attestation_2)
    return attester_slashing


def get_indexed_attestation_participants(spec: Any, indexed_att: Any) -> List[int]:
    """
    Wrapper around index-attestation to return the list of participant indices, regardless of spec phase.
    """
    return list(indexed_att.attesting_indices)


def set_indexed_attestation_participants(spec: Any, indexed_att: Any, participants: Sequence[int]) -> None:
    """
    Wrapper around index-attestation to return the list of participant indices, regardless of spec phase.
    """
    indexed_att.attesting_indices = participants


def get_attestation_1_data(spec: Any, att_slashing: Any) -> Any:
    return att_slashing.attestation_1.data


def get_attestation_2_data(spec: Any, att_slashing: Any) -> Any:
    return att_slashing.attestation_2.data


def get_max_attester_slashings(spec: Any) -> int:
    if is_post_electra(spec):
        return spec.MAX_ATTESTER_SLASHINGS_ELECTRA
    else:
        return spec.MAX_ATTESTER_SLASHINGS