from typing import Tuple, Optional
from eth_typing import BLSPubkey
from curdleproofs import GenerateWhiskTrackerProof, WhiskTracker
from eth2spec.test.helpers.keys import whisk_ks_initial
from py_arkworks_bls12381 import G1Point, Scalar
whisk_initial_tracker_cache_by_index = {}
whisk_initial_k_commitment_cache_by_index = {}
whisk_initial_tracker_cache_by_k_r_G = {}
INITIAL_R = 1
G1 = G1Point()

def compute_whisk_initial_tracker_cached(i: int) -> Union[dict, str, bytes, list[str], list[tuple[typing.Union[int,float]]], dict[typing.Any, dict[str, str]]]:
    if i in whisk_initial_tracker_cache_by_index:
        return whisk_initial_tracker_cache_by_index[i]
    tracker = compute_whisk_tracker(whisk_ks_initial(i), INITIAL_R)
    whisk_initial_tracker_cache_by_index[i] = tracker
    whisk_initial_tracker_cache_by_k_r_G[tracker.k_r_G] = i
    return tracker

def compute_whisk_initial_k_commitment_cached(i: int) -> Union[dict, int, None, set, list, list[tuple[typing.Union[int,typing.Any]]], list[str]]:
    if i in whisk_initial_k_commitment_cache_by_index:
        return whisk_initial_k_commitment_cache_by_index[i]
    commitment = compute_whisk_k_commitment(whisk_ks_initial(i))
    whisk_initial_k_commitment_cache_by_index[i] = commitment
    return commitment

def resolve_known_tracker(tracker: dict) -> Union[float, None]:
    if tracker.k_r_G in whisk_initial_tracker_cache_by_k_r_G:
        return whisk_initial_tracker_cache_by_k_r_G[tracker.k_r_G]
    else:
        return None

def g1point_to_bytes(point: Union[bytes, float, list[bytes]]) -> bytes:
    return bytes(point.to_compressed_bytes())

def compute_whisk_k_commitment(k: Union[int, float, bytes]) -> Union[bytes, int]:
    return g1point_to_bytes(G1 * Scalar(k))

def compute_whisk_tracker(k: Union[float, int], r: Union[int, float]) -> WhiskTracker:
    r_G = G1 * Scalar(r)
    k_r_G = r_G * Scalar(k)
    return WhiskTracker(g1point_to_bytes(r_G), g1point_to_bytes(k_r_G))

def compute_whisk_tracker_and_commitment(k: Union[int, float], r: Union[int, float]) -> tuple[typing.Union[WhiskTracker,int,str,tuple]]:
    k_G = G1 * Scalar(k)
    r_G = G1 * Scalar(r)
    k_r_G = r_G * Scalar(k)
    tracker = WhiskTracker(g1point_to_bytes(r_G), g1point_to_bytes(k_r_G))
    commitment = g1point_to_bytes(k_G)
    return (tracker, commitment)

def set_as_first_proposal(spec: Union[int, list[int]], state: Union[int, list[int]], proposer_index: Union[int, list[int]]) -> None:
    if state.whisk_trackers[proposer_index].r_G != spec.BLS_G1_GENERATOR:
        assert state.whisk_trackers[proposer_index].r_G == spec.BLSG1Point()
        state.whisk_trackers[proposer_index].r_G = spec.BLS_G1_GENERATOR

def is_first_proposal(spec: Union[int, list[int], tuple[int]], state: Union[int, list[int], tuple[int]], proposer_index: Union[int, list[int], tuple[int]]) -> bool:
    return state.whisk_trackers[proposer_index].r_G == spec.BLS_G1_GENERATOR

def register_tracker(state: Union[int, typing.Iterable[T]], proposer_index: Union[int, typing.Iterable[T]], k: Union[int, typing.Iterable[T]], r: Union[int, typing.Iterable[T]]) -> None:
    tracker, k_commitment = compute_whisk_tracker_and_commitment(k, r)
    state.whisk_trackers[proposer_index] = tracker
    state.whisk_k_commitments[proposer_index] = k_commitment

def set_registration(body: Union[int, bytes, list[int], None], k: Union[int, dict], r: Union[int, None]) -> None:
    tracker, k_commitment = compute_whisk_tracker_and_commitment(k, r)
    body.whisk_registration_proof = GenerateWhiskTrackerProof(tracker, Scalar(k))
    body.whisk_tracker = tracker
    body.whisk_k_commitment = k_commitment

def set_opening_proof(spec: Union[str, int, dict], state: Union[list[int], int], block: int, proposer_index: Union[int, typing.Sequence[typing.Sequence[str]]], k: Union[bytes, dict, bytearray], r: Union[bytes, int, bytearray]) -> None:
    tracker, k_commitment = compute_whisk_tracker_and_commitment(k, r)
    state.whisk_proposer_trackers[state.slot % spec.WHISK_PROPOSER_TRACKERS_COUNT] = tracker
    state.whisk_k_commitments[proposer_index] = k_commitment
    block.proposer_index = proposer_index
    block.body.whisk_opening_proof = GenerateWhiskTrackerProof(tracker, Scalar(k))