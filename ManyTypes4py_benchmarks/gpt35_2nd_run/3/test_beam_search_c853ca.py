from typing import List, Optional, Tuple
import numpy as np
import pytest
import torch as pt
import sockeye.beam_search
import sockeye.constants as C
import sockeye.lexicon
import sockeye.utils

def test_length_penalty_default() -> None:
    lengths: pt.Tensor = pt.tensor([[1], [2], [3]])
    length_penalty: sockeye.beam_search.LengthPenalty = sockeye.beam_search.LengthPenalty(1.0, 0.0)
    expected_lp: pt.Tensor = pt.tensor([[1.0], [2.0], [3.0]])
    pt.testing.assert_close(length_penalty(lengths), expected_lp)

def test_length_penalty() -> None:
    lengths: pt.Tensor = pt.tensor([[1], [2], [3]])
    length_penalty: sockeye.beam_search.LengthPenalty = sockeye.beam_search.LengthPenalty(0.2, 5.0)
    expected_lp: pt.Tensor = pt.tensor([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])
    pt.testing.assert_close(length_penalty(lengths), expected_lp)

def test_length_penalty_int_input() -> None:
    length: int = 1
    length_penalty: sockeye.beam_search.LengthPenalty = sockeye.beam_search.LengthPenalty(0.2, 5.0)
    expected_lp: List[float] = [6 ** 0.2 / 6 ** 0.2]
    assert np.isclose(length_penalty(length), expected_lp)

def test_brevity_penalty_default() -> None:
    hyp_lengths: pt.Tensor = pt.tensor([[1], [2], [3]])
    ref_lengths: pt.Tensor = pt.tensor([[2], [3], [2]])
    brevity_penalty: sockeye.beam_search.BrevityPenalty = sockeye.beam_search.BrevityPenalty(0.0)
    expected_bp: pt.Tensor = pt.tensor([[0], [0], [0]], dtype=pt.long)
    pt.testing.assert_close(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)

def test_brevity_penalty() -> None:
    hyp_lengths: pt.Tensor = pt.tensor([[1], [2], [3]])
    ref_lengths: pt.Tensor = pt.tensor([[7], [2], [91]])
    brevity_penalty: sockeye.beam_search.BrevityPenalty = sockeye.beam_search.BrevityPenalty(3.5)
    expected_bp: pt.Tensor = pt.tensor([[3.5 * (1 - 7 / 1)], [0.0], [3.5 * (1 - 91 / 3)]])
    pt.testing.assert_close(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)

def test_brevity_penalty_int_input() -> None:
    hyp_length: int = 3
    ref_length: int = 5
    brevity_penalty: sockeye.beam_search.BrevityPenalty = sockeye.beam_search.BrevityPenalty(2.0)
    expected_bp: List[float] = [2.0 * (1 - 5 / 3)]
    assert np.isclose(brevity_penalty(hyp_length, ref_length), expected_bp)

def test_candidate_scorer() -> None:
    scorer: sockeye.beam_search.CandidateScorer = sockeye.beam_search.CandidateScorer(length_penalty_alpha=1.0, length_penalty_beta=0.0, brevity_penalty_weight=0.1)
    raw_scores: pt.Tensor = pt.rand(5).unsqueeze(1)
    lengths: pt.Tensor = pt.tensor([1, 2, 3, 4, 5])
    reference_lengths: pt.Tensor = pt.tensor([2, 3, 4, 5, 6])
    scores: pt.Tensor = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores: pt.Tensor = scorer.unnormalize(scores, lengths, reference_lengths)
    pt.testing.assert_close(unnormalized_scores, raw_scores)
    raw_scores: float = 5.6
    lengths: int = 3
    reference_lengths: int = 4
    scores: pt.Tensor = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores: pt.Tensor = scorer.unnormalize(scores, lengths, reference_lengths)
    assert np.allclose(unnormalized_scores, raw_scores)

def numpy_topk(scores: np.ndarray, k: int, offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...

@pytest.mark.parametrize('batch_size, beam_size, target_vocab_size', [(1, 5, 200), (5, 5, 200), (1, 1, 200), (5, 1, 200), (10, 10, 100)])
def test_topk_func(batch_size: int, beam_size: int, target_vocab_size: int) -> None:
    ...

@pytest.mark.parametrize('target_vocab_size', [2, 10, 500, 1024])
def test_greedytop1(target_vocab_size: int) -> None:
    ...

@pytest.mark.parametrize('batch_size, beam_size, target_vocab_size, top_n', [(1, 5, 200, 0), (5, 5, 200, 0), (1, 100, 200, 5), (5, 100, 200, 5)])
def test_samplek_func(batch_size: int, beam_size: int, target_vocab_size: int, top_n: int) -> None:
    ...

@pytest.mark.parametrize('use_unk_dist', [False, True])
def test_update_scores(use_unk_dist: bool) -> None:
    ...

class _TestInference(sockeye.beam_search._Inference):

    def __init__(self, output_vocab_size: int):
        ...

    def state_structure(self) -> Tuple[str, str]:
        ...

    def encode_and_initialize(self, inputs: pt.Tensor, valid_length: Optional[pt.Tensor] = None) -> Tuple[List[pt.Tensor], pt.Tensor, Optional[pt.Tensor]]:
        ...

    def decode_step(self, step_input: pt.Tensor, states: List[pt.Tensor], vocab_slice_ids: Optional[pt.Tensor] = None, *args) -> Tuple[pt.Tensor, List[pt.Tensor], Optional[pt.Tensor]]:
        ...

    @property
    def model_output_vocab_size(self) -> int:
        ...

    @property
    def model_output_factor_vocab_size(self) -> Optional[int]:
        ...

def test_beam_search() -> None:
    ...

def test_get_nvs_vocab_slice_ids() -> None:
    ...

def test_get_vocab_slice_ids_blocking() -> None:
    ...
