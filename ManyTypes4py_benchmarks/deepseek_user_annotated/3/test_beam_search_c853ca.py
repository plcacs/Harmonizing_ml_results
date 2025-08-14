# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Optional, Tuple, Union, Any
import numpy as onp
import pytest
import torch as pt
import numpy as np

import sockeye.beam_search
import sockeye.constants as C
import sockeye.lexicon
import sockeye.utils


def test_length_penalty_default() -> None:
    lengths: pt.Tensor = pt.tensor([[1], [2], [3]])
    length_penalty: sockeye.beam_search.LengthPenalty = sockeye.beam_search.LengthPenalty(1.0, 0.0)
    expected_lp: pt.Tensor = pt.tensor([[1.0], [2.], [3.]])
    pt.testing.assert_close(length_penalty(lengths), expected_lp)


def test_length_penalty() -> None:
    lengths: pt.Tensor = pt.tensor([[1], [2], [3]])
    length_penalty: sockeye.beam_search.LengthPenalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp: pt.Tensor = pt.tensor([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])
    pt.testing.assert_close(length_penalty(lengths), expected_lp)


def test_length_penalty_int_input() -> None:
    length: int = 1
    length_penalty: sockeye.beam_search.LengthPenalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp: List[float] = [6 ** 0.2 / 6 ** 0.2]
    assert onp.isclose(length_penalty(length), expected_lp)


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
    assert onp.isclose(brevity_penalty(hyp_length, ref_length), expected_bp)


def test_candidate_scorer() -> None:
    scorer: sockeye.beam_search.CandidateScorer = sockeye.beam_search.CandidateScorer(length_penalty_alpha=1.0,
                                                                                     length_penalty_beta=0.0,
                                                                                     brevity_penalty_weight=0.1)

    raw_scores: pt.Tensor = pt.rand(5).unsqueeze(1)
    lengths: pt.Tensor = pt.tensor([1, 2, 3, 4, 5])
    reference_lengths: pt.Tensor = pt.tensor([2, 3, 4, 5, 6])

    scores: pt.Tensor = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores: pt.Tensor = scorer.unnormalize(scores, lengths, reference_lengths)
    pt.testing.assert_close(unnormalized_scores, raw_scores)

    # int/float input
    raw_scores: float = 5.6
    lengths: int = 3
    reference_lengths: int = 4

    scores: pt.Tensor = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores: pt.Tensor = scorer.unnormalize(scores, lengths, reference_lengths)
    assert onp.allclose(unnormalized_scores, raw_scores)


def numpy_topk(scores: onp.ndarray,
               k: int,
               offset: onp.ndarray) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Get the lowest k elements per sentence from a `scores` matrix using an intermediary Numpy conversion.
    This should be equivalent to sockeye.utils.topk() and is used as a comparative implementation in testing.

    :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
    :param k: The number of smallest scores to return.
    :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
    :return: The row indices, column indices and values of the k smallest items in matrix.
    """
    # (batch_size, beam_size * target_vocab_size)
    folded_scores: onp.ndarray = scores.reshape((-1, k * scores.shape[-1]))
    batch_size: int = folded_scores.shape[0]

    # Get the scores
    # Indexes into folded_scores: (batch_size, beam_size)
    flat_idxs: onp.ndarray = onp.argpartition(folded_scores, range(k))[:, :k]
    # Score values: (batch_size, beam_size)
    values: onp.ndarray = onp.array(folded_scores[onp.arange(folded_scores.shape[0])[:, None], flat_idxs])
    best_hyp_indices: onp.ndarray
    best_word_indices: onp.ndarray
    best_hyp_indices, best_word_indices = onp.array(onp.unravel_index(onp.ravel(flat_idxs), scores.shape),
                                                    dtype='int32')

    if batch_size > 1:
        # Offsetting the indices to match the shape of the scores matrix
        best_hyp_indices += offset

    values = values.reshape((-1, 1))
    return best_hyp_indices, best_word_indices, values


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size",
                         [(1, 5, 200),
                          (5, 5, 200),
                          (1, 1, 200),
                          (5, 1, 200),
                          (10, 10, 100)])
def test_topk_func(batch_size: int, beam_size: int, target_vocab_size: int) -> None:
    # Random model scores. Shape: (batch_size * beam_size, target_vocab_size)
    scores: onp.ndarray = onp.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    # offset for batch sizes > 1
    offset: onp.ndarray = onp.repeat(onp.arange(0, batch_size * beam_size, beam_size, dtype='int32'), beam_size)

    np_hyp: onp.ndarray
    np_word: onp.ndarray
    np_values: onp.ndarray
    np_hyp, np_word, np_values = numpy_topk(scores, k=beam_size, offset=offset)

    topk: sockeye.beam_search.TopK = sockeye.beam_search.TopK(k=beam_size)
    pt_hyp: pt.Tensor
    pt_word: pt.Tensor
    pt_values: pt.Tensor
    pt_hyp, pt_word, pt_values = topk(pt.tensor(scores))
    if batch_size > 1:
        # Offsetting the indices to match the shape of the scores matrix
        pt_hyp += pt.tensor(offset)
    assert onp.allclose(pt_hyp.detach().numpy(), np_hyp)
    assert onp.allclose(pt_word.detach().numpy(), np_word)
    assert onp.allclose(pt_values.detach().numpy(), np_values)


@pytest.mark.parametrize("target_vocab_size", [2, 10, 500, 1024])
def test_greedytop1(target_vocab_size: int) -> None:
    batch_size: int = 1
    beam_size: int = 1
    target_vocab_size: int = 50
    # Random model scores. Shape: (batch_size * beam_size, target_vocab_size)
    scores: onp.ndarray = onp.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    expected_hyp_index: onp.ndarray
    expected_word_index: onp.ndarray
    expected_value: onp.ndarray
    expected_hyp_index, expected_word_index, expected_value = numpy_topk(scores, k=beam_size, offset=None)
    assert expected_hyp_index[0] == 0
    assert expected_value.shape == (1, 1)

    greedy_top1: sockeye.beam_search.GreedyTop1 = sockeye.beam_search.GreedyTop1()

    best_word_index: pt.Tensor = greedy_top1(pt.tensor(scores), None, None).detach().numpy()
    assert best_word_index.shape == (1, 1)
    assert best_word_index[0, 0] == expected_word_index[0]

    target_factors: pt.Tensor = pt.ones(1, 1, 2, dtype=pt.int32)
    best_word_index_with_factors: pt.Tensor = greedy_top1(pt.tensor(scores), None, target_factors).detach().numpy()
    assert best_word_index_with_factors.shape == (1, 2)
    assert best_word_index_with_factors[0, 0] == expected_word_index[0]
    assert best_word_index_with_factors[0, 1] == target_factors[:, :, 1].item()


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size, top_n",
                         [(1, 5, 200, 0),
                          (5, 5, 200, 0),
                          (1, 100, 200, 5),
                          (5, 100, 200, 5)])
def test_samplek_func(batch_size: int, beam_size: int, target_vocab_size: int, top_n: int) -> None:
    # arrange scores increasing values from left to right, so the best item is always index 0, next-best 1, and so on
    scores: pt.Tensor = pt.tensor([list(range(1, target_vocab_size + 1)) for _ in range(batch_size * beam_size)])

    samplek: sockeye.beam_search.SampleK = sockeye.beam_search.SampleK(n=top_n)

    # 0..(batch_size * beam_size)-1
    expected_hyps: pt.Tensor = pt.tensor(range(batch_size * beam_size), dtype=pt.int32)
    finished: pt.Tensor = pt.rand(batch_size * beam_size) > 0.5

    for i in [1, 2]:
        hyps: pt.Tensor
        words: pt.Tensor
        values: pt.Tensor
        hyps, words, values = samplek(scores, scores, finished)
        assert hyps.shape[0] == batch_size * beam_size

        # The indices should always be the integers from 0 to batch*beam-1
        assert (hyps == expected_hyps).sum().item() == (batch_size * beam_size)
        if top_n != 0:
            # Scores are increasing left-to-right, so best items are all the lowest word IDs.
            # No word id greater than the cap (top_n) should be selected
            assert (words >= top_n).sum().item() == 0

        # word index should be zero for all finished hypotheses
        assert pt.where(finished, words, finished.long()).sum().item() == 0


@pytest.mark.parametrize("use_unk_dist", [False, True])
def test_update_scores(use_unk_dist: bool) -> None:
    vocab_size: int = 10
    batch_beam_size: int = 3
    us: sockeye.beam_search.UpdateScores = sockeye.beam_search.UpdateScores(prevent_unk=use_unk_dist)
    pad_dist: onp.ndarray = onp.full((1, vocab_size), fill_value=onp.inf, dtype='float32')
    pad_dist[0, 0] = 0
    eos_dist: onp.ndarray = onp.full((batch_beam_size, vocab_size), fill_value=onp.inf, dtype='float32')
    eos_dist[:, C.EOS_ID] = 0

    lengths: onp.ndarray = onp.array([0, 1, 0], dtype='int32')
    max_lengths: onp.ndarray = onp.array([1, 2, 3], dtype='int32')  # first on reaches max length
    scores_accumulated: onp.ndarray = onp.ones((3, 1), dtype='float32')
    finished: pt.Tensor = pt.tensor([False,  # not finished
                          True,  # finished
                          False],  # not finished
                         dtype=pt.bool)
    target_dists: onp.ndarray = onp.random.uniform(0, 1, (3, vocab_size)).astype('float32')

    scores: pt.Tensor
    lengths: pt.Tensor
    scores, lengths = us(pt.tensor(target_dists), finished,
                         pt.tensor(scores_accumulated), pt.tensor(lengths), pt.tensor(max_lengths),
                         pt.tensor(pad_dist), pt.tensor(eos_dist))
    scores_np: onp.ndarray = scores.detach().numpy()
    lengths_np: onp.ndarray = lengths
    pt.testing.assert_close(lengths, pt.tensor([1, 1, 1], dtype=pt.int32))  # all lengths but finished updated + 1
    assert (scores_np[0] == (1. + target_dists[0] + eos_dist)).all()  # 1 reached max length, force eos
    assert (scores_np[1] == (1. + pad_dist[0]).tolist()).all()  # 2 finished, force pad, keep score
    if use_unk_dist:
        assert scores_np[2, C.UNK_ID] == onp.inf  # 3 scores of <unk> should be np.inf
        target_dists[2, C.UNK_ID] = onp.inf
        assert (scores_np[2] == (1. + target_dists[2])).all()  # 3 scores + previous scores
    else:
        assert (scores_np[2] == (1. + target_dists[2])).all()  # 3 scores + previous scores


class _TestInference(sockeye.beam_search._Inference):

    def __init__(self, output_vocab_size: int) -> None:
        self.output_vocab_size: int = output_vocab_size
        self.states: List[Any] = []

    def state_structure(self) -> str:
        return C.STEP_STATE + C.STEP_STATE  # is this the correct structure to use for self.states?

    def encode_and_initialize(self,
                              inputs: pt.Tensor,
                              valid_length: Optional[pt.Tensor] = None) -> Tuple[List[pt.Tensor], pt.Tensor, Optional[pt.Tensor]]:
        batch_size: int = inputs.shape[0]
        # 'lengths'
        internal_lengths: pt.Tensor = pt.zeros(batch_size, 1, dtype=pt.int)
        num_decode_step_calls: pt.Tensor = pt.zeros(1, dtype=pt.int)
        self.states: List[pt.Tensor] = [internal_lengths, num_decode_step_calls]  # TODO add nested states
        predicted_output_length: pt.Tensor = pt.ones(batch_size, 1)  # does that work?
        nvs_prediction: Optional[pt.Tensor] = None
        return self.states, predicted_output_length, nvs_prediction

    def decode_step(self,
                    step_input: pt.Tensor,
                    states: List[pt.Tensor],
                    vocab_slice_ids: Optional[pt.Tensor] = None, *args) -> Tuple[pt.Tensor, List[pt.Tensor], Optional[pt.Tensor]]:
        batch_beam_size: int
        num_target_factors: int
        batch_beam_size, num_target_factors = step_input.size()
        print('step_input', step_input)

        internal_lengths: pt.Tensor
        num_decode_step_calls: pt.T