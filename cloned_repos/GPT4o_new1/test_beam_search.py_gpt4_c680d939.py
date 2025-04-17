from typing import List, Optional, Tuple

import numpy as onp
import pytest
import torch as pt
import numpy as np

import sockeye.beam_search
import sockeye.constants as C
import sockeye.lexicon
import sockeye.utils


def test_length_penalty_default() -> None:
    lengths = pt.tensor([[1], [2], [3]])
    length_penalty = sockeye.beam_search.LengthPenalty(1.0, 0.0)
    expected_lp = pt.tensor([[1.0], [2.], [3.]])
    pt.testing.assert_close(length_penalty(lengths), expected_lp)


def test_length_penalty() -> None:
    lengths = pt.tensor([[1], [2], [3]])
    length_penalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp = pt.tensor([[6 ** 0.2 / 6 ** 0.2], [7 ** 0.2 / 6 ** 0.2], [8 ** 0.2 / 6 ** 0.2]])
    pt.testing.assert_close(length_penalty(lengths), expected_lp)


def test_length_penalty_int_input() -> None:
    length = 1
    length_penalty = sockeye.beam_search.LengthPenalty(.2, 5.0)
    expected_lp = [6 ** 0.2 / 6 ** 0.2]
    assert onp.isclose(length_penalty(length), expected_lp)


def test_brevity_penalty_default() -> None:
    hyp_lengths = pt.tensor([[1], [2], [3]])
    ref_lengths = pt.tensor([[2], [3], [2]])
    brevity_penalty = sockeye.beam_search.BrevityPenalty(0.0)
    expected_bp = pt.tensor([[0], [0], [0]], dtype=pt.long)
    pt.testing.assert_close(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)


def test_brevity_penalty() -> None:
    hyp_lengths = pt.tensor([[1], [2], [3]])
    ref_lengths = pt.tensor([[7], [2], [91]])
    brevity_penalty = sockeye.beam_search.BrevityPenalty(3.5)
    expected_bp = pt.tensor([[3.5 * (1 - 7 / 1)], [0.0], [3.5 * (1 - 91 / 3)]])
    pt.testing.assert_close(brevity_penalty(hyp_lengths, ref_lengths), expected_bp)


def test_brevity_penalty_int_input() -> None:
    hyp_length = 3
    ref_length = 5
    brevity_penalty = sockeye.beam_search.BrevityPenalty(2.0)
    expected_bp = [2.0 * (1 - 5 / 3)]

    assert onp.isclose(brevity_penalty(hyp_length, ref_length), expected_bp)


def test_candidate_scorer() -> None:
    scorer = sockeye.beam_search.CandidateScorer(length_penalty_alpha=1.0,
                                                 length_penalty_beta=0.0,
                                                 brevity_penalty_weight=0.1)

    raw_scores = pt.rand(5).unsqueeze(1)
    lengths = pt.tensor([1, 2, 3, 4, 5])
    reference_lengths = pt.tensor([2, 3, 4, 5, 6])

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    pt.testing.assert_close(unnormalized_scores, raw_scores)

    # int/float input
    raw_scores = 5.6
    lengths = 3
    reference_lengths = 4

    scores = scorer(raw_scores, lengths, reference_lengths)
    unnormalized_scores = scorer.unnormalize(scores, lengths, reference_lengths)
    assert onp.allclose(unnormalized_scores, raw_scores)


def numpy_topk(scores: onp.ndarray,
               k: int,
               offset: onp.ndarray) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    folded_scores = scores.reshape((-1, k * scores.shape[-1]))
    batch_size = folded_scores.shape[0]

    flat_idxs = onp.argpartition(folded_scores, range(k))[:, :k]
    values = onp.array(folded_scores[onp.arange(folded_scores.shape[0])[:, None], flat_idxs])
    best_hyp_indices, best_word_indices = onp.array(onp.unravel_index(onp.ravel(flat_idxs), scores.shape),
                                                    dtype='int32')

    if batch_size > 1:
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
    scores = onp.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    offset = onp.repeat(onp.arange(0, batch_size * beam_size, beam_size, dtype='int32'), beam_size)

    np_hyp, np_word, np_values = numpy_topk(scores, k=beam_size, offset=offset)

    topk = sockeye.beam_search.TopK(k=beam_size)
    pt_hyp, pt_word, pt_values = topk(pt.tensor(scores))
    if batch_size > 1:
        pt_hyp += pt.tensor(offset)
    assert onp.allclose(pt_hyp.detach().numpy(), np_hyp)
    assert onp.allclose(pt_word.detach().numpy(), np_word)
    assert onp.allclose(pt_values.detach().numpy(), np_values)


@pytest.mark.parametrize("target_vocab_size", [2, 10, 500, 1024])
def test_greedytop1(target_vocab_size: int) -> None:
    batch_size = 1
    beam_size = 1
    target_vocab_size = 50
    scores = onp.random.uniform(0, 1, (batch_size * beam_size, target_vocab_size))
    expected_hyp_index, expected_word_index, expected_value = numpy_topk(scores, k=beam_size, offset=None)
    assert expected_hyp_index[0] == 0
    assert expected_value.shape == (1, 1)

    greedy_top1 = sockeye.beam_search.GreedyTop1()

    best_word_index = greedy_top1(pt.tensor(scores), None, None).detach().numpy()
    assert best_word_index.shape == (1, 1)
    assert best_word_index[0, 0] == expected_word_index[0]

    target_factors = pt.ones(1, 1, 2, dtype=pt.int32)
    best_word_index_with_factors = greedy_top1(pt.tensor(scores), None, target_factors).detach().numpy()
    assert best_word_index_with_factors.shape == (1, 2)
    assert best_word_index_with_factors[0, 0] == expected_word_index[0]
    assert best_word_index_with_factors[0, 1] == target_factors[:, :, 1].item()


@pytest.mark.parametrize("batch_size, beam_size, target_vocab_size, top_n",
                         [(1, 5, 200, 0),
                          (5, 5, 200, 0),
                          (1, 100, 200, 5),
                          (5, 100, 200, 5)])
def test_samplek_func(batch_size: int, beam_size: int, target_vocab_size: int, top_n: int) -> None:
    scores = pt.tensor([list(range(1, target_vocab_size + 1)) for _ in range(batch_size * beam_size)])

    samplek = sockeye.beam_search.SampleK(n=top_n)

    expected_hyps = pt.tensor(range(batch_size * beam_size), dtype=pt.int32)
    finished = pt.rand(batch_size * beam_size) > 0.5

    for i in [1, 2]:
        hyps, words, values = samplek(scores, scores, finished)
        assert hyps.shape[0] == batch_size * beam_size

        assert (hyps == expected_hyps).sum().item() == (batch_size * beam_size)
        if top_n != 0:
            assert (words >= top_n).sum().item() == 0

        assert pt.where(finished, words, finished.long()).sum().item() == 0


@pytest.mark.parametrize("use_unk_dist", [False, True])
def test_update_scores(use_unk_dist: bool) -> None:
    vocab_size = 10
    batch_beam_size = 3
    us = sockeye.beam_search.UpdateScores(prevent_unk=use_unk_dist)
    pad_dist = onp.full((1, vocab_size), fill_value=onp.inf, dtype='float32')
    pad_dist[0, 0] = 0
    eos_dist = onp.full((batch_beam_size, vocab_size), fill_value=onp.inf, dtype='float32')
    eos_dist[:, C.EOS_ID] = 0

    lengths = onp.array([0, 1, 0], dtype='int32')
    max_lengths = onp.array([1, 2, 3], dtype='int32')
    scores_accumulated = onp.ones((3, 1), dtype='float32')
    finished = pt.tensor([False, True, False], dtype=pt.bool)
    target_dists = onp.random.uniform(0, 1, (3, vocab_size)).astype('float32')

    scores, lengths = us(pt.tensor(target_dists), finished,
                         pt.tensor(scores_accumulated), pt.tensor(lengths), pt.tensor(max_lengths),
                         pt.tensor(pad_dist), pt.tensor(eos_dist))
    scores = scores.detach().numpy()
    lengths = lengths
    pt.testing.assert_close(lengths, pt.tensor([1, 1, 1], dtype=pt.int32))
    assert (scores[0] == (1. + target_dists[0] + eos_dist)).all()
    assert (scores[1] == (1. + pad_dist[0]).tolist()).all()
    if use_unk_dist:
        assert scores[2, C.UNK_ID] == onp.inf
        target_dists[2, C.UNK_ID] = onp.inf
        assert (scores[2] == (1. + target_dists[2])).all()
    else:
        assert (scores[2] == (1. + target_dists[2])).all()


class _TestInference(sockeye.beam_search._Inference):

    def __init__(self, output_vocab_size: int) -> None:
        self.output_vocab_size = output_vocab_size
        self.states = []

    def state_structure(self) -> str:
        return C.STEP_STATE + C.STEP_STATE

    def encode_and_initialize(self,
                              inputs: pt.Tensor,
                              valid_length: Optional[pt.Tensor] = None) -> Tuple[List, pt.Tensor, Optional[pt.Tensor]]:
        batch_size = inputs.shape[0]
        internal_lengths = pt.zeros(batch_size, 1, dtype=pt.int)
        num_decode_step_calls = pt.zeros(1, dtype=pt.int)
        self.states = [internal_lengths, num_decode_step_calls]
        predicted_output_length = pt.ones(batch_size, 1)
        nvs_prediction = None
        return self.states, predicted_output_length, nvs_prediction

    def decode_step(self,
                    step_input: pt.Tensor,
                    states: List,
                    vocab_slice_ids: Optional[pt.Tensor] = None, *args) -> Tuple[pt.Tensor, List, Optional[pt.Tensor]]:
        batch_beam_size, num_target_factors = step_input.size()
        print('step_input', step_input)

        internal_lengths, num_decode_step_calls = states
        num_decode_step_calls = num_decode_step_calls.item()
        if num_decode_step_calls == 0:
            assert (step_input == C.BOS_ID).all()

        if step_input[:, 0].item() == C.BOS_ID:
            scores = pt.tensor([0, 0, 0, 0, 1])
        elif step_input[:, 0].item() == C.EOS_ID:
            scores = pt.tensor([1, 0, 0, 0, 0])
        else:
            scores = pt.tensor([0, 0, 0, 0, 1])

        scores *= -1

        internal_lengths += 1
        num_decode_step_calls += 1

        self.states = states = [internal_lengths, pt.tensor([num_decode_step_calls], dtype=pt.int)]
        return scores, states, None

    @property
    def model_output_vocab_size(self) -> int:
        return self.output_vocab_size

    @property
    def model_output_factor_vocab_size(self) -> Optional[int]:
        return None


def test_beam_search() -> None:
    device = pt.device('cpu')
    dtype = pt.float32
    num_source_factors = 1
    num_target_factors = 1
    vocab_size = len(C.VOCAB_SYMBOLS) + 1
    beam_size = 1
    bos_id = 2
    eos_id = 3

    inference = _TestInference(output_vocab_size=vocab_size)
    bs = sockeye.beam_search.BeamSearch(
        beam_size=beam_size,
        dtype=dtype,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        output_vocab_size=vocab_size,
        scorer=sockeye.beam_search.CandidateScorer(),
        num_source_factors=num_source_factors,
        num_target_factors=num_target_factors,
        inference=inference,
        beam_search_stop=C.BEAM_SEARCH_STOP_ALL,
        sample=None)

    batch_size = 1
    max_length = 3
    source = pt.tensor([[C.BOS_ID, 4, C.EOS_ID, C.PAD_ID, C.PAD_ID]], dtype=dtype).reshape(1, -1, 1)
    source_length = (source != C.PAD_ID).sum(1).reshape(-1)

    restrict_lexicon = None
    max_output_lengths = pt.tensor([max_length], dtype=pt.int)

    bs_out = bs(source, source_length, restrict_lexicon, max_output_lengths)
    r = bs_out

    print('beam search lengths', r.lengths)
    print('internal lengths', inference.states[0])
    pt.testing.assert_close(r.lengths, inference.states[0].squeeze(1))
    assert inference.states[1] == max_length


def test_get_nvs_vocab_slice_ids() -> None:
    nvs_prediction = pt.tensor([[0.1, 0.1, 0.1, 0.1, 0.7,  0.0, 0.8,  0.0,  0.0, 0.0],
                                [0.1, 0.1, 0.1, 0.1, 0.55, 0.0, 0.49, 0.05, 0.0, 0.0]])
    expected_bow = pt.tensor([0, 1, 2, 3, 4, 6, C.EOS_ID, C.EOS_ID])
    bow, output_vocab_size = sockeye.beam_search._get_nvs_vocab_slice_ids(nvs_thresh=0.5,
                                                                          nvs_prediction=nvs_prediction)
    assert output_vocab_size == expected_bow.shape[0]
    pt.testing.assert_close(bow, expected_bow)

    nvs_prediction = pt.tensor([[0.1, 0.1, 0.1, 0.1, 0.7,  0.0, 0.0,  0.8,  0.0, 0.0]])
    expected_bow = pt.tensor([0, 1, 2, 3, 4, 7, C.EOS_ID, C.EOS_ID])
    bow, output_vocab_size = sockeye.beam_search._get_nvs_vocab_slice_ids(nvs_thresh=0.5,
                                                                          nvs_prediction=nvs_prediction)
    assert output_vocab_size == expected_bow.shape[0]
    pt.testing.assert_close(bow, expected_bow)

    nvs_prediction = pt.tensor([[0.1, 0.1, 0.1, 0.1, 0.7,  0.0, 0.0,  0.8,  0.0, 0.0]])
    expected_bow = pt.tensor([0, 1, 2, 3, C.EOS_ID, C.EOS_ID, C.EOS_ID, C.EOS_ID])
    bow, output_vocab_size = sockeye.beam_search._get_nvs_vocab_slice_ids(nvs_thresh=0.9,
                                                                          nvs_prediction=nvs_prediction)
    assert output_vocab_size == expected_bow.shape[0]
    pt.testing.assert_close(bow, expected_bow)

    nvs_prediction = pt.tensor([[0.1, 0.1, 0.1, 0.1, 0.7,  0.0, 0.8,  0.0,  0.0, 0.0],
                                [0.1, 0.1, 0.1, 0.1, 0.55, 0.0, 0.49, 0.05, 0.0, 0.0]])
    target_prefix = pt.tensor([[8, 8], [8, 8]])
    expected_bow = pt.tensor([0, 1, 2, 3, 4, 6, 8, C.EOS_ID])
    bow, output_vocab_size = sockeye.beam_search._get_nvs_vocab_slice_ids(nvs_thresh=0.5,
                                                                          nvs_prediction=nvs_prediction,
                                                                          target_prefix=target_prefix)
    assert output_vocab_size == expected_bow.shape[0]
    pt.testing.assert_close(bow, expected_bow)

    nvs_prediction = pt.tensor([[0.1, 0.1, 0.1, 0.1, 0.7,  0.0, 0.8,  0.0,  0.0, 0.0],
                                [0.1, 0.1, 0.1, 0.1, 0.55, 0.0, 0.49, 0.05, 0.0, 0.0]])
    expected_bow = pt.tensor([0, 1, 2, 3, 4, C.EOS_ID, C.EOS_ID, C.EOS_ID])
    restrict_lexicon = sockeye.lexicon.StaticBlockLexicon(
        np.array([6])
    )
    bow, output_vocab_size = sockeye.beam_search._get_nvs_vocab_slice_ids(nvs_thresh=0.5,
                                                                          nvs_prediction=nvs_prediction,
                                                                          restrict_lexicon=restrict_lexicon)
    assert output_vocab_size == expected_bow.shape[0]
    pt.testing.assert_close(bow, expected_bow)


def test_get_vocab_slice_ids_blocking() -> None:
    restrict_lexicon = sockeye.lexicon.StaticBlockLexicon(
        np.array([3])
    )
    source_words = pt.tensor([1, 2, 3])
    vocab_slice_ids, _ = sockeye.beam_search._get_vocab_slice_ids(
        restrict_lexicon=restrict_lexicon,
        source_words=source_words,
        eos_id=C.EOS_ID,
        beam_size=5,
        target_prefix=None,
        output_vocab_size=6
    )
    expected_vocab_slice_ids = pt.tensor([0, 1, 2, 4, 5, C.EOS_ID, C.EOS_ID, C.EOS_ID])
    pt.testing.assert_close(vocab_slice_ids, expected_vocab_slice_ids)
