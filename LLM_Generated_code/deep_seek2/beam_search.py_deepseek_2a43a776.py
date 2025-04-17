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

import functools
import logging
import operator
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch as pt

import sockeye.constants as C
from . import lexicon
from . import utils
from .model import SockeyeModel

logger = logging.getLogger(__name__)


class _Inference(ABC):

    @abstractmethod
    def state_structure(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def encode_and_initialize(self,
                              inputs: pt.Tensor,
                              valid_length: pt.Tensor) -> Tuple[List[pt.Tensor], pt.Tensor, Optional[pt.Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def decode_step(self,
                    step_input: pt.Tensor,
                    states: List[pt.Tensor],
                    vocab_slice_ids: Optional[pt.Tensor] = None,
                    target_prefix_factor_mask: Optional[pt.Tensor] = None,
                    factor_vocab_size: Optional[int] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], Optional[List[pt.Tensor]]]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def model_output_vocab_size(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def model_output_factor_vocab_size(self) -> int:
        raise NotImplementedError()


class _SingleModelInference(_Inference):

    def __init__(self,
                 model: SockeyeModel,
                 skip_softmax: bool = False,
                 constant_length_ratio: float = 0.0,
                 knn_lambda: float = C.DEFAULT_KNN_LAMBDA) -> None:
        self._model: SockeyeModel = model
        self._skip_softmax: bool = skip_softmax
        self._const_lr: float = constant_length_ratio
        self.knn_lambda: float = knn_lambda

    def state_structure(self) -> List[str]:
        return [self._model.state_structure()]

    def encode_and_initialize(self, inputs: pt.Tensor, valid_length: pt.Tensor) -> Tuple[List[pt.Tensor], pt.Tensor, Optional[pt.Tensor]]:
        states, predicted_output_length, nvs_prediction = self._model.encode_and_initialize(inputs, valid_length, self._const_lr)
        return states, predicted_output_length, nvs_prediction

    def decode_step(self,
                    step_input: pt.Tensor,
                    states: List[pt.Tensor],
                    vocab_slice_ids: Optional[pt.Tensor] = None,
                    target_prefix_factor_mask: Optional[pt.Tensor] = None,
                    factor_vocab_size: Optional[int] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], Optional[List[pt.Tensor]]]:
        logits, knn_probs, states, target_factor_outputs = self._model.decode_step(step_input, states, vocab_slice_ids)
        if not self._skip_softmax:
            if knn_probs is None:  # no knn used
                probs = pt.log_softmax(logits, dim=-1)
            else:
                probs = pt.log(self.knn_lambda * pt.softmax(logits, dim=-1) + (1 - self.knn_lambda) * knn_probs)
        else:
            assert knn_probs is None, "Can't skip softmax with KNN."
            probs = logits

        scores = -probs  # shape: (batch*beam, output_vocab_size/len(vocab_slice_ids))

        target_factors = None  # type: Optional[pt.Tensor]
        if target_factor_outputs:
            predictions = []  # type: List[pt.Tensor]
            for i, tf_logits in enumerate(target_factor_outputs, 1):
                if not self._skip_softmax:
                    tf_logits = pt.log_softmax(tf_logits, dim=-1)
                tf_scores = -tf_logits
                if target_prefix_factor_mask is not None:
                    tf_scores += target_prefix_factor_mask[:, :, i-1].reshape(-1, factor_vocab_size)
                # target factors are greedily chosen, and score and index are collected via torch.min.
                # Shape per factor: (batch*beam, 1, 2), where last dimension holds values and indices.
                tf_prediction = pt.cat(tf_scores.min(dim=-1, keepdim=True), dim=1).unsqueeze(1)
                predictions.append(tf_prediction)
            # Shape: (batch*beam, num_secondary_factors, 2)
            target_factors = pt.cat(predictions, dim=1) if len(predictions) > 1 else predictions[0]

        return scores, states, target_factors

    @property
    def model_output_vocab_size(self) -> int:
        return self._model.output_layer_vocab_size

    @property
    def model_output_factor_vocab_size(self) -> int:
        return self._model.factor_vocab_size


class _EnsembleInference(_Inference):

    def __init__(self,
                 models: List[SockeyeModel],
                 ensemble_mode: str = 'linear',
                 constant_length_ratio: float = 0.0,
                 knn_lambda: float = C.DEFAULT_KNN_LAMBDA) -> None:
        self._models: List[SockeyeModel] = models
        if ensemble_mode == 'linear':
            self._interpolation: Callable[[List[pt.Tensor]], pt.Tensor] = self.linear_interpolation
        elif ensemble_mode == 'log_linear':
            self._interpolation: Callable[[List[pt.Tensor]], pt.Tensor] = self.log_linear_interpolation
        else:
            raise ValueError()
        self._const_lr: float = constant_length_ratio
        self.knn_lambda: float = knn_lambda

    def state_structure(self) -> List[List[str]]:
        return [model.state_structure() for model in self._models]

    def encode_and_initialize(self, inputs: pt.Tensor, valid_length: pt.Tensor) -> Tuple[List[pt.Tensor], pt.Tensor, Optional[pt.Tensor]]:
        model_states = []  # type: List[pt.Tensor]
        predicted_output_lengths = []  # type: List[pt.Tensor]
        nvs_predictions = []
        for model in self._models:
            states, predicted_output_length, nvs_prediction = model.encode_and_initialize(inputs, valid_length, self._const_lr)
            if nvs_prediction is not None:
                nvs_predictions.append(nvs_prediction)

            predicted_output_lengths.append(predicted_output_length)
            model_states += states
        # average predicted output lengths, (batch, 1)
        predicted_output_lengths = pt.stack(predicted_output_lengths, dim=1).float().mean(dim=1)  # type: ignore
        nvs_prediction = pt.stack(nvs_predictions, dim=1).mean(dim=1) if nvs_predictions else None

        return model_states, predicted_output_lengths, nvs_prediction

    def decode_step(self,
                    step_input: pt.Tensor,
                    states: List[pt.Tensor],
                    vocab_slice_ids: Optional[pt.Tensor] = None,
                    target_prefix_factor_mask: Optional[pt.Tensor] = None,
                    factor_vocab_size: Optional[int] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], Optional[List[pt.Tensor]]]:
        outputs = []  # type: List[pt.Tensor]
        new_states = []  # type: List[pt.Tensor]
        factor_outputs = []  # type: List[List[pt.Tensor]]
        state_index = 0
        for model, model_state_structure in zip(self._models, self.state_structure()):
            model_states = states[state_index:state_index+len(model_state_structure)]
            state_index += len(model_state_structure)
            logits, knn_probs, model_states, target_factor_outputs = model.decode_step(step_input, model_states, vocab_slice_ids)
            if knn_probs is None:
                probs = logits.softmax(dim=-1)
            else:
                probs = self.knn_lambda * pt.softmax(logits, dim=-1) + (1 - self.knn_lambda) * knn_probs
            outputs.append(probs)
            if target_factor_outputs:
                target_factor_probs = [tfo.softmax(dim=-1) for tfo in target_factor_outputs]
                if target_prefix_factor_mask is not None:
                    for i in range(len(target_factor_probs)):
                        target_factor_probs[i] += target_prefix_factor_mask[:, :, i].reshape(-1, factor_vocab_size)
                factor_outputs.append(target_factor_probs)
            new_states += model_states
        scores = self._interpolation(outputs)

        target_factors = None  # type: Optional[pt.Tensor]
        if factor_outputs:
            predictions = []  # type: List[pt.Tensor]
            for model_tf_logits in zip(*factor_outputs):
                tf_prediction = pt.cat(self._interpolation(model_tf_logits).min(dim=-1, keepdim=True), dim=1).unsqueeze(1)
                predictions.append(tf_prediction)
            target_factors = pt.cat(predictions, dim=1) if len(predictions) > 1 else predictions[0]
        return scores, new_states, target_factors

    @staticmethod
    def linear_interpolation(predictions: List[pt.Tensor]) -> pt.Tensor:
        return -(utils.average_tensors(predictions).log())

    @staticmethod
    def log_linear_interpolation(predictions: List[pt.Tensor]) -> pt.Tensor:
        log_probs = utils.average_tensors([p.log() for p in predictions])
        return -(log_probs.log_softmax())

    @property
    def model_output_vocab_size(self) -> int:
        return self._models[0].output_layer_vocab_size

    @property
    def model_output_factor_vocab_size(self) -> int:
        return self._models[0].factor_vocab_size


@dataclass
class SearchResult:
    """
    Holds return values from Search algorithms
    """
    best_hyp_indices: pt.Tensor
    best_word_indices: pt.Tensor
    accumulated_scores: pt.Tensor
    lengths: pt.Tensor
    estimated_reference_lengths: pt.Tensor


class UpdateScores(pt.nn.Module):
    """
    A Module that updates the scores from the decoder step with accumulated scores.
    Finished hypotheses receive their accumulated score for C.PAD_ID.
    Hypotheses at maximum length are forced to produce C.EOS_ID.
    All other options are set to infinity.
    """

    def __init__(self, prevent_unk: bool = False):
        super().__init__()
        self.prevent_unk: bool = prevent_unk
        assert C.PAD_ID == 0, "This block only works with PAD_ID == 0"

    def forward(self, target_dists: pt.Tensor, finished: pt.Tensor, scores_accumulated: pt.Tensor, lengths: pt.Tensor, max_lengths: pt.Tensor, pad_dist: pt.Tensor, eos_dist: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:

        if self.prevent_unk:  # make sure to avoid generating <unk>
            target_dists[:, C.UNK_ID] = np.inf

        # broadcast hypothesis score to each prediction.
        # scores_accumulated. Shape: (batch*beam, 1)
        # target_dists. Shape: (batch*beam, vocab_size)
        scores = target_dists + scores_accumulated

        # Special treatment for finished rows.
        # Finished rows are inf everywhere except column zero (pad_id), which holds the accumulated model score.
        # Items that are finished get their previous accumulated score for the <pad> symbol,
        # infinity otherwise.
        pad_dist = scores_accumulated + pad_dist  # (batch*beam, vocab_size)
        scores = pt.where(finished.unsqueeze(1), pad_dist, scores)
        # Update lengths of all items, except those that were already finished.
        lengths = lengths + ~finished
        # Items that are at their maximum length and not finished now are forced to produce the <eos> symbol.
        # That is, we keep scores for hypotheses below max length or finished, and 'force-eos' the rest.
        below_max_length = lengths < max_lengths  # type: pt.Tensor
        scores = pt.where(pt.logical_or(below_max_length, finished).unsqueeze(1), scores, eos_dist + scores)

        return scores, lengths


class LengthPenalty(pt.nn.Module):
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016 (note that in the paper beta has a different meaning,
    and a fixed value 5 was used for this parameter)

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.denominator: float = (self.beta + 1.) ** self.alpha

    def forward(self, lengths: Union[int, float, pt.Tensor]) -> Union[float, pt.Tensor]:
        if self.alpha == 0.0:
            if isinstance(lengths, (int, float)):
                return 1.0
            else:
                return pt.ones_like(lengths)
        else:
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator


class BrevityPenalty(pt.nn.Module):
    """
    Calculates the logarithmic brevity penalty as:
      weight * log min(1, exp(1 - ref_len / hyp_len)) = weight * min(0, 1 - ref_len / hyp_len).

    :param weight: Linear weight.
    """

    def __init__(self, weight: float = 0.0) -> None:
        super().__init__()
        self.weight: float = weight

    def forward(self, hyp_lengths: Union[int, float, pt.Tensor], reference_lengths: Union[int, float, pt.Tensor]) -> Union[float, pt.Tensor]:
        if self.weight == 0.0:
            if isinstance(hyp_lengths, (int, float)):
                return 0.0
            else:
                return pt.zeros_like(hyp_lengths)
        else:
            # log_bp is always <= 0.0
            if isinstance(hyp_lengths, (int, float)):
                log_bp = min(0.0, 1.0 - reference_lengths / hyp_lengths)
            else:
                log_bp = pt.minimum(pt.zeros_like(hyp_lengths, dtype=pt.float),
                                    1.0 - reference_lengths / hyp_lengths.float())
            return self.weight * log_bp


class CandidateScorer(pt.nn.Module):

    def __init__(self,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0,
                 brevity_penalty_weight: float = 0.0) -> None:
        super().__init__()
        self._lp: LengthPenalty = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)
        self._bp: Optional[BrevityPenalty] = None  # type: Optional[BrevityPenalty]
        if brevity_penalty_weight > 0.0:
            self._bp = BrevityPenalty(weight=brevity_penalty_weight)

    def forward(self, scores: Union[float, pt.Tensor], lengths: Union[int, float, pt.Tensor], reference_lengths: Union[int, float, pt.Tensor]) -> Union[float, pt.Tensor]:
        lp = self._lp(lengths)
        if self._bp is not None:
            bp = self._bp(lengths, reference_lengths)
        else:
            bp = 0.0
        if isinstance(scores, (int, float)):
            return scores / lp - bp
        else:
            if isinstance(lp, pt.Tensor):
                lp = lp.to(scores.dtype)
            if isinstance(bp, pt.Tensor):
                bp = bp.to(scores.dtype)
            return (scores.squeeze(1) / lp - bp).unsqueeze(1)

    def unnormalize(self, scores: Union[float, pt.Tensor], lengths: Union[int, float, pt.Tensor], reference_lengths: Union[int, float, pt.Tensor]) -> Union[float, pt.Tensor]:
        bp = 0.0 if self._bp is None else self._bp(lengths, reference_lengths)
        if isinstance(scores, (int, float)):
            return (scores + bp) * self._lp(lengths)
        else:
            return ((scores.squeeze(1) + bp) * self._lp(lengths)).unsqueeze(1)


class SortNormalizeAndUpdateFinished(pt.nn.Module):
    """
    A Module for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self,
                 pad_id: int,
                 eos_id: int,
                 scorer: CandidateScorer,
                 expect_factors: bool) -> None:
        super().__init__()
        self.pad_id: int = pad_id
        self.eos_id: int = eos_id
        self._scorer: CandidateScorer = scorer
        self.expect_factors: bool = expect_factors

    def forward(self, best_hyp_indices: pt.Tensor, best_word_indices: pt.Tensor,
                finished: pt.Tensor, scores_accumulated: pt.Tensor, lengths: pt.Tensor, reference_lengths: pt.Tensor,
                *factor_args: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, List[pt.Tensor], pt.Tensor, pt.Tensor]:

        # Reorder fixed-size beam data according to best_hyp_indices (ascending)
        finished = finished.index_select(0, best_hyp_indices)
        lengths = lengths.index_select(0, best_hyp_indices)
        reference_lengths = reference_lengths.index_select(0, best_hyp_indices)

        # Normalize hypotheses that JUST finished
        all_finished = pt.logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)
        newly_finished = pt.logical_xor(all_finished, finished).unsqueeze(1)
        scores_accumulated = pt.where(newly_finished,
                                      self._scorer(scores_accumulated, lengths, reference_lengths),
                                      scores_accumulated)

        # Recompute finished. Hypotheses are finished if they are extended with <pad> or <eos>
        finished = pt.logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)

        best_word_indices = best_word_indices.unsqueeze(1)

        # Traced modules do not allow optional return values or None, but lists. We return
        # primary scores and optional factor scores therefore in a list.
        scores = [scores_accumulated]  # type: List[pt.Tensor]
        if self.expect_factors:
            factors, factor_scores_accumulated = factor_args
            # factors: (batch*beam, num_secondary_factors, 2)
            f_sorted = factors.index_select(0, best_hyp_indices)
            factor_scores, factor_indices = f_sorted[:, :, 0], f_sorted[:, :, 1]
            # updated_factor_scores: (batch*beam, num_secondary_factors)
            updated_factor_scores = factor_scores_accumulated.index_select(0, best_hyp_indices) + factor_scores
            # Concatenate sorted secondary target factors to best_word_indices. Shape: (batch*beam, num_factors)
            best_word_indices = pt.cat((best_word_indices, factor_indices.int()), dim=1)
            scores.append(updated_factor_scores)

        return best_word_indices, finished, scores, lengths, reference_lengths


class TopK(pt.nn.Module):
    """
    Batch-wise topk operation.
    Forward method uses imperative shape inference, since both batch_size and vocab_size are dynamic
    during translation (due to variable batch size and potential vocabulary selection).

    NOTE: This module wouldn't support dynamic batch sizes when traced!
    """

    def __init__(self, k: int) -> None:
        """
        :param k: The number of smallest scores to return.
        """
        super().__init__()
        self.k: int = k

    def forward(self, scores: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        Get the lowest k elements per sentence from a `scores` matrix.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :return: The row indices, column indices and values of the k smallest items in matrix.
        """
        batch_times_beam, vocab_size = scores.size()
        batch_size = pt.div(batch_times_beam, self.k, rounding_mode='trunc')
        # Shape: (batch size, beam_size * vocab_size)
        scores = scores.view(batch_size, self.k * vocab_size)

        values, indices = pt.topk(scores, k=self.k, dim=1, largest=False, sorted=True)

        # Project indices back into original shape (which is different for t==1 and t>1)
        values, indices = values.view(-1, 1), indices.view(-1)

        best_hyp_indices, best_word_indices = indices.div(vocab_size, rounding_mode='floor'), indices.fmod(vocab_size)

        return best_hyp_indices, best_word_indices, values


class SampleK(pt.nn.Module):
    """
    A Module for selecting a random word from each hypothesis according to its distribution.
    """
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n: int = n

    def forward(self, scores: pt.Tensor, target_dists: pt.Tensor, finished: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        Choose an extension of each hypothesis from its softmax distribution.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param target_dists: The non-cumulative target distributions (ignored).
        :param finished: The list of finished hypotheses.
        :return: The row indices, column indices, and values of the sampled words.
        """
        # Map the negative logprobs to probabilities so as to have a distribution
        target_dists = pt.exp(-target_dists)

        # n == 0 means sample from the full vocabulary. Otherwise, we sample from the top n.
        if self.n != 0:
            # select the top n in each row, via a mask
            values, indices = pt.topk(target_dists, k=self.n, dim=1, largest=True, sorted=True)
            # set items not chosen by topk to 0
            target_dists = pt.scatter(pt.zeros_like(target_dists), 1, indices, values)
            # renormalize
            target_dists = target_dists / target_dists.sum(1, keepdim=True)

        # Sample from the target distributions over words, then get the corresponding values from the cumulative scores
        # shape: (batch,)
        best_word_indices = pt.multinomial(target_dists, 1).squeeze(1)
        # Zeroes for finished hypotheses.
        best_word_indices = best_word_indices.masked_fill(finished, 0)
        # (batch, 1)
        values = scores.gather(dim=1, index=best_word_indices.long().unsqueeze(1))
        # (batch,)
        best_hyp_indices = pt.arange(0, best_word_indices.size()[0], device=best_word_indices.device)

        return best_hyp_indices, best_word_indices, values


class RepeatStates(pt.nn.Module):

    def __init__(self, beam_size: int, state_structure: List[List[str]]) -> None:
        super().__init__()
        self.beam_size: int = beam_size
        self.flat_structure: List[str] = functools.reduce(operator.add, state_structure)

    def forward(self, *states: pt.Tensor) -> List[pt.Tensor]:
        repeated_states = []
        assert len(states) == len(self.flat_structure), "Number of states do not match the defined state structure"
        for state, state_format in zip(states, self.flat_structure):
            if state_format == C.STEP_STATE or state_format == C.MASK_STATE:
                # Steps and source_bias have batch dimension on axis 0
                repeat_axis = 0
            elif state_format == C.DECODER_STATE or state_format == C.ENCODER_STATE:
                # Decoder and encoder layer states have batch dimension on axis 1
                repeat_axis = 1
            else:
                raise ValueError("Provided state format %s not recognized." % state_format)
            repeated_state = state.repeat_interleave(repeats=self.beam_size, dim=repeat_axis)
            repeated_states.append(repeated_state)
        return repeated_states


class SortStates(pt.nn.Module):

    def __init__(self, state_structure: List[List[str]]):
        super().__init__()
        self.flat_structure: List[str] = functools.reduce(operator.add, state_structure)

    def forward(self, best_hyp_indices: pt.Tensor, *states: pt.Tensor) -> List[pt.Tensor]:
        sorted_states = []
        assert len(states) == len(self.flat_structure), "Number of states do not match the defined state structure"
        for state, state_format in zip(states, self.flat_structure):
            if state_format == C.STEP_STATE:
                # Steps and source_bias have batch dimension on axis 0
                sorted_state = state.index_select(0, best_hyp_indices)
            elif state_format == C.DECODER_STATE:
                # Decoder and encoder layer states have batch dimension on axis 1
                sorted_state = state.index_select(1, best_hyp_indices)
            elif state_format == C.ENCODER_STATE or state_format == C.MASK_STATE:
                # No need for takes on encoder layer states
                sorted_state = state
            else:
                raise ValueError("Provided state format %s not recognized." % state_format)
            sorted_states.append(sorted_state)
        return sorted_states


def _get_vocab_slice_ids(restrict_lexicon: lexicon.RestrictLexicon,
                         source_words: pt.Tensor,
                         eos_id: int,
                         beam_size: int,
                         target_prefix: Optional[pt.Tensor] = None,
                         output_vocab_size: Optional[int] = None) -> Tuple[pt.Tensor, int]:
    device = source_words.device
    if not restrict_lexicon.is_blocking():
        vocab_slice_ids_np = restrict_lexicon.get_allowed_trg_ids(source_words.cpu().int().numpy()) # type: ignore
    else:
        utils.check_condition(output_vocab_size is not None,
                              "output_vocab_size required for blocking restrict lexicon.")
        full_vocab = np.arange(0, output_vocab_size, dtype='int32')
        source_ids = source_words.cpu().int().numpy() if restrict_lexicon.requires_src_ids() else None
        vocab_slice_ids_np = np.setdiff1d(full_vocab, restrict_lexicon.get_blocked_trg_ids(source_ids), assume_unique=True)

    vocab_slice_ids = pt.tensor(vocab_slice_ids_np, device=device, dtype=pt.int64)
    if target_prefix is not None:
        # Ensuring that target prefix ids are part of vocab_slice_ids
        vocab_slice_ids = pt.concat([vocab_slice_ids, target_prefix.flatten().type(pt.int64)], -1).unique()
    # Pad to a multiple of 8.
    vocab_slice_ids = pt.nn.functional.pad(vocab_slice_ids,
                                           pad=(0, 7 - ((vocab_slice_ids.size(-1) - 1) % 8)),
                                           mode='constant', value=eos_id)
    vocab_slice_ids_size = vocab_slice_ids.size()[0]
    if vocab_slice_ids_size < beam_size + 1:
        # This fixes an edge case for toy models, where the number of vocab ids from the lexicon is
        # smaller than the beam size.
        logger.warning("Padding vocab_slice_ids (%d) with EOS to have at least %d+1 elements to expand",
                       vocab_slice_ids_size, beam_size)
        n = beam_size - vocab_slice_ids_size + 1
        vocab_slice_ids = pt.cat((vocab_slice_ids, pt.full((n,),
                                                           fill_value=eos_id,
                                                           device=device,
                                                           dtype=pt.int32)), dim=0)

    logger.debug(f'decoder softmax size: {vocab_slice_ids_size}')
    return vocab_slice_ids, vocab_slice_ids_size


def _get_nvs_vocab_slice_ids(
        nvs_thresh: float,
        nvs_prediction: pt.Tensor,
        restrict_lexicon: Optional[lexicon.RestrictLexicon] = None,
        target_prefix: Optional[pt.Tensor] = None
    ) -> Tuple[pt.Tensor, int]:
    """
    Return the vocab slice ids based on the Neural Vocabulary Selection model's predictions.
    :param nvs_thresh: The threshold for selecting a word (between 0.0 and 1.0).
    :param nvs_prediction: Shape: (batch size, vocab_size).
    :param restrict_lexicon: An optional blocking lexicon to forcefully turn specific words off.
    :param target_prefix: Shape: (batch size, vocab_size).
    """
    nvs_prediction_above_thresh = (nvs_prediction > nvs_thresh)
    # merge batch dimension (batch size, vocab_size) -> (1, vocab_size)
    if nvs_prediction_above_thresh.shape[0] > 1:
        nvs_prediction_above_thresh = pt.any(nvs_prediction_above_thresh, dim=0, keepdim=True)

    if restrict_lexicon is not None:
        utils.check_condition(
            restrict_lexicon.is_blocking() and not restrict_lexicon.requires_src_ids(),
            "Only a blocking, static lexicon is supported when Neural Vocabulary Selection (NVS) is used."
        )
        blocked_tokens = pt.from_numpy(restrict_lexicon.get_blocked_trg_ids()).long().to(nvs_prediction_above_thresh.device)
        nvs_prediction_above_thresh[0, blocked_tokens] = False

    # Add special symbols:
    pt_symbols = pt.tensor([C.PAD_ID, C.UNK_ID, C.BOS_ID, C.EOS_ID], device=nvs_prediction_above_thresh.device)
    nvs_prediction_above_thresh[0, pt_symbols] = True

    if target_prefix is not None:
        nvs_prediction_above_thresh[0, target_prefix.flatten().long()] = True

    bow = nvs_prediction_above_thresh.nonzero(as_tuple=True)[1].unique()

    # pad to a multiple of 8.
    if len(bow) % 8 != 0:
        bow = pt.nn.functional.pad(bow, (0, 7 - ((len(bow) - 1) % 8)), mode='constant', value=C.EOS_ID)

    output_vocab_size = bow.shape[0]
    logger.debug(f'decoder softmax size: {output_vocab_size}')

    return bow, output_vocab_size


class Search(pt.nn.Module):
    def __init__(self,
                 dtype: pt.dtype,
                 bos_id: int,
                 eos_id: int,
                 device: pt.device,
                 num_source_factors: int,
                 num_target_factors: int,
                 skip_nvs: bool = False,
                 nvs_thresh: float = 0.5):
        super().__init__()
        self.dtype: pt.dtype = dtype
        self.bos_id: int = bos_id
        self.eos_id: int = eos_id
        self.device: pt.device = device
        self.num_source_factors: int = num_source_factors
        self.num_target_factors: int = num_target_factors
        self.skip_nvs: bool = skip_nvs
        self.nvs_thresh: float = nvs_thresh

        self.output_vocab_sizes: utils.OnlineMeanAndVariance = utils.OnlineMeanAndVariance()

    def update_output_vocab_size(self, size : Union[float, int]) -> None:
        self.output_vocab_sizes.update(size)

    def log_search_stats(self) -> None:
        logger.debug(f'decoder softmax size: {self.output_vocab_sizes.mean:.1f} (avg)')


class GreedySearch(Search):
    """
    Implements greedy search, not supporting various features from the BeamSearch class
    (scoring, sampling, ensembling, batch decoding).
    """

    def __init__(self,
                 dtype: pt.dtype,
                 bos_id: int,
                 eos_id: int,
                 device: pt.device,
                 num_source_factors: int,
                 num_target_factors: int,
                 inference: _SingleModelInference,
                 skip_nvs: bool = False,
                 nvs_thresh: float = 0.5):
        super().__init__(dtype, bos_id, eos_id, device, num_source_factors,
                         num_target_factors, skip_nvs, nvs_thresh)
        self.output_vocab_size: int = inference.model_output_vocab_size
        self.output_factor_vocab_size: int = inference.model_output_factor_vocab_size
        self._inference: _SingleModelInference = inference
        assert inference._skip_softmax, "skipping softmax must be enabled for GreedySearch"
        self.work_block: GreedyTop1 = GreedyTop1()

    def forward(self,
                source: pt.Tensor,
                source_length: pt.Tensor,
                restrict_lexicon: Optional[lexicon.RestrictLexicon] = None,
                max_output_lengths: pt.T