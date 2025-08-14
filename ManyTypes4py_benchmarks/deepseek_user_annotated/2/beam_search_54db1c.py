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
from typing import Optional, Tuple, List, Union, Any, Dict, Callable

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
                    factor_vocab_size: Optional[int] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], List[pt.Tensor]]:
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
        self._model = model
        self._skip_softmax = skip_softmax
        self._const_lr = constant_length_ratio
        self.knn_lambda = knn_lambda

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
                    factor_vocab_size: Optional[int] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], List[pt.Tensor]]:
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
        self._models = models
        if ensemble_mode == 'linear':
            self._interpolation = self.linear_interpolation
        elif ensemble_mode == 'log_linear':
            self._interpolation = self.log_linear_interpolation
        else:
            raise ValueError()
        self._const_lr = constant_length_ratio
        self.knn_lambda = knn_lambda

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
                    factor_vocab_size: Optional[int] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], List[pt.Tensor]]:
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
        self.prevent_unk = prevent_unk
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
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.) ** self.alpha

    def forward(self, lengths: Union[pt.Tensor, int]) -> Union[pt.Tensor, float]:
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
        self.weight = weight

    def forward(self, hyp_lengths: Union[pt.Tensor, int], reference_lengths: Union[pt.Tensor, int]) -> Union[pt.Tensor, float]:
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
        self._lp = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)
        self._bp = None  # type: Optional[BrevityPenalty]
        if brevity_penalty_weight > 0.0:
            self._bp = BrevityPenalty(weight=brevity_penalty_weight)

    def forward(self, scores: Union[pt.Tensor, float], lengths: Union[pt.Tensor, int], reference_lengths: Union[pt.Tensor, int]) -> Union[pt.Tensor, float]:
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

    def unnormalize(self, scores: Union[pt.Tensor, float], lengths: Union[pt.Tensor, int], reference_lengths: Union[pt.Tensor, int]) -> Union[pt.Tensor, float]:
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
        self.pad_id = pad_id
        self.eos_id = eos_id
        self