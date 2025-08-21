"""
Code for scoring.
"""
import logging
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch as pt

from . import constants as C
from . import data_io
from . import inference
from . import vocab
from .beam_search import CandidateScorer
from .model import SockeyeModel
from .output_handler import OutputHandler

logger: logging.Logger = logging.getLogger(__name__)


class BatchScorer(pt.nn.Module):

    def __init__(
        self,
        scorer: CandidateScorer,
        score_type: int = C.SCORING_TYPE_DEFAULT,
        constant_length_ratio: Optional[float] = None,
        softmax_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.score_type: int = score_type
        self.scorer: CandidateScorer = scorer
        self.constant_length_ratio: Optional[float] = constant_length_ratio
        assert softmax_temperature is None, 'not implemented'

    def forward(
        self,
        logits: pt.Tensor,
        labels: pt.Tensor,
        length_ratio: pt.Tensor,
        source_length: pt.Tensor,
        target_length: pt.Tensor,
        factor_logits_and_labels: Optional[List[Tuple[pt.Tensor, pt.Tensor]]] = None,
    ) -> pt.Tensor:
        """
        :param logits: Model logits for primary output words. Shape: (batch, length, vocab_size).
        :param labels: Gold targets. Shape: (batch, length).
        :param length_ratio: Length Ratios. Shape: (batch,).
        :param source_length: Source lengths. Shape: (batch,).
        :param target_length: Target lengths. Shape: (batch,).
        :param factor_logits_and_labels: List of target factor logits and corresponding labels.
               Shape: (batch, length, factor_vocab_size).
        :return: Sequence scores. Shape: (batch,).
        """
        logprobs = logits.log_softmax(dim=-1)
        token_scores = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        if self.score_type == C.SCORING_TYPE_NEGLOGPROB:
            token_scores = -token_scores
        scores = token_scores.masked_fill_(labels == C.PAD_ID, 0.0).sum(dim=-1, keepdims=True)
        if self.constant_length_ratio is not None and self.constant_length_ratio > 0.0:
            predicted_output_length = source_length * self.constant_length_ratio
        else:
            predicted_output_length = source_length * length_ratio
        scores = self.scorer(scores, target_length, predicted_output_length)
        if factor_logits_and_labels is not None:
            factor_scores: List[pt.Tensor] = []
            for factor_logit, factor_label in factor_logits_and_labels:
                factor_logprobs = factor_logit.log_softmax(dim=-1)
                factor_token_scores = factor_logprobs.gather(dim=-1, index=factor_label.unsqueeze(-1)).squeeze(-1)
                if self.score_type == C.SCORING_TYPE_NEGLOGPROB:
                    factor_token_scores = -factor_token_scores
                fs = factor_token_scores.masked_fill_(factor_label == C.PAD_ID, 0.0).sum(dim=-1, keepdims=True)
                factor_scores.append(fs)
            scores = pt.cat([scores] + factor_scores, dim=1)
        return scores


class Scorer:
    """
    Scorer class takes a ScoringModel and uses it to score a stream of parallel sentences.
    It also takes the vocabularies so that the original sentences can be printed out, if desired.

    :param model: The model to score with.
    :param batch_scorer: BatchScorer block to score each batch.
    :param source_vocabs: The source vocabularies.
    :param target_vocabs: The target vocabularies.
    :param device: Torch device to load batches to (should be set to model device).
    """

    def __init__(
        self,
        model: SockeyeModel,
        batch_scorer: BatchScorer,
        source_vocabs: List[Dict[str, int]],
        target_vocabs: List[Dict[str, int]],
        device: pt.device,
    ) -> None:
        self.source_vocab_inv: Dict[int, str] = vocab.reverse_vocab(source_vocabs[0])
        self.target_vocab_inv: Dict[int, str] = vocab.reverse_vocab(target_vocabs[0])
        self.model: SockeyeModel = model
        self.traced_model: Optional[pt.jit.ScriptModule] = None
        self.batch_scorer: BatchScorer = batch_scorer
        self.traced_batch_scorer: Optional[pt.jit.ScriptModule] = None
        self.device: pt.device = device
        self.exclude_list: Set[int] = {C.BOS_ID, C.EOS_ID, C.PAD_ID}
        self.num_target_factors: int = self.model.num_target_factors

    def score_batch(self, batch: Any) -> np.ndarray:
        batch = batch.load(self.device)
        model_inputs: Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor] = (
            batch.source,
            batch.source_length,
            batch.target,
            batch.target_length,
        )
        if self.traced_model is None:
            self.traced_model = pt.jit.trace(self.model, model_inputs, strict=False)
        outputs: Dict[str, pt.Tensor] = self.traced_model(*model_inputs)
        scorer_inputs: List[Any] = [
            outputs[C.LOGITS_NAME],
            batch.labels[C.TARGET_LABEL_NAME].long(),
            outputs.get(C.LENRATIO_NAME, pt.zeros_like(batch.source_length[:, 0])),
            batch.source_length[:, 0],
            batch.target_length,
        ]
        if self.num_target_factors > 1:
            factor_logits_and_labels: List[Tuple[pt.Tensor, pt.Tensor]] = [
                (outputs[C.FACTOR_LOGITS_NAME % i], batch.labels[C.TARGET_FACTOR_LABEL_NAME % i].long())
                for i in range(1, self.num_target_factors)
            ]
            scorer_inputs.append(factor_logits_and_labels)
        if self.traced_batch_scorer is None:
            logger.debug('Tracing batch_scorer')
            self.traced_batch_scorer = pt.jit.trace(self.batch_scorer, scorer_inputs, strict=False)
        scores: pt.Tensor = self.traced_batch_scorer(*scorer_inputs)
        scores_cpu: pt.Tensor = scores.cpu()
        if self.model.dtype == pt.bfloat16:
            scores_cpu = scores_cpu.to(dtype=pt.float32)
        return scores_cpu.numpy()

    @pt.inference_mode(True)
    def score(self, score_iter: Iterable[Any], output_handler: OutputHandler) -> None:
        total_time: float = 0.0
        sentence_no: int = 0
        batch_no: int = 0
        for batch_no, batch in enumerate(score_iter, 1):
            batch_tic = time.time()
            batch_scores = self.score_batch(batch)
            batch_time = time.time() - batch_tic
            total_time += batch_time
            for sentno, (source, target, scores) in enumerate(
                zip(batch.source[:, :, 0], batch.target[:, :, 0], batch_scores), 1
            ):
                sentence_no += 1
                source_ids: List[int] = source.tolist()
                source_tokens: List[str] = list(data_io.ids2tokens(source_ids, self.source_vocab_inv, self.exclude_list))
                target_ids: List[int] = target.tolist()
                target_tokens: List[str] = list(data_io.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))
                target_string: str = C.TOKEN_SEPARATOR.join(target_tokens)
                if source[0] == C.PAD_ID or target[0] == C.PAD_ID:
                    scores = [-np.inf] * self.num_target_factors
                output_handler.handle(
                    inference.TranslatorInput(sentence_no, source_tokens),
                    inference.TranslatorOutput(sentence_no, target_string, target_tokens, score=scores[0], factor_scores=scores[1:]),
                    batch_time,
                )
        if sentence_no != 0:
            logger.info(
                'Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f',
                sentence_no,
                math.ceil(sentence_no / batch_no),
                total_time,
                total_time / sentence_no,
                sentence_no / total_time,
            )
        else:
            logger.info('Processed 0 lines.')