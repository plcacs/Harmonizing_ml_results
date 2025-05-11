import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch as pt
import numpy as np
from . import constants as C
from . import utils

logger = logging.getLogger(__name__)

class Loss(pt.nn.Module):
    """
    Generic Loss interface.
    A loss has a name, a configuration, and stores information about the output and label it requires from the model(s),
    as well as a weight (default 1.0) and a method to create the corresponding metric.
    """

    def __init__(
        self,
        name: str,
        output_name: str,
        label_name: str,
        weight: float = 1.0,
        metric_prefix: str = ''
    ) -> None:
        super().__init__()
        self._name: str = name
        self._output_name: str = output_name
        self._label_name: str = label_name
        self._weight: float = weight
        self._metric: Optional['LossMetric'] = None
        self._metric_prefix: str = metric_prefix
        logger.info(
            "Loss: %s | weight=%.2f | metric: %s (%s) | output_name: '%s' | label_name: '%s'",
            self._name,
            self.weight,
            self.metric.name,
            self.metric.short_name,
            self.output_name,
            self.label_name
        )

    def __call__(self, outputs: Dict[str, pt.Tensor], labels: Dict[str, pt.Tensor]) -> Any:
        """
        Loss retrieves the required output and label.
        """
        utils.check_condition(
            self.output_name in outputs,
            f"output '{self.output_name}' not found. Loss requires this output key"
        )
        utils.check_condition(
            self.label_name in labels,
            f"label '{self.label_name}' not found. Loss requires this label key"
        )
        output: pt.Tensor = outputs[self.output_name]
        label: pt.Tensor = labels[self.label_name]
        return super().__call__(output, label)

    @abstractmethod
    def create_metric(self) -> 'LossMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        raise NotImplementedError()

    @property
    def metric(self) -> 'LossMetric':
        if self._metric is None:
            self._metric = self.create_metric()
        return self._metric

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_name(self) -> str:
        return self._output_name

    @property
    def label_name(self) -> str:
        return self._label_name

class LossMetric(ABC):

    def __init__(
        self,
        name: str,
        short_name: Optional[str] = None,
        prefix: str = ''
    ) -> None:
        self._name: str = prefix + name
        self._short_name: str = prefix + short_name if short_name else self._name
        self._sum: float = 0.0
        self._num_inst: float = 0.0

    def __repr__(self) -> str:
        return f'{self.name}({self._sum:.2f}/{self._num_inst:.2f}={self.get():.2f})'

    def __str__(self) -> str:
        return f'{self.short_name}={self.get()}'

    @property
    def name(self) -> str:
        return self._name

    @property
    def short_name(self) -> str:
        return self._short_name

    def update(self, loss: float, num_samples: float) -> None:
        self._sum += loss
        self._num_inst += num_samples

    def get(self) -> float:
        return self._sum / self._num_inst if self._num_inst else float('nan')

    def reset(self) -> None:
        self._sum = 0.0
        self._num_inst = 0.0

class CrossEntropyLoss(Loss):
    """
    Computes a cross-entropy loss, normalized by the number of valid (non-pad) tokens.
    Uses an efficient implementation for label smoothing and avoids the obscure SoftmaxOutput op.
    """

    def __init__(
        self,
        name: str = C.CROSS_ENTROPY,
        weight: float = 1.0,
        label_smoothing: float = 0.0,
        dtype: Any = C.DTYPE_FP32,
        output_name: str = C.LOGITS_NAME,
        label_name: str = C.TARGET_LABEL_NAME,
        ignore_label: int = C.PAD_ID,
        metric_prefix: str = '',
        label_smoothing_impl: str = 'mxnet'
    ) -> None:
        super().__init__(
            name=name,
            output_name=output_name,
            label_name=label_name,
            weight=weight,
            metric_prefix=metric_prefix
        )
        self.ignore_label: int = ignore_label
        self._alpha: float = label_smoothing
        self._dtype: Any = dtype
        self._reduction: str = 'mean'
        if label_smoothing == 0 or label_smoothing_impl == 'torch':
            self._ce_impl = self._torch_cross_entropy_loss
        elif label_smoothing > 0.0 and label_smoothing_impl == 'mxnet':
            self._ce_impl = self._smoothed_loss_as_in_mxnet
        elif label_smoothing > 0.0 and label_smoothing_impl == 'fairseq':
            self._ce_impl = self._smoothed_loss_as_in_fairseq
        else:
            raise ValueError('unknown label_smoothing impl. choose from mxnet, fairseq, or torch.')

    def _smoothed_loss_as_in_mxnet(self, logits: pt.Tensor, labels: pt.Tensor) -> pt.Tensor:
        """
        Computes label-smoothed cross-entropy loss just like sockeye.loss.CrossEntropyLossWithoutSoftmaxOutput()
        Notable details:
        - smoothing with 1/vocab_size, not 1/(vocab_size-1) as in fairseq
        - form taken from https://github.com/dmlc/gluon-nlp/blob/b714eaccc67619d7bdcbd1574d30be87d9c73f0c/src/gluonnlp/loss.py#L4
        """
        pred: pt.Tensor = pt.nn.functional.log_softmax(logits, dim=-1)
        nll: pt.Tensor = -pred.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)
        all_scores: pt.Tensor = pred.sum(dim=-1)
        valid_mask: pt.Tensor = labels.not_equal(self.ignore_label)
        pad_mask: pt.Tensor = ~valid_mask
        nll.masked_fill_(pad_mask, 0.0)
        all_scores.masked_fill_(pad_mask, 0.0)
        nll = (1 - self._alpha) * nll - self._alpha / logits.size(-1) * all_scores
        num_valid: pt.Tensor = valid_mask.sum()
        ce: pt.Tensor = nll.sum() * self.weight / num_valid
        return ce

    def _smoothed_loss_as_in_fairseq(self, logits: pt.Tensor, labels: pt.Tensor) -> pt.Tensor:
        """
        Computes smoothed NLL as in fairseq, see
        # https://github.com/pytorch/fairseq/blob/db0175a882e8ae0f30d89b5a610373dbe032d528/fairseq/criterions/label_smoothed_cross_entropy.py#L33
        """
        pred: pt.Tensor = pt.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == logits.dim() - 1:
            labels = labels.unsqueeze(-1)
        nll: pt.Tensor = -pred.gather(dim=-1, index=labels.long())
        smooth_loss: pt.Tensor = pred.sum(dim=-1, keepdim=True)
        pad_mask: pt.Tensor = labels.eq(self.ignore_label)
        nll.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        nll = nll.sum()
        smooth_loss = smooth_loss.sum()
        alpha_i: float = self._alpha / (logits.size(-1) - 1)
        nll = (1.0 - self._alpha - alpha_i) * nll - alpha_i * smooth_loss
        num_valid: pt.Tensor = (~pad_mask).sum()
        ce: pt.Tensor = nll.sum() * self.weight / num_valid
        return ce

    def _torch_cross_entropy_loss(self, logits: pt.Tensor, labels: pt.Tensor) -> pt.Tensor:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.reshape(-1)
        _kwargs: Dict[str, Any] = {
            'weight': None,
            'ignore_index': self.ignore_label,
            'reduction': self._reduction
        }
        if self._alpha > 0.0:
            _kwargs['label_smoothing'] = self._alpha
        ce: pt.Tensor = pt.nn.functional.cross_entropy(logits, labels.long(), **_kwargs)
        ce *= self.weight
        return ce

    def forward(self, logits: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        ce: pt.Tensor = self._ce_impl(logits, labels)
        return (ce, pt.ones(1, device=ce.device))

    def create_metric(self) -> 'PerplexityMetric':
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        return PerplexityMetric(prefix=self._metric_prefix)

class DynamicBCEWithLogitsLoss(pt.nn.BCEWithLogitsLoss):
    """ A version of BCEWithLogitsLoss where the pos_weight can be supplied dynamically in the `forward` call. """

    def __init__(
        self,
        weight: Optional[pt.Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = 'mean',
        pos_weight: Optional[pt.Tensor] = None
    ) -> None:
        super().__init__(reduction=reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(
        self,
        input: pt.Tensor,
        target: pt.Tensor,
        pos_weight: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        if pos_weight is None:
            pos_weight = self.pos_weight
        return pt.nn.functional.binary_cross_entropy_with_logits(
            input,
            target,
            self.weight,
            pos_weight=pos_weight,
            reduction=self.reduction
        )

@pt.jit.script
def _label_to_bow(label: pt.Tensor, num_labels: int) -> pt.Tensor:
    bow: pt.Tensor = pt.zeros(label.shape[0], num_labels, device=label.device)
    bow[pt.arange(0, label.shape[0], dtype=pt.int64)[:, np.newaxis], label.long()] = 1.0
    return bow

class BinaryCrossEntropyBowLoss(Loss):
    """
    Computes the binary cross entropy loss over a bag-of-words of target tokens.
    """

    def __init__(
        self,
        name: str = C.BINARY_CROSS_ENTROPY,
        pos_weight: float = 1.0,
        weight: float = 1.0,
        dtype: Any = C.DTYPE_FP32,
        output_name: str = C.NVS_PRED_NAME,
        label_name: str = C.TARGET_LABEL_NAME,
        num_labels: int = 0,
        metric_prefix: str = ''
    ) -> None:
        super().__init__(
            name=name,
            output_name=output_name,
            label_name=label_name,
            weight=weight,
            metric_prefix=metric_prefix
        )
        self._dtype: Any = dtype
        assert num_labels != 0, 'num_labels required'
        self._num_labels: int = num_labels
        self.ce_loss: DynamicBCEWithLogitsLoss = DynamicBCEWithLogitsLoss(reduction='none')
        self.pos_weight: float = pos_weight

    def forward(self, output: pt.Tensor, label: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        pred: (batch_size, num_vocab) probabilities.
        labels: (batch_size, target_length) words.
        """
        nvs_pred: pt.Tensor = output
        bow: pt.Tensor = _label_to_bow(label, self._num_labels)
        num_positive: pt.Tensor = pt.sum(bow).float()
        num_total: int = bow.shape[0] * bow.shape[1]
        num_negative: pt.Tensor = pt.tensor(num_total) - num_positive
        pos_weight: float = self.pos_weight * num_negative.item() / num_positive.item()
        avg_pos_count: pt.Tensor = pt.mean(pt.sum(bow, dim=1).float())
        implied_pos_count: float = avg_pos_count.item() * (pos_weight - 1)
        scale: float = 1.0 / (self._num_labels + implied_pos_count)
        loss: pt.Tensor = self.ce_loss(nvs_pred, bow, pt.tensor(pos_weight, device=bow.device))
        loss = pt.sum(loss, dim=1) * scale
        ce: pt.Tensor = pt.mean(loss) * self.weight
        return (ce, pt.ones(1, device=ce.device))

    def create_metric(self) -> 'PerplexityMetric':
        return PerplexityMetric(prefix=self._metric_prefix)

class PerplexityMetric(LossMetric):

    def __init__(
        self,
        prefix: str = '',
        name: str = C.PERPLEXITY,
        short_name: str = C.PERPLEXITY_SHORT_NAME
    ) -> None:
        super().__init__(name=name, short_name=short_name, prefix=prefix)

    def update(self, batch_cross_entropy: float, batch_num_valid: float) -> None:
        self._sum += batch_cross_entropy
        self._num_inst += batch_num_valid

    def get(self) -> float:
        return math.exp(super().get())

class PoissonLoss(Loss):
    """
    Computes the Poisson regression loss.
    MSEMetric for this loss will be reporting the mean
    square error between lengths, not length ratios!
    """

    def __init__(
        self,
        name: str = f'{C.LENRATIO_NAME}_{C.LINK_POISSON}',
        weight: float = 1.0,
        output_name: str = C.LENRATIO_NAME,
        label_name: str = C.LENRATIO_LABEL_NAME
    ) -> None:
        super().__init__(
            name=name,
            output_name=output_name,
            label_name=label_name,
            weight=weight
        )

    def forward(self, length_predictions: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Returns Poisson loss and output given data and expected integers as labels.

        :param length_predictions: Length predictions. Shape: (batch_size,).
        :param labels: Targets. Shape: (batch_size,).
        :return: Poisson loss of length predictions of the batch, and number of samples (batch size).
        """
        loss: pt.Tensor = length_predictions - labels * pt.log(pt.clamp(length_predictions, min=1e-10))
        loss = (loss * self.weight).sum()
        num_samples: pt.Tensor = pt.ones_like(length_predictions).sum()
        return (loss, num_samples)

    def create_metric(self) -> 'LossMetric':
        return LossMetric(name=C.LENRATIO_MSE)

class MSELoss(Loss):
    """
    Computes the Mean Squared Error loss.
    MSEMetric for this loss will be reporting the mean square error between length ratios.
    """

    def __init__(
        self,
        name: str = C.LENRATIO_NAME + '_' + C.LINK_NORMAL,
        weight: float = 1.0,
        output_name: str = C.LENRATIO_NAME,
        label_name: str = C.LENRATIO_LABEL_NAME
    ) -> None:
        super().__init__(
            name=name,
            output_name=output_name,
            label_name=label_name,
            weight=weight
        )

    def forward(self, length_predictions: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        Returns MSE loss.

        :param length_predictions: Length predictions. Shape: (batch_size,).
        :param labels: Targets. Shape: (batch_size,).
        :return: MSE loss of length predictions of the batch.
        """
        loss: pt.Tensor = self.weight / 2 * pt.square(length_predictions - labels)
        loss = loss.sum()
        num_samples: pt.Tensor = pt.ones_like(length_predictions).sum()
        return (loss, num_samples)

    def create_metric(self) -> 'LossMetric':
        return LossMetric(name=C.LENRATIO_MSE)
