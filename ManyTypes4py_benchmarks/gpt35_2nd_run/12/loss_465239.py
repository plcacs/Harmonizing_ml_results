import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch as pt
import numpy as np
from . import constants as C
from . import utils

logger: logging.Logger = logging.getLogger(__name__)

class Loss(pt.nn.Module):
    def __init__(self, name: str, output_name: str, label_name: str, weight: float = 1.0, metric_prefix: str = ''):
        super().__init__()
        self._name: str = name
        self._output_name: str = output_name
        self._label_name: str = label_name
        self._weight: float = weight
        self._metric: Optional[LossMetric] = None
        self._metric_prefix: str = metric_prefix
        logger.info(f"Loss: {self._name} | weight={self.weight:.2f} | metric: {self.metric.name} ({self.metric.short_name}) | output_name: '{self.output_name}' | label_name: '{self.label_name}'")

    def __call__(self, outputs: Dict[str, Any], labels: Dict[str, Any]) -> Any:
        utils.check_condition(self.output_name in outputs, f"output '{self.output_name}' not found. Loss requires this output key")
        utils.check_condition(self.label_name in labels, f"label '{self.label_name}' not found. Loss requires this label key")
        output = outputs[self.output_name]
        label = labels[self.label_name]
        return super().__call__(output, label)

    @abstractmethod
    def create_metric(self) -> LossMetric:
        raise NotImplementedError()

    @property
    def metric(self) -> LossMetric:
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
    def __init__(self, name: str, short_name: Optional[str] = None, prefix: str = ''):
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
    def __init__(self, name: str = C.CROSS_ENTROPY, weight: float = 1.0, label_smoothing: float = 0.0, dtype: str = C.DTYPE_FP32, output_name: str = C.LOGITS_NAME, label_name: str = C.TARGET_LABEL_NAME, ignore_label: int = C.PAD_ID, metric_prefix: str = '', label_smoothing_impl: str = 'mxnet'):
        super().__init__(name=name, output_name=output_name, label_name=label_name, weight=weight, metric_prefix=metric_prefix)
        self.ignore_label: int = ignore_label
        self._alpha: float = label_smoothing
        self._dtype: str = dtype
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
        ...

    def _smoothed_loss_as_in_fairseq(self, logits: pt.Tensor, labels: pt.Tensor) -> pt.Tensor:
        ...

    def _torch_cross_entropy_loss(self, logits: pt.Tensor, labels: pt.Tensor) -> pt.Tensor:
        ...

    def forward(self, logits: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        ...

    def create_metric(self) -> PerplexityMetric:
        ...

class DynamicBCEWithLogitsLoss(pt.nn.BCEWithLogitsLoss):
    def __init__(self, weight: Optional[pt.Tensor] = None, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = 'mean', pos_weight: Optional[pt.Tensor] = None):
        ...

    def forward(self, input: pt.Tensor, target: pt.Tensor, pos_weight: Optional[pt.Tensor] = None) -> pt.Tensor:
        ...

@pt.jit.script
def _label_to_bow(label: pt.Tensor, num_labels: int) -> pt.Tensor:
    ...

class BinaryCrossEntropyBowLoss(Loss):
    def __init__(self, name: str = C.BINARY_CROSS_ENTROPY, pos_weight: float = 1.0, weight: float = 1.0, dtype: str = C.DTYPE_FP32, output_name: str = C.NVS_PRED_NAME, label_name: str = C.TARGET_LABEL_NAME, num_labels: int = 0, metric_prefix: str = ''):
        ...

    def forward(self, output: pt.Tensor, label: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        ...

    def create_metric(self) -> PerplexityMetric:
        ...

class PerplexityMetric(LossMetric):
    def __init__(self, prefix: str = '', name: str = C.PERPLEXITY, short_name: str = C.PERPLEXITY_SHORT_NAME):
        ...

    def update(self, batch_cross_entropy: float, batch_num_valid: float) -> None:
        ...

    def get(self) -> float:
        ...

class PoissonLoss(Loss):
    def __init__(self, name: str = f'{C.LENRATIO_NAME}_{C.LINK_POISSON}', weight: float = 1.0, output_name: str = C.LENRATIO_NAME, label_name: str = C.LENRATIO_LABEL_NAME):
        ...

    def forward(self, length_predictions: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        ...

    def create_metric(self) -> LossMetric:
        ...

class MSELoss(Loss):
    def __init__(self, name: str = C.LENRATIO_NAME + '_' + C.LINK_NORMAL, weight: float = 1.0, output_name: str = C.LENRATIO_NAME, label_name: str = C.LENRATIO_LABEL_NAME):
        ...

    def forward(self, length_predictions: pt.Tensor, labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        ...

    def create_metric(self) -> LossMetric:
        ...
