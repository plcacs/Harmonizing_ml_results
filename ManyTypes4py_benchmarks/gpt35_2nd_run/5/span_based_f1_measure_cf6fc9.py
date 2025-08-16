from typing import Dict, List, Optional, Set, Callable
import torch
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans, bioul_tags_to_spans, iob1_tags_to_spans, bmes_tags_to_spans, TypedStringSpan

TAGS_TO_SPANS_FUNCTION_TYPE: Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]] = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]

@Metric.register('span_f1')
class SpanBasedF1Measure(Metric):
    def __init__(self, vocabulary: Vocabulary, tag_namespace: str = 'tags', ignore_classes: Optional[List[str]] = None, label_encoding: str = 'BIO', tags_to_spans_function: Optional[Callable] = None) -> None:
    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None, prediction_map: Optional[torch.Tensor] = None) -> None:
    @staticmethod
    def _handle_continued_spans(spans: List[TypedStringSpan]) -> List[TypedStringSpan]:
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int) -> Tuple[float, float, float]:
    def reset(self) -> None:
