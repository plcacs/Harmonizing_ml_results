from typing import Dict, List, Optional, Set, Callable, Tuple, Union
from collections import defaultdict
import torch
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans, bioul_tags_to_spans, iob1_tags_to_spans, bmes_tags_to_spans, TypedStringSpan

TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]

@Metric.register('span_f1')
class SpanBasedF1Measure(Metric):
    def __init__(
        self,
        vocabulary: Vocabulary,
        tag_namespace: str = 'tags',
        ignore_classes: Optional[List[str]] = None,
        label_encoding: Optional[str] = 'BIO',
        tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None
    ) -> None:
        if label_encoding and tags_to_spans_function:
            raise ConfigurationError('Both label_encoding and tags_to_spans_function are provided. Set "label_encoding=None" explicitly to enable tags_to_spans_function.')
        if label_encoding:
            if label_encoding not in ['BIO', 'IOB1', 'BIOUL', 'BMES']:
                raise ConfigurationError("Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'.")
        elif tags_to_spans_function is None:
            raise ConfigurationError('At least one of the (label_encoding, tags_to_spans_function) should be provided.')
        self._label_encoding: Optional[str] = label_encoding
        self._tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = tags_to_spans_function
        self._label_vocabulary: Dict[int, str] = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        prediction_map: Optional[torch.Tensor] = None
    ) -> None:
        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        predictions, gold_labels, mask, prediction_map = self.detach_tensors(predictions, gold_labels, mask, prediction_map)
        num_classes: int = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(f'A gold label passed to SpanBasedF1Measure contains an id >= {num_classes}, the number of classes.')
        sequence_lengths: torch.Tensor = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions: torch.Tensor = predictions.max(-1)[1]
        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())
        argmax_predictions = argmax_predictions.float()
        batch_size: int = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction: torch.Tensor = argmax_predictions[i, :]
            sequence_gold_label: torch.Tensor = gold_labels[i, :]
            length: int = sequence_lengths[i]
            if length == 0:
                continue
            predicted_string_labels: List[str] = [self._label_vocabulary[label_id] for label_id in sequence_prediction[:length].tolist()]
            gold_string_labels: List[str] = [self._label_vocabulary[label_id] for label_id in sequence_gold_label[:length].tolist()]
            if self._label_encoding is None and self._tags_to_spans_function:
                tags_to_spans_function: TAGS_TO_SPANS_FUNCTION_TYPE = self._tags_to_spans_function
            elif self._label_encoding == 'BIO':
                tags_to_spans_function = bio_tags_to_spans
            elif self._label_encoding == 'IOB1':
                tags_to_spans_function = iob1_tags_to_spans
            elif self._label_encoding == 'BIOUL':
                tags_to_spans_function = bioul_tags_to_spans
            elif self._label_encoding == 'BMES':
                tags_to_spans_function = bmes_tags_to_spans
            else:
                raise ValueError(f"Unexpected label encoding scheme '{self._label_encoding}'")
            predicted_spans: List[TypedStringSpan] = tags_to_spans_function(predicted_string_labels, self._ignore_classes)
            gold_spans: List[TypedStringSpan] = tags_to_spans_function(gold_string_labels, self._ignore_classes)
            predicted_spans = self._handle_continued_spans(predicted_spans)
            gold_spans = self._handle_continued_spans(gold_spans)
            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    @staticmethod
    def _handle_continued_spans(spans: List[TypedStringSpan]) -> List[TypedStringSpan]:
        span_set: Set[TypedStringSpan] = set(spans)
        continued_labels: List[str] = [label[2:] for label, span in span_set if label.startswith('C-')]
        for label in continued_labels:
            continued_spans: Set[TypedStringSpan] = {span for span in span_set if label in span[0]}
            span_start: int = min((span[1][0] for span in continued_spans))
            span_end: int = max((span[1][1] for span in continued_spans))
            replacement_span: TypedStringSpan = (label, (span_start, span_end))
            span_set.difference_update(continued_spans)
            span_set.add(replacement_span)
        return list(span_set)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if is_distributed():
            raise RuntimeError('Distributed aggregation for SpanBasedF1Measure is currently not supported.')
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics: Dict[str, float] = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag])
            precision_key: str = 'precision' + '-' + tag
            recall_key: str = 'recall' + '-' + tag
            f1_key: str = 'f1-measure' + '-' + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()), sum(self._false_positives.values()), sum(self._false_negatives.values()))
        all_metrics['precision-overall'] = precision
        all_metrics['recall-overall'] = recall
        all_metrics['f1-measure-overall'] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int) -> Tuple[float, float, float]:
        precision: float = true_positives / (true_positives + false_positives + 1e-13)
        recall: float = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure: float = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return (precision, recall, f1_measure)

    def reset(self) -> None:
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
