from typing import Dict, List, Any
import pytest
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, global_distributed_metric, multi_device, run_distributed_test
from sklearn.metrics import precision_recall_fscore_support
from torch.testing import assert_allclose
from allennlp.training.metrics import FBetaMultiLabelMeasure


class FBetaMultiLabelMeasureTest(AllenNlpTestCase):

    def setup_method(self) -> None:
        super().setup_method()
        self.predictions: torch.Tensor = torch.tensor([[0.55, 0.25, 0.1, 0.1, 0.2],
                                                        [0.1, 0.6, 0.1, 0.95, 0.0],
                                                        [0.9, 0.8, 0.75, 0.8, 0.0],
                                                        [0.49, 0.5, 0.95, 0.55, 0.0],
                                                        [0.6, 0.49, 0.6, 0.65, 0.85],
                                                        [0.85, 0.4, 0.1, 0.2, 0.0]])
        self.targets: torch.Tensor = torch.tensor([[1, 1, 0, 0, 0],
                                                    [0, 1, 0, 1, 0],
                                                    [1, 1, 0, 1, 0],
                                                    [1, 1, 1, 1, 0],
                                                    [1, 1, 1, 1, 0],
                                                    [0, 0, 0, 0, 0]])
        self.pred_sum: List[int] = [4, 3, 3, 4, 1]
        self.true_sum: List[int] = [4, 5, 2, 4, 0]
        self.true_positive_sum: List[int] = [3, 3, 2, 4, 0]
        self.true_negative_sum: List[int] = [1, 1, 3, 2, 5]
        self.total_sum: List[int] = [30, 30, 30, 30, 30]
        desired_precisions: List[float] = [3 / 4, 3 / 3, 2 / 3, 4 / 4, 0 / 1]
        desired_recalls: List[float] = [3 / 4, 3 / 5, 2 / 2, 4 / 4, 0.0]
        desired_fscores: List[float] = [2 * p * r / (p + r) if p + r != 0.0 else 0.0 for p, r in zip(desired_precisions, desired_recalls)]
        self.desired_precisions: List[float] = desired_precisions
        self.desired_recalls: List[float] = desired_recalls
        self.desired_fscores: List[float] = desired_fscores

    @multi_device
    def test_config_errors(self, device: torch.device) -> None:
        pytest.raises(ConfigurationError, FBetaMultiLabelMeasure, beta=0.0)
        pytest.raises(ConfigurationError, FBetaMultiLabelMeasure, average='mega')
        pytest.raises(ConfigurationError, FBetaMultiLabelMeasure, labels=[])

    @multi_device
    def test_runtime_errors(self, device: torch.device) -> None:
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        pytest.raises(RuntimeError, fbeta.get_metric)

    @multi_device
    def test_fbeta_multilabel_state(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(self.predictions, self.targets)
        assert_allclose(fbeta._pred_sum.tolist(), self.pred_sum)
        assert_allclose(fbeta._true_sum.tolist(), self.true_sum)
        assert_allclose(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        assert_allclose(fbeta._true_negative_sum.tolist(), self.true_negative_sum)
        assert_allclose(fbeta._total_sum.tolist(), self.total_sum)

    @multi_device
    def test_fbeta_multilabel_metric(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(self.predictions, self.targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    @multi_device
    def test_fbeta_multilable_with_extra_dimensions(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(self.predictions.unsqueeze(1), self.targets.unsqueeze(1))
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    @multi_device
    def test_fbeta_multilabel_with_mask(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        mask: torch.Tensor = torch.tensor([True, True, True, True, True, False], device=device).unsqueeze(-1)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(self.predictions, self.targets, mask)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        assert_allclose(fbeta._pred_sum.tolist(), [3, 3, 3, 4, 1])
        assert_allclose(fbeta._true_sum.tolist(), [4, 5, 2, 4, 0])
        assert_allclose(fbeta._true_positive_sum.tolist(), [3, 3, 2, 4, 0])
        desired_precisions: List[float] = [3 / 3, 3 / 3, 2 / 3, 4 / 4, 0 / 1]
        desired_recalls: List[float] = [3 / 4, 3 / 5, 2 / 2, 4 / 4, 0.0]
        desired_fscores: List[float] = [2 * p * r / (p + r) if p + r != 0.0 else 0.0 for p, r in zip(desired_precisions, desired_recalls)]
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multilabel_macro_average_metric(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure(average='macro')
        fbeta(self.predictions, self.targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: float = metric['precision']
        recalls: float = metric['recall']
        fscores: float = metric['fscore']
        macro_precision: torch.Tensor = torch.tensor(self.desired_precisions).mean()
        macro_recall: torch.Tensor = torch.tensor(self.desired_recalls).mean()
        macro_fscore: torch.Tensor = torch.tensor(self.desired_fscores).mean()
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    @multi_device
    def test_fbeta_multilabel_micro_average_metric(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure(average='micro')
        fbeta(self.predictions, self.targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: float = metric['precision']
        recalls: float = metric['recall']
        fscores: float = metric['fscore']
        true_positives: torch.Tensor = torch.tensor([3, 3, 2, 4, 0], dtype=torch.float32)
        false_positives: torch.Tensor = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
        false_negatives: torch.Tensor = torch.tensor([1, 2, 0, 0, 0], dtype=torch.float32)
        mean_true_positive: torch.Tensor = true_positives.mean()
        mean_false_positive: torch.Tensor = false_positives.mean()
        mean_false_negative: torch.Tensor = false_negatives.mean()
        micro_precision: torch.Tensor = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall: torch.Tensor = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore: torch.Tensor = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multilabel_with_explicit_labels(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure(labels=[4, 3, 2, 1, 0])
        fbeta(self.predictions, self.targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        desired_precisions: List[float] = self.desired_precisions[::-1]
        desired_recalls: List[float] = self.desired_recalls[::-1]
        desired_fscores: List[float] = self.desired_fscores[::-1]
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multilabel_with_macro_average(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        labels: List[int] = [0, 1]
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure(average='macro', labels=labels)
        fbeta(self.predictions, self.targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: torch.Tensor = metric['precision']
        recalls: torch.Tensor = metric['recall']
        fscores: torch.Tensor = metric['fscore']
        macro_precision: torch.Tensor = torch.tensor(self.desired_precisions)[labels].mean()
        macro_recall: torch.Tensor = torch.tensor(self.desired_recalls)[labels].mean()
        macro_fscore: torch.Tensor = torch.tensor(self.desired_fscores)[labels].mean()
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

    @multi_device
    def test_fbeta_multilabel_with_micro_average(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        labels: List[int] = [1, 3]
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure(average='micro', labels=labels)
        fbeta(self.predictions, self.targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: torch.Tensor = metric['precision']
        recalls: torch.Tensor = metric['recall']
        fscores: torch.Tensor = metric['fscore']
        true_positives: torch.Tensor = torch.tensor([3, 4], dtype=torch.float32)
        false_positives: torch.Tensor = torch.tensor([0, 0], dtype=torch.float32)
        false_negatives: torch.Tensor = torch.tensor([2, 0], dtype=torch.float32)
        mean_true_positive: torch.Tensor = true_positives.mean()
        mean_false_positive: torch.Tensor = false_positives.mean()
        mean_false_negative: torch.Tensor = false_negatives.mean()
        micro_precision: torch.Tensor = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall: torch.Tensor = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore: torch.Tensor = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multilabel_with_weighted_average(self, device: torch.device) -> None:
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        labels: List[int] = [0, 1]
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure(average='weighted', labels=labels)
        fbeta(self.predictions, self.targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: float = metric['precision']
        recalls: float = metric['recall']
        fscores: float = metric['fscore']
        weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            torch.where(self.predictions >= fbeta._threshold,
                        torch.ones_like(self.predictions),
                        torch.zeros_like(self.predictions)).cpu().numpy(),
            labels=labels,
            average='weighted'
        )
        assert_allclose(precisions, weighted_precision)
        assert_allclose(recalls, weighted_recall)
        assert_allclose(fscores, weighted_fscore)

    @multi_device
    def test_fbeta_multilabel_handles_batch_size_of_one(self, device: torch.device) -> None:
        predictions: torch.Tensor = torch.tensor([[0.2862, 0.5479, 0.1627, 0.2033]], device=device)
        targets: torch.Tensor = torch.tensor([[0, 1, 0, 0]], device=device)
        mask: torch.Tensor = torch.tensor([[True]], device=device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(predictions, targets, mask)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        assert_allclose(precisions, [0.0, 1.0, 0.0, 0.0])
        assert_allclose(recalls, [0.0, 1.0, 0.0, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_false_last_class(self, device: torch.device) -> None:
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([[1, 0], [1, 0]], device=device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [0.5, 0.0])
        assert_allclose(fscores, [0.6667, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_last_class(self, device: torch.device) -> None:
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([[1, 0], [0, 1]], device=device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [1.0, 0.0])
        assert_allclose(fscores, [1.0, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_other_class(self, device: torch.device) -> None:
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([[0, 1], [1, 0]], device=device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_all_class(self, device: torch.device) -> None:
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([[0, 1], [0, 1]], device=device)
        fbeta: FBetaMultiLabelMeasure = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, Any] = fbeta.get_metric()
        precisions: List[float] = metric['precision']
        recalls: List[float] = metric['recall']
        fscores: List[float] = metric['fscore']
        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

    def test_distributed_fbeta_multilabel_measure(self) -> None:
        predictions: List[torch.Tensor] = [
            torch.tensor([[0.55, 0.25, 0.1, 0.1, 0.2],
                          [0.1, 0.6, 0.1, 0.95, 0.0],
                          [0.9, 0.8, 0.75, 0.8, 0.0]]),
            torch.tensor([[0.49, 0.5, 0.95, 0.55, 0.0],
                          [0.6, 0.49, 0.6, 0.65, 0.85],
                          [0.85, 0.4, 0.1, 0.2, 0.0]])
        ]
        targets: List[torch.Tensor] = [
            torch.tensor([[1, 1, 0, 0, 0],
                          [0, 1, 0, 1, 0],
                          [1, 1, 0, 1, 0]]),
            torch.tensor([[1, 1, 1, 1, 0],
                          [1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]])
        ]
        metric_kwargs: Dict[str, Any] = {'predictions': predictions, 'gold_labels': targets}
        desired_metrics: Dict[str, Any] = {
            'precision': self.desired_precisions,
            'recall': self.desired_recalls,
            'fscore': self.desired_fscores
        }
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            FBetaMultiLabelMeasure(),
            metric_kwargs,
            desired_metrics,
            exact=False
        )

    def test_multiple_distributed_runs(self) -> None:
        predictions: List[torch.Tensor] = [
            torch.tensor([[0.55, 0.25, 0.1, 0.1, 0.2],
                          [0.1, 0.6, 0.1, 0.95, 0.0],
                          [0.9, 0.8, 0.75, 0.8, 0.0]]),
            torch.tensor([[0.49, 0.5, 0.95, 0.55, 0.0],
                          [0.6, 0.49, 0.6, 0.65, 0.85],
                          [0.85, 0.4, 0.1, 0.2, 0.0]])
        ]
        targets: List[torch.Tensor] = [
            torch.tensor([[1, 1, 0, 0, 0],
                          [0, 1, 0, 1, 0],
                          [1, 1, 0, 1, 0]]),
            torch.tensor([[1, 1, 1, 1, 0],
                          [1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]])
        ]
        metric_kwargs: Dict[str, Any] = {'predictions': predictions, 'gold_labels': targets}
        desired_metrics: Dict[str, Any] = {
            'precision': self.desired_precisions,
            'recall': self.desired_recalls,
            'fscore': self.desired_fscores
        }
        run_distributed_test(
            [-1, -1],
            multiple_runs,
            FBetaMultiLabelMeasure(),
            metric_kwargs,
            desired_metrics,
            exact=False
        )

def multiple_runs(global_rank: int, world_size: int, gpu_id: int,
                  metric: FBetaMultiLabelMeasure,
                  metric_kwargs: Dict[str, List[torch.Tensor]],
                  desired_values: Dict[str, Any],
                  exact: bool = True) -> None:
    kwargs: Dict[str, Any] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    metric_values: Dict[str, Any] = metric.get_metric()
    for key in desired_values:
        assert_allclose(desired_values[key], metric_values[key])