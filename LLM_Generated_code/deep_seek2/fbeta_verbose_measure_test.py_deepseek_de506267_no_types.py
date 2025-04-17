from typing import Any, Dict, List, Tuple, Union
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.testing import assert_allclose
import pytest
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device, run_distributed_test, global_distributed_metric
from allennlp.training.metrics import FBetaVerboseMeasure

class FBetaVerboseMeasureTest(AllenNlpTestCase):

    def setup_method(self):
        super().setup_method()
        self.predictions: torch.Tensor = torch.tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.5, 0.1, 0.2, 0.0], [0.1, 0.2, 0.1, 0.7, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]])
        self.targets: torch.Tensor = torch.tensor([0, 4, 1, 0, 3, 0])
        self.pred_sum: List[int] = [1, 4, 0, 1, 0]
        self.true_sum: List[int] = [3, 1, 0, 1, 1]
        self.true_positive_sum: List[int] = [1, 1, 0, 1, 0]
        self.true_negative_sum: List[int] = [3, 2, 6, 5, 5]
        self.total_sum: List[int] = [6, 6, 6, 6, 6]
        desired_precisions: List[float] = [1.0, 0.25, 0.0, 1.0, 0.0]
        desired_recalls: List[float] = [1 / 3, 1.0, 0.0, 1.0, 0.0]
        desired_fscores: List[float] = [2 * p * r / (p + r) if p + r != 0.0 else 0.0 for p, r in zip(desired_precisions, desired_recalls)]
        self.desired_precisions: List[float] = desired_precisions
        self.desired_recalls: List[float] = desired_recalls
        self.desired_fscores: List[float] = desired_fscores

    @multi_device
    def test_config_errors(self, device):
        pytest.raises(ConfigurationError, FBetaVerboseMeasure, beta=0.0)
        pytest.raises(ConfigurationError, FBetaVerboseMeasure, labels=[])

    @multi_device
    def test_runtime_errors(self, device):
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        pytest.raises(RuntimeError, fbeta.get_metric)

    @multi_device
    def test_fbeta_multiclass_state(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        assert_allclose(fbeta._pred_sum.tolist(), self.pred_sum)
        assert_allclose(fbeta._true_sum.tolist(), self.true_sum)
        assert_allclose(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        assert_allclose(fbeta._true_negative_sum.tolist(), self.true_negative_sum)
        assert_allclose(fbeta._total_sum.tolist(), self.total_sum)

    @multi_device
    def test_fbeta_multiclass_metric(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(self.predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(self.predictions.size(1))]
        fscores: List[float] = [metric[f'{i}-fscore'] for i in range(self.predictions.size(1))]
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    @multi_device
    def test_fbeta_multiclass_with_mask(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        mask: torch.Tensor = torch.tensor([True, True, True, True, True, False], device=device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets, mask)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(self.predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(self.predictions.size(1))]
        fscores: List[float] = [metric[f'{i}-fscore'] for i in range(self.predictions.size(1))]
        assert_allclose(fbeta._pred_sum.tolist(), [1, 3, 0, 1, 0])
        assert_allclose(fbeta._true_sum.tolist(), [2, 1, 0, 1, 1])
        assert_allclose(fbeta._true_positive_sum.tolist(), [1, 1, 0, 1, 0])
        desired_precisions: List[float] = [1.0, 1 / 3, 0.0, 1.0, 0.0]
        desired_recalls: List[float] = [0.5, 1.0, 0.0, 1.0, 0.0]
        desired_fscores: List[float] = [2 * p * r / (p + r) if p + r != 0.0 else 0.0 for p, r in zip(desired_precisions, desired_recalls)]
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multiclass_macro_average_metric(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: float = metric['macro-precision']
        recalls: float = metric['macro-recall']
        fscores: float = metric['macro-fscore']
        num_labels: int = self.predictions.size(1)
        labels: np.ndarray = np.arange(num_labels)
        macro_precision: float
        macro_recall: float
        macro_fscore: float
        macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), average='macro', labels=labels)
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    @multi_device
    def test_fbeta_multiclass_micro_average_metric(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: float = metric['micro-precision']
        recalls: float = metric['micro-recall']
        fscores: float = metric['micro-fscore']
        micro_precision: float
        micro_recall: float
        micro_fscore: float
        micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), average='micro')
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multiclass_weighted_average_metric(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: float = metric['weighted-precision']
        recalls: float = metric['weighted-recall']
        fscores: float = metric['weighted-fscore']
        weighted_precision: float
        weighted_recall: float
        weighted_fscore: float
        weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), average='weighted')
        assert_allclose(precisions, weighted_precision)
        assert_allclose(recalls, weighted_recall)
        assert_allclose(fscores, weighted_fscore)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure(labels=[4, 3, 2, 1, 0])
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(self.predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(self.predictions.size(1))]
        fscores: List[float] = [metric[f'{i}-fscore'] for i in range(self.predictions.size(1))]
        desired_precisions: List[float] = self.desired_precisions[::-1]
        desired_recalls: List[float] = self.desired_recalls[::-1]
        desired_fscores: List[float] = self.desired_fscores[::-1]
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels_macro(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        labels: List[int] = [0, 1]
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure(labels=labels)
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: float = metric['macro-precision']
        recalls: float = metric['macro-recall']
        fscores: float = metric['macro-fscore']
        macro_precision: float
        macro_recall: float
        macro_fscore: float
        macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), average='macro', labels=labels)
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels_micro(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        labels: List[int] = [1, 3]
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure(labels=labels)
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: float = metric['micro-precision']
        recalls: float = metric['micro-recall']
        fscores: float = metric['micro-fscore']
        micro_precision: float
        micro_recall: float
        micro_fscore: float
        micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), average='micro', labels=labels)
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels_weighted(self, device):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)
        labels: List[int] = [0, 1]
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure(labels=labels)
        fbeta(self.predictions, self.targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: float = metric['weighted-precision']
        recalls: float = metric['weighted-recall']
        fscores: float = metric['weighted-fscore']
        weighted_precision: float
        weighted_recall: float
        weighted_fscore: float
        weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), labels=labels, average='weighted')
        assert_allclose(precisions, weighted_precision)
        assert_allclose(recalls, weighted_recall)
        assert_allclose(fscores, weighted_fscore)

    @multi_device
    def test_fbeta_handles_batch_size_of_one(self, device):
        predictions: torch.Tensor = torch.tensor([[0.2862, 0.3479, 0.1627, 0.2033]], device=device)
        targets: torch.Tensor = torch.tensor([1], device=device)
        mask: torch.Tensor = torch.tensor([True], device=device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(predictions, targets, mask)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(predictions.size(1))]
        assert_allclose(precisions, [0.0, 1.0, 0.0, 0.0])
        assert_allclose(recalls, [0.0, 1.0, 0.0, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_false_last_class(self, device):
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([0, 0], device=device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(predictions.size(1))]
        fscores: List[float] = [metric[f'{i}-fscore'] for i in range(predictions.size(1))]
        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [0.5, 0.0])
        assert_allclose(fscores, [0.6667, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_true_last_class(self, device):
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([0, 1], device=device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(predictions.size(1))]
        fscores: List[float] = [metric[f'{i}-fscore'] for i in range(predictions.size(1))]
        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [1.0, 0.0])
        assert_allclose(fscores, [1.0, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_true_other_class(self, device):
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([1, 0], device=device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(predictions.size(1))]
        fscores: List[float] = [metric[f'{i}-fscore'] for i in range(predictions.size(1))]
        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_true_all_class(self, device):
        predictions: torch.Tensor = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        targets: torch.Tensor = torch.tensor([1, 1], device=device)
        fbeta: FBetaVerboseMeasure = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric: Dict[str, float] = fbeta.get_metric()
        precisions: List[float] = [metric[f'{i}-precision'] for i in range(predictions.size(1))]
        recalls: List[float] = [metric[f'{i}-recall'] for i in range(predictions.size(1))]
        fscores: List[float] = [metric[f'{i}-fscore'] for i in range(predictions.size(1))]
        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

    def test_distributed_fbeta_measure(self):
        predictions: List[torch.Tensor] = [torch.tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]]), torch.tensor([[0.1, 0.5, 0.1, 0.2, 0.0], [0.1, 0.2, 0.1, 0.7, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]])]
        targets: List[torch.Tensor] = [torch.tensor([0, 4, 1]), torch.tensor([0, 3, 0])]
        metric_kwargs: Dict[str, List[Any]] = {'predictions': predictions, 'gold_labels': targets}
        desired_metrics: Dict[str, float] = {}
        for i, (p, r, f) in enumerate(zip(self.desired_precisions, self.desired_recalls, self.desired_fscores)):
            desired_metrics[f'{i}-precision'] = p
            desired_metrics[f'{i}-recall'] = r
            desired_metrics[f'{i}-fscore'] = f
        num_labels: int = self.predictions.size(1)
        labels: np.ndarray = np.arange(num_labels)
        for avg in ['macro', 'micro', 'weighted']:
            avg_precision: float
            avg_recall: float
            avg_fscore: float
            avg_precision, avg_recall, avg_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), average=avg, labels=labels)
            desired_metrics[f'{avg}-precision'] = avg_precision
            desired_metrics[f'{avg}-recall'] = avg_recall
            desired_metrics[f'{avg}-fscore'] = avg_fscore
        run_distributed_test([-1, -1], global_distributed_metric, FBetaVerboseMeasure(), metric_kwargs, desired_metrics, exact=False)

    def test_multiple_distributed_runs(self):
        predictions: List[torch.Tensor] = [torch.tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]]), torch.tensor([[0.1, 0.5, 0.1, 0.2, 0.0], [0.1, 0.2, 0.1, 0.7, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]])]
        targets: List[torch.Tensor] = [torch.tensor([0, 4, 1]), torch.tensor([0, 3, 0])]
        metric_kwargs: Dict[str, List[Any]] = {'predictions': predictions, 'gold_labels': targets}
        desired_metrics: Dict[str, float] = {}
        for i, (p, r, f) in enumerate(zip(self.desired_precisions, self.desired_recalls, self.desired_fscores)):
            desired_metrics[f'{i}-precision'] = p
            desired_metrics[f'{i}-recall'] = r
            desired_metrics[f'{i}-fscore'] = f
        num_labels: int = self.predictions.size(1)
        labels: np.ndarray = np.arange(num_labels)
        for avg in ['macro', 'micro', 'weighted']:
            avg_precision: float
            avg_recall: float
            avg_fscore: float
            avg_precision, avg_recall, avg_fscore, _ = precision_recall_fscore_support(self.targets.cpu().numpy(), self.predictions.argmax(dim=1).cpu().numpy(), average=avg, labels=labels)
            desired_metrics[f'{avg}-precision'] = avg_precision
            desired_metrics[f'{avg}-recall'] = avg_recall
            desired_metrics[f'{avg}-fscore'] = avg_fscore
        run_distributed_test([-1, -1], multiple_runs, FBetaVerboseMeasure(), metric_kwargs, desired_metrics, exact=False)

def multiple_runs(global_rank, world_size, gpu_id, metric, metric_kwargs, desired_values, exact=True):
    kwargs: Dict[str, Any] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    metric_values: Dict[str, float] = metric.get_metric()
    for key in desired_values:
        assert_allclose(desired_values[key], metric_values[key])