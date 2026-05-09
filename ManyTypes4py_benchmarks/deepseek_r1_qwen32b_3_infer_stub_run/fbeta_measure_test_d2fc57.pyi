from typing import Any, Dict, List, Optional, Tuple, Union
from torch import Tensor
from sklearn.metrics import precision_recall_fscore_support
from torch.testing import assert_allclose
from pytest import raises
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device, run_distributed_test, global_distributed_metric
from allennlp.training.metrics import FBetaMeasure

class FBetaMeasureTest(AllenNlpTestCase):
    def setup_method(self) -> None:
        self.predictions: Tensor
        self.targets: Tensor
        self.pred_sum: List[int]
        self.true_sum: List[int]
        self.true_positive_sum: List[int]
        self.true_negative_sum: List[int]
        self.total_sum: List[int]
        self.desired_precisions: List[float]
        self.desired_recalls: List[float]
        self.desired_fscores: List[float]

    @multi_device
    def test_config_errors(self, device: str) -> None:
        ...

    @multi_device
    def test_runtime_errors(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_state(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_metric(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_with_mask(self, device: str, mask: Optional[Tensor] = None) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_macro_average_metric(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_micro_average_metric(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels(self, device: str, labels: List[int]) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_with_macro_average(self, device: str, labels: List[int]) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_with_micro_average(self, device: str, labels: List[int]) -> None:
        ...

    @multi_device
    def test_fbeta_multiclass_with_weighted_average(self, device: str, labels: List[int]) -> None:
        ...

    @multi_device
    def test_fbeta_handles_batch_size_of_one(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_handles_no_prediction_false_last_class(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_handles_no_prediction_true_last_class(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_handles_no_prediction_true_other_class(self, device: str) -> None:
        ...

    @multi_device
    def test_fbeta_handles_no_prediction_true_all_class(self, device: str) -> None:
        ...

    def test_distributed_fbeta_measure(self) -> None:
        ...

    def test_multiple_distributed_runs(self) -> None:
        ...

def multiple_runs(global_rank: int, world_size: int, gpu_id: int, metric: FBetaMeasure, metric_kwargs: Dict[str, Any], desired_values: Dict[str, Any], exact: bool = True) -> None:
    ...