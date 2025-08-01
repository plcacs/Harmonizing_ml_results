from typing import Any, Dict, List
import torch
import pytest
from allennlp.common.testing import AllenNlpTestCase, multi_device, global_distributed_metric, run_distributed_test
from allennlp.training.metrics import BooleanAccuracy

class BooleanAccuracyTest(AllenNlpTestCase):

    @multi_device
    def test_accuracy_computation(self, device: torch.device) -> None:
        accuracy = BooleanAccuracy()
        predictions = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], device=device)
        targets = torch.tensor([[0, 1], [2, 2], [4, 5], [7, 7]], device=device)
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 2 / 4
        mask = torch.ones(4, 2, device=device).bool()
        mask[1, 1] = 0
        accuracy(predictions, targets, mask)
        assert accuracy.get_metric() == 5 / 8
        targets[1, 1] = 3
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 8 / 12
        accuracy.reset()
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 3 / 4

    @multi_device
    def test_skips_completely_masked_instances(self, device: torch.device) -> None:
        accuracy = BooleanAccuracy()
        predictions = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], device=device)
        targets = torch.tensor([[0, 1], [2, 2], [4, 5], [7, 7]], device=device)
        mask = torch.tensor([[False, False], [True, False], [True, True], [True, True]], device=device)
        accuracy(predictions, targets, mask)
        assert accuracy.get_metric() == 2 / 3

    @multi_device
    def test_incorrect_gold_labels_shape_catches_exceptions(self, device: torch.device) -> None:
        accuracy = BooleanAccuracy()
        predictions = torch.rand([5, 7], device=device)
        incorrect_shape_labels = torch.rand([5, 8], device=device)
        with pytest.raises(ValueError):
            accuracy(predictions, incorrect_shape_labels)

    @multi_device
    def test_incorrect_mask_shape_catches_exceptions(self, device: torch.device) -> None:
        accuracy = BooleanAccuracy()
        predictions = torch.rand([5, 7], device=device)
        labels = torch.rand([5, 7], device=device)
        incorrect_shape_mask = torch.randint(0, 2, [5, 8], device=device).bool()
        with pytest.raises(ValueError):
            accuracy(predictions, labels, incorrect_shape_mask)

    @multi_device
    def test_does_not_divide_by_zero_with_no_count(self, device: torch.device) -> None:
        accuracy = BooleanAccuracy()
        assert accuracy.get_metric() == pytest.approx(0.0)

    def test_distributed_accuracy(self) -> None:
        predictions: List[torch.Tensor] = [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4, 5], [6, 7]])]
        targets: List[torch.Tensor] = [torch.tensor([[0, 1], [2, 2]]), torch.tensor([[4, 5], [7, 7]])]
        metric_kwargs: Dict[str, List[torch.Tensor]] = {'predictions': predictions, 'gold_labels': targets}
        desired_values: float = 0.5
        run_distributed_test([-1, -1], global_distributed_metric, BooleanAccuracy(), metric_kwargs, desired_values, exact=True)

    def test_distributed_accuracy_unequal_batches(self) -> None:
        predictions: List[torch.Tensor] = [torch.tensor([[0, 1], [2, 3], [4, 5]]), torch.tensor([[6, 7]])]
        targets: List[torch.Tensor] = [torch.tensor([[0, 1], [2, 2], [4, 5]]), torch.tensor([[7, 7]])]
        metric_kwargs: Dict[str, List[torch.Tensor]] = {'predictions': predictions, 'gold_labels': targets}
        desired_values: float = 0.5
        run_distributed_test([-1, -1], global_distributed_metric, BooleanAccuracy(), metric_kwargs, desired_values, exact=True)

    def test_multiple_distributed_runs(self) -> None:
        predictions: List[torch.Tensor] = [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4, 5], [6, 7]])]
        targets: List[torch.Tensor] = [torch.tensor([[0, 1], [2, 2]]), torch.tensor([[4, 5], [7, 7]])]
        metric_kwargs: Dict[str, List[torch.Tensor]] = {'predictions': predictions, 'gold_labels': targets}
        desired_values: float = 0.5
        run_distributed_test([-1, -1], multiple_runs, BooleanAccuracy(), metric_kwargs, desired_values, exact=True)

def multiple_runs(global_rank: int,
                  world_size: int,
                  gpu_id: int,
                  metric: BooleanAccuracy,
                  metric_kwargs: Dict[str, List[torch.Tensor]],
                  desired_values: float,
                  exact: bool = True) -> None:
    kwargs: Dict[str, torch.Tensor] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    assert desired_values == metric.get_metric()