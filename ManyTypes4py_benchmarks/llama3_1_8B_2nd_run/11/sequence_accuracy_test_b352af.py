from typing import Any, Dict, List, Tuple, Union
import torch
from torch.testing import assert_allclose
from allennlp.common.testing import AllenNlpTestCase, multi_device, global_distributed_metric, run_distributed_test
from allennlp.training.metrics import SequenceAccuracy

class SequenceAccuracyTest(AllenNlpTestCase):
    """Tests for SequenceAccuracy metric."""

    @multi_device
    def test_sequence_accuracy(self, device: torch.device) -> None:
        """Test sequence accuracy with no mask."""
        accuracy: SequenceAccuracy = SequenceAccuracy()
        gold: torch.Tensor = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1]], device=device)
        predictions: torch.Tensor = torch.tensor([[[1, 2, 3], [1, 2, -1]], [[2, 4, 8], [2, 5, 9]], [[-1, -1, -1], [0, 1, -1]]], device=device)
        accuracy(predictions, gold)
        actual_accuracy: float = accuracy.get_metric()['accuracy']
        assert_allclose(actual_accuracy, 2 / 3)

    @multi_device
    def test_sequence_accuracy_respects_mask(self, device: torch.device) -> None:
        """Test sequence accuracy with mask."""
        accuracy: SequenceAccuracy = SequenceAccuracy()
        gold: torch.Tensor = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1], [11, 13, 17]], device=device)
        predictions: torch.Tensor = torch.tensor([[[1, 2, 3], [1, 2, -1]], [[2, 4, 8], [2, 5, 9]], [[-1, -1, -1], [0, 1, -1]], [[12, 13, 17], [11, 13, 18]]], device=device)
        mask: torch.Tensor = torch.tensor([[False, True, True], [True, True, True], [True, True, False], [True, False, True]], device=device)
        accuracy(predictions, gold, mask)
        actual_accuracy: float = accuracy.get_metric()['accuracy']
        assert_allclose(actual_accuracy, 3 / 4)

    @multi_device
    def test_sequence_accuracy_accumulates_and_resets_correctly(self, device: torch.device) -> None:
        """Test sequence accuracy accumulation and reset."""
        accuracy: SequenceAccuracy = SequenceAccuracy()
        gold: torch.Tensor = torch.tensor([[1, 2, 3]], device=device)
        accuracy(torch.tensor([[[1, 2, 3]]], device=device), gold)
        accuracy(torch.tensor([[[1, 2, 4]]], device=device), gold)
        actual_accuracy: float = accuracy.get_metric(reset=True)['accuracy']
        assert_allclose(actual_accuracy, 1 / 2)
        assert accuracy.correct_count == 0
        assert accuracy.total_count == 0

    @multi_device
    def test_get_metric_on_new_object_works(self, device: torch.device) -> None:
        """Test getting metric on a new object."""
        accuracy: SequenceAccuracy = SequenceAccuracy()
        actual_accuracy: float = accuracy.get_metric(reset=True)['accuracy']
        assert_allclose(actual_accuracy, 0)

    def test_distributed_sequence_accuracy(self) -> None:
        """Test distributed sequence accuracy."""
        gold: torch.Tensor = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1], [11, 13, 17]])
        predictions: torch.Tensor = torch.tensor([[[1, 2, 3], [1, 2, -1]], [[2, 4, 8], [2, 5, 9]], [[-1, -1, -1], [0, 1, -1]], [[12, 13, 17], [11, 13, 18]]])
        mask: torch.Tensor = torch.tensor([[False, True, True], [True, True, True], [True, True, False], [True, False, True]])
        gold: List[torch.Tensor] = [gold[:2], gold[2:]]
        predictions: List[torch.Tensor] = [predictions[:2], predictions[2:]]
        mask: List[torch.Tensor] = [mask[:2], mask[2:]]
        metric_kwargs: Dict[str, Any] = {'predictions': predictions, 'gold_labels': gold, 'mask': mask}
        desired_values: Dict[str, float] = {'accuracy': 3 / 4}
        run_distributed_test([-1, -1], global_distributed_metric, SequenceAccuracy(), metric_kwargs, desired_values, exact=False)

    def test_multiple_distributed_runs(self) -> None:
        """Test multiple distributed runs."""
        gold: torch.Tensor = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1], [11, 13, 17]])
        predictions: torch.Tensor = torch.tensor([[[1, 2, 3], [1, 2, -1]], [[2, 4, 8], [2, 5, 9]], [[-1, -1, -1], [0, 1, -1]], [[12, 13, 17], [11, 13, 18]]])
        mask: torch.Tensor = torch.tensor([[False, True, True], [True, True, True], [True, True, False], [True, False, True]])
        gold: List[torch.Tensor] = [gold[:2], gold[2:]]
        predictions: List[torch.Tensor] = [predictions[:2], predictions[2:]]
        mask: List[torch.Tensor] = [mask[:2], mask[2:]]
        metric_kwargs: Dict[str, Any] = {'predictions': predictions, 'gold_labels': gold, 'mask': mask}
        desired_values: Dict[str, float] = {'accuracy': 3 / 4}
        run_distributed_test([-1, -1], multiple_runs, SequenceAccuracy(), metric_kwargs, desired_values, exact=True)

def multiple_runs(global_rank: int, world_size: int, gpu_id: int, metric: SequenceAccuracy, metric_kwargs: Dict[str, Any], desired_values: Dict[str, float], exact: bool = True) -> None:
    """Test multiple distributed runs."""
    kwargs: Dict[str, Any] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    assert desired_values['accuracy'] == metric.get_metric()['accuracy']
