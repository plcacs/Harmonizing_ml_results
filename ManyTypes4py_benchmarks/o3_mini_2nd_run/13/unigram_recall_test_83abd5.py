from typing import Any, Dict, List
import torch
from torch.testing import assert_allclose
from allennlp.common.testing import AllenNlpTestCase, multi_device, global_distributed_metric, run_distributed_test
from allennlp.training.metrics import UnigramRecall

class UnigramRecallTest(AllenNlpTestCase):

    @multi_device
    def test_sequence_recall(self, device: torch.device) -> None:
        recall = UnigramRecall()
        gold = torch.tensor([[1, 2, 3], [2, 4, 8], [7, 1, 1]], device=device)
        predictions = torch.tensor(
            [
                [[1, 2, 3], [1, 2, -1]],
                [[2, 4, 8], [2, 5, 9]],
                [[-1, -1, -1], [7, 1, -1]]
            ],
            device=device
        )
        recall(predictions, gold)
        actual_recall = recall.get_metric()['unigram_recall']
        assert_allclose(actual_recall, 1)

    @multi_device
    def test_sequence_recall_respects_mask(self, device: torch.device) -> None:
        recall = UnigramRecall()
        gold = torch.tensor([[2, 4, 8], [1, 2, 3], [7, 1, 1], [11, 14, 17]], device=device)
        predictions = torch.tensor(
            [
                [[2, 4, 8], [2, 5, 9]],
                [[-1, 2, 4], [3, 8, -1]],
                [[-1, -1, -1], [7, 2, -1]],
                [[12, 13, 17], [11, 13, 18]]
            ],
            device=device
        )
        mask = torch.tensor(
            [
                [True, True, True],
                [False, True, True],
                [True, True, False],
                [True, False, True]
            ],
            device=device
        )
        recall(predictions, gold, mask)
        actual_recall = recall.get_metric()['unigram_recall']
        assert_allclose(actual_recall, 7 / 8)

    @multi_device
    def test_sequence_recall_accumulates_and_resets_correctly(self, device: torch.device) -> None:
        recall = UnigramRecall()
        gold = torch.tensor([[1, 2, 3]], device=device)
        recall(torch.tensor([[[1, 2, 3]]], device=device), gold)
        recall(torch.tensor([[[7, 8, 4]]], device=device), gold)
        actual_recall = recall.get_metric(reset=True)['unigram_recall']
        assert_allclose(actual_recall, 1 / 2)
        assert recall.correct_count == 0
        assert recall.total_count == 0

    @multi_device
    def test_get_metric_on_new_object_works(self, device: torch.device) -> None:
        recall = UnigramRecall()
        actual_recall = recall.get_metric(reset=True)['unigram_recall']
        assert_allclose(actual_recall, 0)

    def test_distributed_accuracy(self) -> None:
        gold = torch.tensor([[2, 4, 8], [1, 2, 3], [7, 1, 1], [11, 14, 17]])
        predictions = torch.tensor(
            [
                [[2, 4, 8], [2, 5, 9]],
                [[-1, 2, 4], [3, 8, -1]],
                [[-1, -1, -1], [7, 2, -1]],
                [[12, 13, 17], [11, 13, 18]]
            ]
        )
        mask = torch.tensor(
            [
                [True, True, True],
                [False, True, True],
                [True, True, False],
                [True, False, True]
            ]
        )
        gold_parts: List[torch.Tensor] = [gold[:2], gold[2:]]
        predictions_parts: List[torch.Tensor] = [predictions[:2], predictions[2:]]
        mask_parts: List[torch.Tensor] = [mask[:2], mask[2:]]
        metric_kwargs: Dict[str, Any] = {'predictions': predictions_parts, 'gold_labels': gold_parts, 'mask': mask_parts}
        desired_values: Dict[str, Any] = {'unigram_recall': 7 / 8}
        run_distributed_test([-1, -1], global_distributed_metric, UnigramRecall(), metric_kwargs, desired_values, exact=False)

    def test_multiple_distributed_runs(self) -> None:
        gold = torch.tensor([[2, 4, 8], [1, 2, 3], [7, 1, 1], [11, 14, 17]])
        predictions = torch.tensor(
            [
                [[2, 4, 8], [2, 5, 9]],
                [[-1, 2, 4], [3, 8, -1]],
                [[-1, -1, -1], [7, 2, -1]],
                [[12, 13, 17], [11, 13, 18]]
            ]
        )
        mask = torch.tensor(
            [
                [True, True, True],
                [False, True, True],
                [True, True, False],
                [True, False, True]
            ]
        )
        gold_parts: List[torch.Tensor] = [gold[:2], gold[2:]]
        predictions_parts: List[torch.Tensor] = [predictions[:2], predictions[2:]]
        mask_parts: List[torch.Tensor] = [mask[:2], mask[2:]]
        metric_kwargs: Dict[str, Any] = {'predictions': predictions_parts, 'gold_labels': gold_parts, 'mask': mask_parts}
        desired_values: Dict[str, Any] = {'unigram_recall': 7 / 8}
        run_distributed_test([-1, -1], multiple_runs, UnigramRecall(), metric_kwargs, desired_values, exact=True)

def multiple_runs(global_rank: int,
                  world_size: int,
                  gpu_id: int,
                  metric: UnigramRecall,
                  metric_kwargs: Dict[str, List[torch.Tensor]],
                  desired_values: Dict[str, Any],
                  exact: bool = True) -> None:
    kwargs: Dict[str, torch.Tensor] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    assert desired_values['unigram_recall'] == metric.get_metric()['unigram_recall']