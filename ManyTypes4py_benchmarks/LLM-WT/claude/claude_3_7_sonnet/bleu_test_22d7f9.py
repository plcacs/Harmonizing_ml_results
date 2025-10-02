from typing import Any, Dict, List, Tuple, Union, Set, Optional, Counter as CounterType
import math
from collections import Counter
import torch
from torch.testing import assert_allclose
from allennlp.common.testing import AllenNlpTestCase, multi_device, global_distributed_metric, run_distributed_test
from allennlp.training.metrics import BLEU
from allennlp.training.util import ngrams, get_valid_tokens_mask

class BleuTest(AllenNlpTestCase):

    def setup_method(self) -> None:
        super().setup_method()
        self.metric = BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0})

    @multi_device
    def test_get_valid_tokens_mask(self, device: str) -> None:
        tensor = torch.tensor([[1, 2, 3, 0], [0, 1, 1, 0]], device=device)
        result = get_valid_tokens_mask(tensor, self.metric._exclude_indices).long()
        check = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 0]], device=device)
        assert_allclose(result, check)

    @multi_device
    def test_ngrams(self, device: str) -> None:
        tensor = torch.tensor([1, 2, 3, 1, 2, 0], device=device)
        exclude_indices = self.metric._exclude_indices
        counts: CounterType[Tuple[int, ...]] = Counter(ngrams(tensor, 1, exclude_indices))
        unigram_check: Dict[Tuple[int, ...], int] = {(1,): 2, (2,): 2, (3,): 1}
        assert counts == unigram_check
        counts = Counter(ngrams(tensor, 2, exclude_indices))
        bigram_check: Dict[Tuple[int, ...], int] = {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        assert counts == bigram_check
        counts = Counter(ngrams(tensor, 3, exclude_indices))
        trigram_check: Dict[Tuple[int, ...], int] = {(1, 2, 3): 1, (2, 3, 1): 1, (3, 1, 2): 1}
        assert counts == trigram_check
        counts = Counter(ngrams(tensor, 7, exclude_indices))
        assert counts == {}

    @multi_device
    def test_bleu_computed_correctly(self, device: str) -> None:
        self.metric.reset()
        predictions = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], device=device)
        gold_targets = torch.tensor([[2, 0, 0], [1, 0, 0], [1, 1, 2]], device=device)
        self.metric(predictions, gold_targets)
        assert self.metric._prediction_lengths == 6
        assert self.metric._reference_lengths == 5
        assert self.metric._precision_matches[1] == 0 + 1 + 2
        assert self.metric._precision_totals[1] == 1 + 2 + 3
        assert self.metric._precision_matches[2] == 0 + 0 + 1
        assert self.metric._precision_totals[2] == 0 + 1 + 2
        assert self.metric._get_brevity_penalty() == 1.0
        bleu = self.metric.get_metric(reset=True)['BLEU']
        check = math.exp(0.5 * (math.log(3) - math.log(6)) + 0.5 * (math.log(1) - math.log(3)))
        assert_allclose(bleu, check)

    @multi_device
    def test_bleu_computed_with_zero_counts(self, device: str) -> None:
        self.metric.reset()
        assert self.metric.get_metric()['BLEU'] == 0

    def test_distributed_bleu(self) -> None:
        predictions: List[torch.Tensor] = [torch.tensor([[1, 0, 0], [1, 1, 0]]), torch.tensor([[1, 1, 1]])]
        gold_targets: List[torch.Tensor] = [torch.tensor([[2, 0, 0], [1, 0, 0]]), torch.tensor([[1, 1, 2]])]
        check = math.exp(0.5 * (math.log(3) - math.log(6)) + 0.5 * (math.log(1) - math.log(3)))
        metric_kwargs: Dict[str, List[torch.Tensor]] = {'predictions': predictions, 'gold_targets': gold_targets}
        desired_values: Dict[str, float] = {'BLEU': check}
        run_distributed_test([-1, -1], global_distributed_metric, BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0}), metric_kwargs, desired_values, exact=False)

    def test_multiple_distributed_runs(self) -> None:
        predictions: List[torch.Tensor] = [torch.tensor([[1, 0, 0], [1, 1, 0]]), torch.tensor([[1, 1, 1]])]
        gold_targets: List[torch.Tensor] = [torch.tensor([[2, 0, 0], [1, 0, 0]]), torch.tensor([[1, 1, 2]])]
        check = math.exp(0.5 * (math.log(3) - math.log(6)) + 0.5 * (math.log(1) - math.log(3)))
        metric_kwargs: Dict[str, List[torch.Tensor]] = {'predictions': predictions, 'gold_targets': gold_targets}
        desired_values: Dict[str, float] = {'BLEU': check}
        run_distributed_test([-1, -1], multiple_runs, BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0}), metric_kwargs, desired_values, exact=False)

def multiple_runs(global_rank: int, world_size: int, gpu_id: int, metric: BLEU, 
                 metric_kwargs: Dict[str, List[torch.Tensor]], 
                 desired_values: Dict[str, float], exact: bool = True) -> None:
    kwargs: Dict[str, torch.Tensor] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    assert_allclose(desired_values['BLEU'], metric.get_metric()['BLEU'])
