from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from torch.testing import assert_allclose
from allennlp.common.testing import AllenNlpTestCase, multi_device, global_distributed_metric, run_distributed_test
from allennlp.training.metrics import Entropy


class EntropyTest(AllenNlpTestCase):

    @multi_device
    def func_mbc8xo1y(self, device: str) -> None:
        metric = Entropy()
        logits = torch.tensor([[10000, -10000, -10000, -1000], [10000, -
            10000, -10000, -1000]], dtype=torch.float, device=device)
        metric(logits)
        assert metric.get_metric()['entropy'] == 0.0

    @multi_device
    def func_oi62up6b(self, device: str) -> None:
        metric = Entropy()
        logits = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.
            float, device=device)
        metric(logits)
        assert_allclose(metric.get_metric()['entropy'], 1.38629436)
        logits = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.
            float, device=device)
        metric(logits)
        assert_allclose(metric.get_metric()['entropy'], 1.38629436)
        metric.reset()
        assert metric._entropy == 0.0
        assert metric._count == 0.0

    @multi_device
    def func_jtk2q3nu(self, device: str) -> None:
        metric = Entropy()
        logits = torch.tensor([[1, 1, 1, 1], [10000, -10000, -10000, -1000]
            ], dtype=torch.float, device=device)
        mask = torch.tensor([False, True], device=device)
        metric(logits, mask)
        assert metric.get_metric()['entropy'] == 0.0

    def func_bb32ba9h(self) -> None:
        logits = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float)
        logits = [logits[0], logits[1]]
        metric_kwargs = {'logits': logits}
        desired_values = {'entropy': 1.38629436}
        run_distributed_test([-1, -1], global_distributed_metric, Entropy(),
            metric_kwargs, desired_values, exact=False)

    def func_ds3rq3ac(self) -> None:
        logits = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float)
        logits = [logits[0], logits[1]]
        metric_kwargs = {'logits': logits}
        desired_values = {'entropy': 1.38629436}
        run_distributed_test([-1, -1], multiple_runs, Entropy(),
            metric_kwargs, desired_values, exact=False)


def func_ufyi3djw(global_rank: int, world_size: int, gpu_id: Optional[int], metric: Entropy,
    metric_kwargs: Dict[str, Any], desired_values: Dict[str, float], exact: bool = True) -> None:
    kwargs: Dict[str, Any] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for i in range(200):
        metric(**kwargs)
    assert_allclose(desired_values['entropy'], metric.get_metric()['entropy'])
