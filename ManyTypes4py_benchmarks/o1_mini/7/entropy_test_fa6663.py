from typing import Any, Dict, List, Tuple, Union
import torch
from torch.testing import assert_allclose
from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.training.metrics import Entropy


class EntropyTest(AllenNlpTestCase):
    @multi_device
    def test_low_entropy_distribution(self, device: torch.device) -> None:
        metric: Entropy = Entropy()
        logits: torch.Tensor = torch.tensor(
            [
                [10000, -10000, -10000, -1000],
                [10000, -10000, -10000, -1000],
            ],
            dtype=torch.float,
            device=device,
        )
        metric(logits)
        assert metric.get_metric()["entropy"] == 0.0

    @multi_device
    def test_entropy_for_uniform_distribution(self, device: torch.device) -> None:
        metric: Entropy = Entropy()
        logits: torch.Tensor = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.float,
            device=device,
        )
        metric(logits)
        assert_allclose(metric.get_metric()["entropy"], 1.38629436)
        logits = torch.tensor(
            [
                [2, 2, 2, 2],
                [2, 2, 2, 2],
            ],
            dtype=torch.float,
            device=device,
        )
        metric(logits)
        assert_allclose(metric.get_metric()["entropy"], 1.38629436)
        metric.reset()
        assert metric._entropy == 0.0
        assert metric._count == 0.0

    @multi_device
    def test_masked_case(self, device: torch.device) -> None:
        metric: Entropy = Entropy()
        logits: torch.Tensor = torch.tensor(
            [
                [1, 1, 1, 1],
                [10000, -10000, -10000, -1000],
            ],
            dtype=torch.float,
            device=device,
        )
        mask: torch.BoolTensor = torch.tensor([False, True], device=device)
        metric(logits, mask)
        assert metric.get_metric()["entropy"] == 0.0

    def test_distributed_entropy(self) -> None:
        logits: torch.Tensor = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.float,
        )
        logits = [logits[0], logits[1]]
        metric_kwargs: Dict[str, List[torch.Tensor]] = {"logits": logits}
        desired_values: Dict[str, float] = {"entropy": 1.38629436}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            Entropy(),
            metric_kwargs,
            desired_values,
            exact=False,
        )

    def test_multiple_distributed_runs(self) -> None:
        logits: torch.Tensor = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.float,
        )
        logits = [logits[0], logits[1]]
        metric_kwargs: Dict[str, List[torch.Tensor]] = {"logits": logits}
        desired_values: Dict[str, float] = {"entropy": 1.38629436}
        run_distributed_test(
            [-1, -1],
            multiple_runs,
            Entropy(),
            metric_kwargs,
            desired_values,
            exact=False,
        )


def multiple_runs(
    global_rank: int,
    world_size: int,
    gpu_id: int,
    metric: Entropy,
    metric_kwargs: Dict[str, List[torch.Tensor]],
    desired_values: Dict[str, float],
    exact: bool = True,
) -> None:
    kwargs: Dict[str, torch.Tensor] = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for _ in range(200):
        metric(**kwargs)
    assert_allclose(
        desired_values["entropy"], metric.get_metric()["entropy"], atol=1e-6
    )
