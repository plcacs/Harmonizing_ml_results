from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import torch
import numpy as np

class FBetaVerboseMeasureTest:
    def setup_method(self) -> None:
        ...

    @pytest.mark.multi_device
    def test_config_errors(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_runtime_errors(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_state(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_metric(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_with_mask(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_macro_average_metric(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_micro_average_metric(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_weighted_average_metric(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_with_explicit_labels(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_with_explicit_labels_macro(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_with_explicit_labels_micro(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_multiclass_with_explicit_labels_weighted(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_handles_batch_size_of_one(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_handles_no_prediction_false_last_class(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_handles_no_prediction_true_last_class(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_handles_no_prediction_true_other_class(self, device: str) -> None:
        ...

    @pytest.mark.multi_device
    def test_fbeta_handles_no_prediction_true_all_class(self, device: str) -> None:
        ...

    def test_distributed_fbeta_measure(self) -> None:
        ...

    def test_multiple_distributed_runs(self) -> None:
        ...

def multiple_runs(global_rank: int, world_size: int, gpu_id: int, metric: Any, metric_kwargs: Dict[str, Any], desired_values: Dict[str, Union[float, List[float]]], exact: bool = True) -> None:
    ...