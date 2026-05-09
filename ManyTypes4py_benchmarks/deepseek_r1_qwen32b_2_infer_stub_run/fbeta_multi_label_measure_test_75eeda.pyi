from typing import List, Tuple, Dict, Union, Any, Optional, overload
import pytest
import torch
from allennlp.common.testing import AllenNlpTestCase
from torch import Tensor

class FBetaMultiLabelMeasureTest(AllenNlpTestCase):
    def setup_method(self) -> None:
        ...
    
    @overload
    def test_config_errors(self, device: str) -> None:
        ...
    
    @overload
    def test_runtime_errors(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_state(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_metric(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilable_with_extra_dimensions(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_with_mask(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_macro_average_metric(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_micro_average_metric(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_with_explicit_labels(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_with_macro_average(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_with_micro_average(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_with_weighted_average(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_handles_batch_size_of_one(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_handles_no_prediction_false_last_class(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_handles_no_prediction_true_last_class(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_handles_no_prediction_true_other_class(self, device: str) -> None:
        ...
    
    @overload
    def test_fbeta_multilabel_handles_no_prediction_true_all_class(self, device: str) -> None:
        ...
    
    def test_distributed_fbeta_multilabel_measure(self) -> None:
        ...
    
    def test_multiple_distributed_runs(self) -> None:
        ...

def multiple_runs(global_rank: int, world_size: int, gpu_id: int, metric: Any, metric_kwargs: Dict[str, Any], desired_values: Dict[str, Union[List[float], float]], exact: bool = True) -> None:
    ...