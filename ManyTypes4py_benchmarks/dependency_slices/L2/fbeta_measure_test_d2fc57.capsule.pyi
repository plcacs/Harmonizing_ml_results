from typing import Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.testing ===
def multi_device(test_method) -> Any: ...
def global_distributed_metric(global_rank: int, world_size: int, gpu_id: Union[int, torch.device], metric: Metric, metric_kwargs: Dict[str, List[Any]], desired_values: Dict[str, Any], exact: Union[bool, Tuple[float, float]] = ..., number_of_runs: int = ...) -> Any: ...
# re-export: from allennlp.common.testing.test_case import AllenNlpTestCase
# re-export: from allennlp.common.testing.distributed_test import run_distributed_test

# === Internal dependency: allennlp.training.metrics ===
# re-export: from allennlp.training.metrics.fbeta_measure import FBetaMeasure

# === Third-party dependency: pytest ===
# Used symbols: raises

# === Third-party dependency: sklearn.metrics ===
# Used symbols: precision_recall_fscore_support

# === Third-party dependency: torch ===
# Used symbols: float32, tensor

# === Third-party dependency: torch.testing ===
# Used symbols: assert_allclose