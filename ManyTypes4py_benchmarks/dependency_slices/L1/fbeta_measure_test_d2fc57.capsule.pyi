# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.testing ===
def multi_device(test_method): ...
def global_distributed_metric(global_rank, world_size, gpu_id, metric, metric_kwargs, desired_values, exact=..., number_of_runs=...): ...
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.common.testing.distributed_test import run_distributed_test

# === Internal dependency: allennlp.training.metrics ===
from allennlp.training.metrics.fbeta_measure import FBetaMeasure

# === Third-party dependency: pytest ===
# Used symbols: raises

# === Third-party dependency: sklearn.metrics ===
# Used symbols: precision_recall_fscore_support

# === Third-party dependency: torch ===
# Used symbols: float32, tensor

# === Third-party dependency: torch.testing ===
# Used symbols: assert_allclose