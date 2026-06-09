# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.util ===
def is_distributed() -> bool: ...

# === Internal dependency: allennlp.nn.util ===
def dist_reduce_sum(value: _V) -> _V: ...

# === Internal dependency: allennlp.training.metrics.metric ===
class Metric(Registrable):
    ...

# === Third-party dependency: torch ===
# Used symbols: FloatTensor, Tensor, arange, cat, div, full, log, matmul, mean, mm, nn, ones_like, square_, std, sum, tensor, where, zeros, zeros_like

# === Third-party dependency: torch.distributed ===
# Used symbols: ReduceOp, all_reduce