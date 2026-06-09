# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.util ===
def is_distributed() -> bool: ...

# === Internal dependency: allennlp.training.metrics.metric ===
class Metric(Registrable):
    ...

# === Third-party dependency: scipy.stats ===
# Used symbols: wasserstein_distance

# === Third-party dependency: torch ===
# Used symbols: FloatTensor, tensor, zeros

# === Third-party dependency: torch.distributed ===
# Used symbols: ReduceOp, all_reduce

# === Third-party dependency: torch.distributions.categorical ===
class Categorical(Distribution):
    def __init__(self, probs: Tensor | None = ..., logits: Tensor | None = ..., validate_args: bool | None = ...) -> None: ...
    def probs(self) -> Tensor: ...

# === Third-party dependency: torch.distributions.kl ===
def kl_divergence(p: Distribution, q: Distribution) -> Tensor: ...