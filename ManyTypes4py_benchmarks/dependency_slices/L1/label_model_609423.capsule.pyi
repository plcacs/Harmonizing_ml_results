from typing import Any

# === Third-party dependency: munkres ===
class Munkres:
    def __init__(self) -> Any: ...
    def compute(self, cost_matrix: Matrix) -> Sequence[Tuple[int, int]]: ...

# === Third-party dependency: numpy ===
# Used symbols: any, around, array, ceil, clip, copy, diag, exp, log, log10, ndarray, ones, random, tile, where, zeros

# === Internal dependency: snorkel.labeling.analysis ===
class LFAnalysis:
    def __init__(self, L, lfs=...): ...
    def lf_coverages(self): ...

# === Internal dependency: snorkel.labeling.model.base_labeler ===
class BaseLabeler(ABC):
    def predict_proba(self, L): ...

# === Internal dependency: snorkel.labeling.model.graph_utils ===
def get_clique_tree(nodes, edges): ...

# === Internal dependency: snorkel.labeling.model.logger ===
class Logger:
    def __init__(self, log_freq): ...

# === Internal dependency: snorkel.types ===
from .classifier import Config

# === Internal dependency: snorkel.utils.config_utils ===
def merge_config(config, config_updates): ...

# === Internal dependency: snorkel.utils.lr_schedulers ===
class LRSchedulerConfig(Config):
    ...

# === Internal dependency: snorkel.utils.optimizers ===
class OptimizerConfig(Config):
    ...

# === Third-party dependency: torch ===
# Used symbols: Tensor, clamp, cuda, diag, eye, float32, from_numpy, isnan, manual_seed, norm, ones, sum, zeros

# === Third-party dependency: torch.nn ===
# Used symbols: Module, Parameter

# === Third-party dependency: torch.optim ===
# Used symbols: Adam, Adamax, Optimizer, SGD, lr_scheduler

# === Third-party dependency: tqdm ===
# Used symbols: trange