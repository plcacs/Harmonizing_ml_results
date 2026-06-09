from typing import Any

# === Third-party dependency: munkres ===
class Munkres:
    def __init__(self) -> Any: ...
    def compute(self, cost_matrix: Matrix) -> Sequence[Tuple[int, int]]: ...

# === Third-party dependency: numpy ===
# Used symbols: any, around, array, ceil, clip, copy, diag, exp, log, log10, ndarray, ones, random, tile, where, zeros

# === Internal dependency: snorkel.labeling.analysis ===
class LFAnalysis:
    def __init__(self, L: np.ndarray, lfs: Optional[List[LabelingFunction]] = ...) -> None: ...
    def lf_coverages(self) -> np.ndarray: ...

# === Internal dependency: snorkel.labeling.model.base_labeler ===
class BaseLabeler(ABC):
    def predict_proba(self, L: np.ndarray) -> np.ndarray: ...

# === Internal dependency: snorkel.labeling.model.graph_utils ===
def get_clique_tree(nodes: Iterable[int], edges: List[Tuple[int, int]]) -> nx.Graph: ...

# === Internal dependency: snorkel.labeling.model.logger ===
class Logger:
    def __init__(self, log_freq: int) -> None: ...

# === Internal dependency: snorkel.types ===
# re-export: from .classifier import Config

# === Internal dependency: snorkel.utils.config_utils ===
def merge_config(config: Config, config_updates: Dict[str, Any]) -> Config: ...

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