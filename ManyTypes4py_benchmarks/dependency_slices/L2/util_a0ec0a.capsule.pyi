from typing import Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...
def check_for_gpu(device: Union[int, torch.device, List[Union[int, torch.device]]]) -> Any: ...

# === Internal dependency: allennlp.common.params ===
class Params(MutableMapping):
    def __init__(self, params: Dict[str, Any], history: str = ...) -> None: ...
    def as_flat_dict(self) -> Dict[str, Any]: ...

# === Internal dependency: allennlp.common.tqdm ===
class Tqdm:
    ...

# === Internal dependency: allennlp.common.util ===
def sanitize(x: Any) -> Any: ...
def int_to_device(device: Union[int, torch.device]) -> torch.device: ...
def dump_metrics(file_path: Optional[str], metrics: Dict[str, Any], log: bool = ...) -> None: ...

# === Internal dependency: allennlp.data ===
# re-export: from allennlp.data.data_loaders import DataLoader

# === Internal dependency: allennlp.data.Vocabulary ===
from_params: Any

# === Internal dependency: allennlp.data.dataset_readers ===
# re-export: from allennlp.data.dataset_readers.dataset_reader import DatasetReader

# === Internal dependency: allennlp.models.archival ===
CONFIG_NAME: str

# === Internal dependency: allennlp.nn.util ===
def move_to_device(obj, device: Union[torch.device, int]) -> Any: ...
def clamp_tensor(tensor, minimum, maximum) -> Any: ...

# === Third-party dependency: torch ===
# Used symbols: ByteTensor, Tensor, bool, no_grad, ones_like

# === Third-party dependency: torch.nn.utils ===
# Used symbols: clip_grad_norm_