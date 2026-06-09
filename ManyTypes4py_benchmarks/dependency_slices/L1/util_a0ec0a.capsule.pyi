from typing import Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...
def check_for_gpu(device): ...

# === Internal dependency: allennlp.common.params ===
class Params(MutableMapping):
    def __init__(self, params, history=...): ...
    def as_flat_dict(self): ...

# === Internal dependency: allennlp.common.tqdm ===
class Tqdm:
    ...

# === Internal dependency: allennlp.common.util ===
def sanitize(x): ...
def int_to_device(device): ...
def dump_metrics(file_path, metrics, log=...): ...

# === Internal dependency: allennlp.data ===
from allennlp.data.data_loaders import DataLoader

# === Internal dependency: allennlp.data.Vocabulary ===
from_params: Any

# === Internal dependency: allennlp.data.dataset_readers ===
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

# === Internal dependency: allennlp.models.archival ===
CONFIG_NAME = 'config.json'

# === Internal dependency: allennlp.nn.util ===
def move_to_device(obj, device): ...
def clamp_tensor(tensor, minimum, maximum): ...

# === Third-party dependency: torch ===
# Used symbols: ByteTensor, Tensor, bool, no_grad, ones_like

# === Third-party dependency: torch.nn.utils ===
# Used symbols: clip_grad_norm_