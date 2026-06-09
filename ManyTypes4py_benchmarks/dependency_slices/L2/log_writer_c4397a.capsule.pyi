from typing import Any

# === Internal dependency: allennlp.nn.util ===
def tiny_value_of_dtype(dtype: torch.dtype) -> Any: ...

# === Internal dependency: allennlp.training.callbacks.callback ===
class TrainerCallback(Registrable):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...

# === Internal dependency: allennlp.training.util ===
def get_batch_size(batch: Union[Dict, torch.Tensor]) -> int: ...
def get_train_and_validation_metrics(metrics: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...

# === Third-party dependency: torch ===
# Used symbols: Tensor, norm, prod, tensor, utils