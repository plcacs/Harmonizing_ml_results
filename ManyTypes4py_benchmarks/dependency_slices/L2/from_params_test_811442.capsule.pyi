from typing import Any

# === Internal dependency: allennlp.common.Params ===
from_file: Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.from_params ===
def takes_arg(obj, arg: str) -> bool: ...
def remove_optional(annotation: type) -> Any: ...
def create_kwargs(constructor: Callable[..., T], cls: Type[T], params: Params, **extras) -> Dict[str, Any]: ...
class FromParams:
    ...

# === Internal dependency: allennlp.common.registrable ===
class Registrable(FromParams):
    ...

# === Internal dependency: allennlp.common.testing ===
# re-export: from allennlp.common.testing.test_case import AllenNlpTestCase

# === Internal dependency: allennlp.data ===
# re-export: from allennlp.data.data_loaders import DataLoader
# re-export: from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# re-export: from allennlp.data.tokenizers import Tokenizer

# === Internal dependency: allennlp.models.Model ===
from_params: Any

# === Internal dependency: allennlp.models.archival ===
def load_archive(archive_file: Union[str, PathLike], cuda_device: int = ..., overrides: Union[str, Dict[str, Any]] = ..., weights_file: str = ...) -> Archive: ...

# === Internal dependency: allennlp.nn ===
# re-export: from allennlp.nn.initializers import InitializerApplicator

# === Third-party dependency: pytest ===
# Used symbols: raises

# === Third-party dependency: torch ===
# Used symbols: all