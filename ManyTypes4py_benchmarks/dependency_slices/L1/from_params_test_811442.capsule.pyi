from typing import Any

# === Internal dependency: allennlp.common.Params ===
from_file: Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.from_params ===
def takes_arg(obj, arg): ...
def remove_optional(annotation): ...
def create_kwargs(constructor, cls, params, **extras): ...
class FromParams:
    ...

# === Internal dependency: allennlp.common.registrable ===
class Registrable(FromParams):
    ...

# === Internal dependency: allennlp.common.testing ===
from allennlp.common.testing.test_case import AllenNlpTestCase

# === Internal dependency: allennlp.data ===
from allennlp.data.data_loaders import DataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer

# === Internal dependency: allennlp.models.Model ===
from_params: Any

# === Internal dependency: allennlp.models.archival ===
def load_archive(archive_file, cuda_device=..., overrides=..., weights_file=...): ...

# === Internal dependency: allennlp.nn ===
from allennlp.nn.initializers import InitializerApplicator

# === Third-party dependency: pytest ===
# Used symbols: raises

# === Third-party dependency: torch ===
# Used symbols: all