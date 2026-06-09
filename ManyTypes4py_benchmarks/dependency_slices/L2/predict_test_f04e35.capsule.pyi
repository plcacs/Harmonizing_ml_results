from typing import Any

# === Internal dependency: allennlp.commands ===
def main(prog: Optional[str] = ...) -> None: ...

# === Internal dependency: allennlp.commands.predict ===
class Predict(Subcommand): ...

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.testing ===
# re-export: from allennlp.common.testing.test_case import AllenNlpTestCase

# === Internal dependency: allennlp.common.util ===
def push_python_path(path: PathType) -> ContextManagerFunctionReturnType[None]: ...
JsonDict: Any

# === Internal dependency: allennlp.data.dataset_readers ===
# re-export: from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# re-export: from allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader

# === Internal dependency: allennlp.models.Model ===
resolve_class_name: Any

# === Internal dependency: allennlp.models.archival ===
def load_archive(archive_file: Union[str, PathLike], cuda_device: int = ..., overrides: Union[str, Dict[str, Any]] = ..., weights_file: str = ...) -> Archive: ...

# === Internal dependency: allennlp.predictors ===
# re-export: from allennlp.predictors.text_classifier import TextClassifierPredictor

# === Internal dependency: allennlp.predictors.Predictor ===
register: Any

# === Internal dependency: allennlp.predictors.text_classifier ===
__file__: Any

# === Third-party dependency: pytest ===
# Used symbols: raises