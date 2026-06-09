from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: array

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Series

# === Internal dependency: snorkel.map ===
# re-export: from .core import Mapper
# re-export: from .core import lambda_mapper

# === Internal dependency: snorkel.map.core ===
def get_hashable(obj: Any) -> Hashable: ...

# === Internal dependency: snorkel.types ===
# re-export: from .data import DataPoint
# re-export: from .data import FieldMap

# === Third-party dependency: spacy ===
def load(name: Union[str, Path], *, vocab: Union[Vocab, bool] = ..., disable: Union[str, Iterable[str]] = ..., enable: Union[str, Iterable[str]] = ..., exclude: Union[str, Iterable[str]] = ..., config: Union[Dict[str, Any], Config] = ...) -> Language: ...