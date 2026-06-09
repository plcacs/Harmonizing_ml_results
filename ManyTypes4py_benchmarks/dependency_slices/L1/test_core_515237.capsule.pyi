from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: array

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Series

# === Internal dependency: snorkel.map ===
from .core import Mapper
from .core import lambda_mapper

# === Internal dependency: snorkel.map.core ===
def get_hashable(obj): ...

# === Internal dependency: snorkel.types ===
from .data import DataPoint
from .data import FieldMap

# === Third-party dependency: spacy ===
def load(name: Union[str, Path], *, vocab: Union[Vocab, bool] = ..., disable: Union[str, Iterable[str]] = ..., enable: Union[str, Iterable[str]] = ..., exclude: Union[str, Iterable[str]] = ..., config: Union[Dict[str, Any], Config] = ...) -> Language: ...