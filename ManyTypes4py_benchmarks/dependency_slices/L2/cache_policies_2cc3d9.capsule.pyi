from typing import Any

# === Internal dependency: prefect.exceptions ===
class HashError(PrefectException): ...

# === Internal dependency: prefect.logging ===
# re-export: from .loggers import get_logger

# === Internal dependency: prefect.utilities.hashing ===
def hash_objects(*args: Any, hash_algo: Callable[..., Any] = ..., raise_on_failure: bool = ..., **kwargs: Any) -> Optional[str]: ...