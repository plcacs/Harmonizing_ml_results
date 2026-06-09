# === Internal dependency: prefect.exceptions ===
class HashError(PrefectException): ...

# === Internal dependency: prefect.logging ===
from .loggers import get_logger

# === Internal dependency: prefect.utilities.hashing ===
def hash_objects(*args, hash_algo=..., raise_on_failure=..., **kwargs): ...