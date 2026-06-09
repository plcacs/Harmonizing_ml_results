# === Internal dependency: core.dbt.contracts.util ===
SourceKey = Tuple[str, str]

# === Third-party dependency: dbt.artifacts.resources.base ===
class FileHash(dbtClassMixin): ...

# === Third-party dependency: dbt.constants ===
# Used symbols: MAXIMUM_SEED_SIZE

# === Third-party dependency: dbt_common.dataclass_schema ===
class dbtClassMixin(DataClassMessagePackMixin):
    ...
class StrEnum(str, SerializableType, Enum):

# === Third-party dependency: mashumaro.types ===
class SerializableType:
    ...