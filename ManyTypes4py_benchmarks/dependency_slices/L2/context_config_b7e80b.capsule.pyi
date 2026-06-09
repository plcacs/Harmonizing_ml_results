from typing import Any

# === Third-party dependency: dbt.adapters.factory ===
def get_config_class_by_name(name: str) -> Type[AdapterConfig]: ...

# === Third-party dependency: dbt.config ===
# Used symbols: IsFQNResource

# === Third-party dependency: dbt.contracts.graph.model_config ===
def get_config_for(resource_type: NodeType, base = ...) -> Type[BaseConfig]: ...

# === Third-party dependency: dbt.flags ===
def get_flags() -> Any: ...

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt.utils ===
def fqn_search(root: Dict[str, Any], fqn: List[str]) -> Iterator[Dict[str, Any]]: ...

# === Third-party dependency: dbt_common.contracts.config.base ===
class BaseConfig(AdditionalPropertiesAllowed, Replaceable): ...
def merge_config_dicts(orig_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> None: ...

# === Third-party dependency: dbt_common.exceptions ===
# Used symbols: DbtInternalError