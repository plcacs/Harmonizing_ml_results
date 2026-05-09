from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, TypeVar, Union
from dbt.adapters.factory import get_config_class_by_name
from dbt.config import IsFQNResource, Project, RuntimeConfig
from dbt.contracts.graph.model_config import get_config_for
from dbt.exceptions import SchemaConfigError
from dbt.flags import get_flags
from dbt.node_types import NodeType
from dbt.utils import fqn_search
from dbt_common.contracts.config.base import BaseConfig
from dbt_common.exceptions import DbtInternalError

@dataclass
class ModelParts(IsFQNResource):
    ...

T = TypeVar('T')
C = TypeVar('C', bound=BaseConfig)

class ConfigSource:
    def __init__(self, project: Any) -> None:
        ...

    def get_config_dict(self, resource_type: Any) -> Dict:
        ...

class UnrenderedConfig(ConfigSource):
    def __init__(self, project: Any) -> None:
        ...

    def get_config_dict(self, resource_type: Any) -> Dict:
        ...

class RenderedConfig(ConfigSource):
    def __init__(self, project: Any) -> None:
        ...

    def get_config_dict(self, resource_type: Any) -> Dict:
        ...

class BaseContextConfigGenerator(Generic[T]):
    def __init__(self, active_project: Any) -> None:
        ...

    def get_config_source(self, project: Any) -> ConfigSource:
        ...

    def get_node_project(self, project_name: str) -> Any:
        ...

    def _project_configs(self, project: Any, fqn: List[str], resource_type: Any) -> Generator[Dict, None, None]:
        ...

    def _active_project_configs(self, fqn: List[str], resource_type: Any) -> Generator[Dict, None, None]:
        ...

    @abstractmethod
    def _update_from_config(self, result: T, partial: Dict, validate: bool = False) -> T:
        ...

    @abstractmethod
    def initial_result(self, resource_type: NodeType, base: Any) -> T:
        ...

    def calculate_node_config(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict] = None) -> T:
        ...

    @abstractmethod
    def calculate_node_config_dict(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...

class ContextConfigGenerator(BaseContextConfigGenerator[C]):
    def __init__(self, active_project: Any) -> None:
        ...

    def get_config_source(self, project: Any) -> ConfigSource:
        ...

    def initial_result(self, resource_type: NodeType, base: Any) -> C:
        ...

    def _update_from_config(self, result: C, partial: Dict, validate: bool = False) -> C:
        ...

    def translate_hook_names(self, project_dict: Dict) -> Dict:
        ...

    def calculate_node_config_dict(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...

class UnrenderedConfigGenerator(BaseContextConfigGenerator[Dict[str, Any]]):
    def get_config_source(self, project: Any) -> ConfigSource:
        ...

    def calculate_node_config_dict(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...

    def initial_result(self, resource_type: NodeType, base: Any) -> Dict[str, Any]:
        ...

    def _update_from_config(self, result: Dict[str, Any], partial: Dict, validate: bool = False) -> Dict[str, Any]:
        ...

class ContextConfig:
    def __init__(self, active_project: Any, fqn: List[str], resource_type: NodeType, project_name: str) -> None:
        ...

    def add_config_call(self, opts: Dict) -> None:
        ...

    def add_unrendered_config_call(self, opts: Dict) -> None:
        ...

    def build_config_dict(self, base: bool = False, *, rendered: bool = True, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...