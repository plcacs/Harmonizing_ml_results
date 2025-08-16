from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar
from dbt.adapters.factory import get_config_class_by_name
from dbt.config import IsFQNResource, Project, RuntimeConfig
from dbt.contracts.graph.model_config import get_config_for
from dbt.exceptions import SchemaConfigError
from dbt.flags import get_flags
from dbt.node_types import NodeType
from dbt.utils import fqn_search
from dbt_common.contracts.config.base import BaseConfig, merge_config_dicts
from dbt_common.dataclass_schema import ValidationError
from dbt_common.exceptions import DbtInternalError

@dataclass
class ModelParts(IsFQNResource):
    pass

T = TypeVar('T')
C = TypeVar('C', bound=BaseConfig)

class ConfigSource:

    def __init__(self, project: Project):
        self.project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]:
        ...

class UnrenderedConfig(ConfigSource):

    def __init__(self, project: Project):
        self.project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]:
        ...

class RenderedConfig(ConfigSource):

    def __init__(self, project: Project):
        self.project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]:
        ...

class BaseContextConfigGenerator(Generic[T]):

    def __init__(self, active_project: Project):
        self._active_project = active_project

    def get_config_source(self, project: Project) -> ConfigSource:
        ...

    def get_node_project(self, project_name: str) -> Project:
        ...

    def _project_configs(self, project: Project, fqn: str, resource_type: NodeType) -> Iterator[Dict[str, Any]]:
        ...

    def _active_project_configs(self, fqn: str, resource_type: NodeType) -> Iterator[Dict[str, Any]]:
        ...

    @abstractmethod
    def _update_from_config(self, result: T, partial: Dict[str, Any], validate: bool = False) -> T:
        ...

    @abstractmethod
    def initial_result(self, resource_type: NodeType, base: Any) -> T:
        ...

    def calculate_node_config(self, config_call_dict: Dict[str, Any], fqn: str, resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict[str, Any]] = None) -> T:
        ...

    @abstractmethod
    def calculate_node_config_dict(self, config_call_dict: Dict[str, Any], fqn: str, resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

class ContextConfigGenerator(BaseContextConfigGenerator[C]):

    def __init__(self, active_project: Project):
        self._active_project = active_project

    def get_config_source(self, project: Project) -> ConfigSource:
        ...

    def initial_result(self, resource_type: NodeType, base: Any) -> C:
        ...

    def _update_from_config(self, result: C, partial: Dict[str, Any], validate: bool = False) -> C:
        ...

    def translate_hook_names(self, project_dict: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def calculate_node_config_dict(self, config_call_dict: Dict[str, Any], fqn: str, resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

class UnrenderedConfigGenerator(BaseContextConfigGenerator[Dict[str, Any]]):

    def get_config_source(self, project: Project) -> ConfigSource:
        ...

    def calculate_node_config_dict(self, config_call_dict: Dict[str, Any], fqn: str, resource_type: NodeType, project_name: str, base: Any, patch_config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    def initial_result(self, resource_type: NodeType, base: Any) -> Dict[str, Any]:
        ...

    def _update_from_config(self, result: Dict[str, Any], partial: Dict[str, Any], validate: bool = False) -> Dict[str, Any]:
        ...

class ContextConfig:

    def __init__(self, active_project: Project, fqn: str, resource_type: NodeType, project_name: str):
        ...

    def add_config_call(self, opts: Dict[str, Any]):
        ...

    def add_unrendered_config_call(self, opts: Dict[str, Any]):
        ...

    def build_config_dict(self, base: bool = False, *, rendered: bool = True, patch_config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...
