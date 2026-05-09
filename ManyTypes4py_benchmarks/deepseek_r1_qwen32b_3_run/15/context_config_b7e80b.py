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

    def get_config_dict(self, resource_type: NodeType) -> Dict:
        ...

class UnrenderedConfig(ConfigSource):
    def __init__(self, project: Project):
        self.project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict:
        ...

class RenderedConfig(ConfigSource):
    def __init__(self, project: Project):
        self.project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict:
        ...

class BaseContextConfigGenerator(Generic[T]):
    def __init__(self, active_project: Project):
        self._active_project = active_project

    def get_config_source(self, project: Project) -> ConfigSource:
        return RenderedConfig(project)

    def get_node_project(self, project_name: str) -> Project:
        ...

    def _project_configs(self, project: Project, fqn: List[str], resource_type: NodeType) -> Iterator[Dict]:
        ...

    def _active_project_configs(self, fqn: List[str], resource_type: NodeType) -> Iterator[Dict]:
        ...

    @abstractmethod
    def _update_from_config(self, result: T, partial: Dict, validate: bool = False) -> T:
        ...

    @abstractmethod
    def initial_result(self, resource_type: NodeType, base: bool) -> T:
        ...

    def calculate_node_config(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: bool, patch_config_dict: Optional[Dict] = None) -> T:
        ...

    @abstractmethod
    def calculate_node_config_dict(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: bool, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...

class ContextConfigGenerator(BaseContextConfigGenerator[C]):
    def __init__(self, active_project: Project):
        self._active_project = active_project

    def get_config_source(self, project: Project) -> RenderedConfig:
        return RenderedConfig(project)

    def initial_result(self, resource_type: NodeType, base: bool) -> C:
        ...

    def _update_from_config(self, result: C, partial: Dict, validate: bool = False) -> C:
        ...

    def translate_hook_names(self, project_dict: Dict) -> Dict:
        ...

    def calculate_node_config_dict(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: bool, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...

class UnrenderedConfigGenerator(BaseContextConfigGenerator[Dict[str, Any]]):
    def get_config_source(self, project: Project) -> UnrenderedConfig:
        return UnrenderedConfig(project)

    def calculate_node_config_dict(self, config_call_dict: Dict, fqn: List[str], resource_type: NodeType, project_name: str, base: bool, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...

    def initial_result(self, resource_type: NodeType, base: bool) -> Dict:
        return {}

    def _update_from_config(self, result: Dict, partial: Dict, validate: bool = False) -> Dict:
        ...

class ContextConfig:
    def __init__(self, active_project: Project, fqn: List[str], resource_type: NodeType, project_name: str):
        self._config_call_dict = {}
        self._unrendered_config_call_dict = {}
        self._active_project = active_project
        self._fqn = fqn
        self._resource_type = resource_type
        self._project_name = project_name

    def add_config_call(self, opts: Dict):
        merge_config_dicts(self._config_call_dict, opts)

    def add_unrendered_config_call(self, opts: Dict):
        self._unrendered_config_call_dict.update(opts)

    def build_config_dict(self, base: bool = False, *, rendered: bool = True, patch_config_dict: Optional[Dict] = None) -> Dict:
        ...