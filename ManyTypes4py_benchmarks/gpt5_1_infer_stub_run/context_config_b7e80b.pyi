from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar
from dbt.config import IsFQNResource, RuntimeConfig
from dbt.node_types import NodeType
from dbt_common.contracts.config.base import BaseConfig

T = TypeVar("T")
C = TypeVar("C", bound=BaseConfig)


class ModelParts(IsFQNResource):
    ...


class ConfigSource:
    def __init__(self, project: Any) -> None: ...
    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]: ...


class UnrenderedConfig(ConfigSource):
    def __init__(self, project: Any) -> None: ...
    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]: ...


class RenderedConfig(ConfigSource):
    def __init__(self, project: Any) -> None: ...
    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]: ...


class BaseContextConfigGenerator(Generic[T]):
    def __init__(self, active_project: RuntimeConfig) -> None: ...
    def get_config_source(self, project: Any) -> ConfigSource: ...
    def get_node_project(self, project_name: str) -> Any: ...
    def _project_configs(
        self, project: Any, fqn: List[str], resource_type: NodeType
    ) -> Iterator[Dict[str, Any]]: ...
    def _active_project_configs(
        self, fqn: List[str], resource_type: NodeType
    ) -> Iterator[Dict[str, Any]]: ...
    @abstractmethod
    def _update_from_config(self, result: T, partial: Dict[str, Any], validate: bool = False) -> T: ...
    @abstractmethod
    def initial_result(self, resource_type: NodeType, base: bool) -> T: ...
    def calculate_node_config(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> T: ...
    @abstractmethod
    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...


class ContextConfigGenerator(BaseContextConfigGenerator[C]):
    def __init__(self, active_project: RuntimeConfig) -> None: ...
    def get_config_source(self, project: Any) -> ConfigSource: ...
    def initial_result(self, resource_type: NodeType, base: bool) -> C: ...
    def _update_from_config(self, result: C, partial: Dict[str, Any], validate: bool = False) -> C: ...
    def translate_hook_names(self, project_dict: Dict[str, Any]) -> Dict[str, Any]: ...
    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...


class UnrenderedConfigGenerator(BaseContextConfigGenerator[Dict[str, Any]]):
    def get_config_source(self, project: Any) -> ConfigSource: ...
    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
    def initial_result(self, resource_type: NodeType, base: bool) -> Dict[str, Any]: ...
    def _update_from_config(
        self, result: Dict[str, Any], partial: Dict[str, Any], validate: bool = False
    ) -> Dict[str, Any]: ...


class ContextConfig:
    def __init__(
        self,
        active_project: RuntimeConfig,
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
    ) -> None: ...
    def add_config_call(self, opts: Dict[str, Any]) -> None: ...
    def add_unrendered_config_call(self, opts: Dict[str, Any]) -> None: ...
    def build_config_dict(
        self,
        base: bool = False,
        *,
        rendered: bool = True,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...