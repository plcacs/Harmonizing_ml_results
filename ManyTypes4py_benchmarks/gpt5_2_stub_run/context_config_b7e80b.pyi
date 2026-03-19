from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar
from dbt.config import IsFQNResource
from dbt_common.contracts.config.base import BaseConfig

T = TypeVar("T")
C = TypeVar("C", bound=BaseConfig)

class ModelParts(IsFQNResource):
    ...

class ConfigSource:
    project: Any
    def __init__(self, project: Any) -> None: ...
    def get_config_dict(self, resource_type: Any) -> Any: ...

class UnrenderedConfig(ConfigSource):
    def __init__(self, project: Any) -> None: ...
    def get_config_dict(self, resource_type: Any) -> Dict[str, Any]: ...

class RenderedConfig(ConfigSource):
    def __init__(self, project: Any) -> None: ...
    def get_config_dict(self, resource_type: Any) -> Dict[str, Any]: ...

class BaseContextConfigGenerator(Generic[T]):
    _active_project: Any
    def __init__(self, active_project: Any) -> None: ...
    def get_config_source(self, project: Any) -> ConfigSource: ...
    def get_node_project(self, project_name: Any) -> Any: ...
    def _project_configs(self, project: Any, fqn: Any, resource_type: Any) -> Iterator[Dict[str, Any]]: ...
    def _active_project_configs(self, fqn: Any, resource_type: Any) -> Iterator[Dict[str, Any]]: ...
    @abstractmethod
    def _update_from_config(self, result: T, partial: Dict[str, Any], validate: bool = ...) -> T: ...
    @abstractmethod
    def initial_result(self, resource_type: Any, base: Any) -> T: ...
    def calculate_node_config(
        self,
        config_call_dict: Dict[str, Any],
        fqn: Any,
        resource_type: Any,
        project_name: Any,
        base: Any,
        patch_config_dict: Optional[Dict[str, Any]] = ...,
    ) -> T: ...
    @abstractmethod
    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: Any,
        resource_type: Any,
        project_name: Any,
        base: Any,
        patch_config_dict: Optional[Dict[str, Any]] = ...,
    ) -> T: ...

class ContextConfigGenerator(BaseContextConfigGenerator[C]):
    _active_project: Any
    def __init__(self, active_project: Any) -> None: ...
    def get_config_source(self, project: Any) -> RenderedConfig: ...
    def initial_result(self, resource_type: Any, base: Any) -> C: ...
    def _update_from_config(self, result: C, partial: Dict[str, Any], validate: bool = ...) -> C: ...
    def translate_hook_names(self, project_dict: Dict[str, Any]) -> Dict[str, Any]: ...
    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: Any,
        resource_type: Any,
        project_name: Any,
        base: Any,
        patch_config_dict: Optional[Dict[str, Any]] = ...,
    ) -> Dict[str, Any]: ...

class UnrenderedConfigGenerator(BaseContextConfigGenerator[Dict[str, Any]]):
    def get_config_source(self, project: Any) -> UnrenderedConfig: ...
    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: Any,
        resource_type: Any,
        project_name: Any,
        base: Any,
        patch_config_dict: Optional[Dict[str, Any]] = ...,
    ) -> Dict[str, Any]: ...
    def initial_result(self, resource_type: Any, base: Any) -> Dict[str, Any]: ...
    def _update_from_config(self, result: Dict[str, Any], partial: Dict[str, Any], validate: bool = ...) -> Dict[str, Any]: ...

class ContextConfig:
    _config_call_dict: Dict[str, Any]
    _unrendered_config_call_dict: Dict[str, Any]
    _active_project: Any
    _fqn: Any
    _resource_type: Any
    _project_name: Any
    def __init__(self, active_project: Any, fqn: Any, resource_type: Any, project_name: Any) -> None: ...
    def add_config_call(self, opts: Dict[str, Any]) -> None: ...
    def add_unrendered_config_call(self, opts: Dict[str, Any]) -> None: ...
    def build_config_dict(
        self,
        base: bool = ...,
        *,
        rendered: bool = ...,
        patch_config_dict: Optional[Dict[str, Any]] = ...,
    ) -> Dict[str, Any]: ...