from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

from dbt.adapters.factory import get_config_class_by_name
from dbt.config import IsFQNResource, Project
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
    def __init__(self, project: Project) -> None:
        self.project: Project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]:
        ...


class UnrenderedConfig(ConfigSource):
    def __init__(self, project: Project) -> None:
        self.project: Project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]:
        unrendered: Dict[str, Any] = self.project.unrendered.project_dict  # type: ignore[attr-defined]
        if resource_type == NodeType.Seed:
            model_configs = unrendered.get('seeds')
        elif resource_type == NodeType.Snapshot:
            model_configs = unrendered.get('snapshots')
        elif resource_type == NodeType.Source:
            model_configs = unrendered.get('sources')
        elif resource_type == NodeType.Test:
            model_configs = unrendered.get('data_tests')
        elif resource_type == NodeType.Metric:
            model_configs = unrendered.get('metrics')
        elif resource_type == NodeType.SemanticModel:
            model_configs = unrendered.get('semantic_models')
        elif resource_type == NodeType.SavedQuery:
            model_configs = unrendered.get('saved_queries')
        elif resource_type == NodeType.Exposure:
            model_configs = unrendered.get('exposures')
        elif resource_type == NodeType.Unit:
            model_configs = unrendered.get('unit_tests')
        else:
            model_configs = unrendered.get('models')
        if model_configs is None:
            return {}
        else:
            return model_configs


class RenderedConfig(ConfigSource):
    def __init__(self, project: Project) -> None:
        self.project: Project = project

    def get_config_dict(self, resource_type: NodeType) -> Dict[str, Any]:
        if resource_type == NodeType.Seed:
            model_configs = self.project.seeds  # type: ignore[attr-defined]
        elif resource_type == NodeType.Snapshot:
            model_configs = self.project.snapshots  # type: ignore[attr-defined]
        elif resource_type == NodeType.Source:
            model_configs = self.project.sources  # type: ignore[attr-defined]
        elif resource_type == NodeType.Test:
            model_configs = self.project.data_tests  # type: ignore[attr-defined]
        elif resource_type == NodeType.Metric:
            model_configs = self.project.metrics  # type: ignore[attr-defined]
        elif resource_type == NodeType.SemanticModel:
            model_configs = self.project.semantic_models  # type: ignore[attr-defined]
        elif resource_type == NodeType.SavedQuery:
            model_configs = self.project.saved_queries  # type: ignore[attr-defined]
        elif resource_type == NodeType.Exposure:
            model_configs = self.project.exposures  # type: ignore[attr-defined]
        elif resource_type == NodeType.Unit:
            model_configs = self.project.unit_tests  # type: ignore[attr-defined]
        else:
            model_configs = self.project.models  # type: ignore[attr-defined]
        return model_configs


class BaseContextConfigGenerator(Generic[T]):
    def __init__(self, active_project: Project) -> None:
        self._active_project: Project = active_project

    def get_config_source(self, project: Project) -> ConfigSource:
        return RenderedConfig(project)

    def get_node_project(self, project_name: str) -> Project:
        if project_name == self._active_project.project_name:  # type: ignore[attr-defined]
            return self._active_project
        dependencies: Dict[str, Project] = self._active_project.load_dependencies()  # type: ignore[attr-defined]
        if project_name not in dependencies:
            raise DbtInternalError(f'Project name {project_name} not found in dependencies (found {list(dependencies)})')
        return dependencies[project_name]

    def _project_configs(self, project: Project, fqn: List[str], resource_type: NodeType) -> Iterator[T]:
        src: ConfigSource = self.get_config_source(project)
        model_configs: Dict[str, Any] = src.get_config_dict(resource_type)
        for level_config in fqn_search(model_configs, fqn):
            result: Dict[str, Any] = {}
            for key, value in level_config.items():
                if key.startswith('+'):
                    result[key[1:].strip()] = deepcopy(value)
                elif not isinstance(value, dict):
                    result[key] = deepcopy(value)
            yield result  # type: ignore

    def _active_project_configs(self, fqn: List[str], resource_type: NodeType) -> Iterator[T]:
        return self._project_configs(self._active_project, fqn, resource_type)

    @abstractmethod
    def _update_from_config(self, result: T, partial: Dict[str, Any], validate: bool = False) -> T:
        ...

    @abstractmethod
    def initial_result(self, resource_type: NodeType, base: bool) -> T:
        ...

    def calculate_node_config(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> T:
        own_config: Project = self.get_node_project(project_name)
        result: T = self.initial_result(resource_type=resource_type, base=base)
        project_configs = self._project_configs(own_config, fqn, resource_type)
        for fqn_config in project_configs:
            result = self._update_from_config(result, fqn_config)
        if patch_config_dict:
            result = self._update_from_config(result, patch_config_dict)
        result = self._update_from_config(result, config_call_dict)
        if own_config.project_name != self._active_project.project_name:  # type: ignore[attr-defined]
            for fqn_config in self._active_project_configs(fqn, resource_type):
                result = self._update_from_config(result, fqn_config)
        return result

    @abstractmethod
    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> T:
        ...


class ContextConfigGenerator(BaseContextConfigGenerator[C]):
    def __init__(self, active_project: Project) -> None:
        self._active_project: Project = active_project

    def get_config_source(self, project: Project) -> RenderedConfig:
        return RenderedConfig(project)

    def initial_result(self, resource_type: NodeType, base: bool) -> C:
        config_cls = get_config_for(resource_type, base=base)
        result: C = config_cls.from_dict({})  # type: ignore
        return result

    def _update_from_config(self, result: C, partial: Dict[str, Any], validate: bool = False) -> C:
        translated: Dict[str, Any] = self._active_project.credentials.translate_aliases(partial)  # type: ignore[attr-defined]
        translated = self.translate_hook_names(translated)
        adapter_type: str = self._active_project.credentials.type  # type: ignore[attr-defined]
        adapter_config_cls = get_config_class_by_name(adapter_type)
        updated: C = result.update_from(translated, adapter_config_cls, validate=validate)
        return updated

    def translate_hook_names(self, project_dict: Dict[str, Any]) -> Dict[str, Any]:
        if 'pre_hook' in project_dict:
            project_dict['pre-hook'] = project_dict.pop('pre_hook')
        if 'post_hook' in project_dict:
            project_dict['post-hook'] = project_dict.pop('post_hook')
        return project_dict

    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config: C = self.calculate_node_config(
            config_call_dict=config_call_dict,
            fqn=fqn,
            resource_type=resource_type,
            project_name=project_name,
            base=base,
            patch_config_dict=patch_config_dict,
        )
        try:
            finalized = config.finalize_and_validate()
            return finalized.to_dict(omit_none=True)
        except ValidationError as exc:
            raise SchemaConfigError(exc, node=config) from exc


class UnrenderedConfigGenerator(BaseContextConfigGenerator[Dict[str, Any]]):
    def get_config_source(self, project: Project) -> UnrenderedConfig:
        return UnrenderedConfig(project)

    def calculate_node_config_dict(
        self,
        config_call_dict: Dict[str, Any],
        fqn: List[str],
        resource_type: NodeType,
        project_name: str,
        base: bool,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.calculate_node_config(
            config_call_dict=config_call_dict,
            fqn=fqn,
            resource_type=resource_type,
            project_name=project_name,
            base=base,
            patch_config_dict=patch_config_dict,
        )

    def initial_result(self, resource_type: NodeType, base: bool) -> Dict[str, Any]:
        return {}

    def _update_from_config(self, result: Dict[str, Any], partial: Dict[str, Any], validate: bool = False) -> Dict[str, Any]:
        translated: Dict[str, Any] = self._active_project.credentials.translate_aliases(partial)  # type: ignore[attr-defined]
        result.update(translated)
        return result


class ContextConfig:
    def __init__(self, active_project: Project, fqn: List[str], resource_type: NodeType, project_name: str) -> None:
        self._config_call_dict: Dict[str, Any] = {}
        self._unrendered_config_call_dict: Dict[str, Any] = {}
        self._active_project: Project = active_project
        self._fqn: List[str] = fqn
        self._resource_type: NodeType = resource_type
        self._project_name: str = project_name

    def add_config_call(self, opts: Dict[str, Any]) -> None:
        merge_config_dicts(self._config_call_dict, opts)

    def add_unrendered_config_call(self, opts: Dict[str, Any]) -> None:
        self._unrendered_config_call_dict.update(opts)

    def build_config_dict(
        self,
        base: bool = False,
        *,
        rendered: bool = True,
        patch_config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if rendered:
            src = ContextConfigGenerator(self._active_project)
            config_call_dict: Dict[str, Any] = self._config_call_dict
        else:
            src = UnrenderedConfigGenerator(self._active_project)
            if get_flags().state_modified_compare_more_unrendered_values is False:  # type: ignore[attr-defined]
                config_call_dict = self._config_call_dict
            elif self._config_call_dict and (not self._unrendered_config_call_dict):
                config_call_dict = self._config_call_dict
            else:
                config_call_dict = self._unrendered_config_call_dict
        return src.calculate_node_config_dict(
            config_call_dict=config_call_dict,
            fqn=self._fqn,
            resource_type=self._resource_type,
            project_name=self._project_name,
            base=base,
            patch_config_dict=patch_config_dict,
        )