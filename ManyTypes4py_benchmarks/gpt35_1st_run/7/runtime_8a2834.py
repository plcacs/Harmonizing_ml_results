from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableSet, Optional, Tuple, Type
import pytz
from dbt import tracking
from dbt.adapters.contracts.connection import AdapterRequiredConfig, Credentials, HasCredentials
from dbt.adapters.contracts.relation import ComponentName
from dbt.adapters.factory import get_include_paths, get_relation_class_by_name
from dbt.config.project import load_raw_project
from dbt.contracts.graph.manifest import ManifestMetadata
from dbt.contracts.project import Configuration
from dbt.exceptions import ConfigContractBrokenError, DbtProjectError, DbtRuntimeError, NonUniquePackageNameError, UninstalledPackagesFoundError
from dbt_common.dataclass_schema import ValidationError
from dbt_common.events.functions import warn_or_error
from dbt_common.helper_types import DictDefaultEmptyStr, FQNPath, PathSet
from .profile import Profile
from .project import Project
from .renderer import DbtProjectYamlRenderer, ProfileRenderer

def load_project(project_root: str, version_check: bool, profile: Profile, cli_vars: Optional[Dict[str, Any]] = None) -> Project:
    ...

def load_profile(project_root: str, cli_vars: Dict[str, Any], profile_name_override: Optional[str] = None, target_override: Optional[str] = None, threads_override: Optional[int] = None) -> Profile:
    ...

def _project_quoting_dict(proj: Project, profile: Profile) -> Dict[ComponentName, bool]:
    ...

@dataclass
class RuntimeConfig(Project, Profile, AdapterRequiredConfig):
    dependencies: Optional[Dict[str, 'RuntimeConfig']] = None
    invoked_at: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))

    def __post_init__(self) -> None:
        ...

    @classmethod
    def get_profile(cls, project_root: str, cli_vars: Dict[str, Any], args: Any) -> Profile:
        ...

    @classmethod
    def from_parts(cls, project: Project, profile: Profile, args: Any, dependencies: Optional[Dict[str, 'RuntimeConfig']] = None) -> 'RuntimeConfig':
        ...

    def new_project(self, project_root: str) -> 'RuntimeConfig':
        ...

    def serialize(self) -> Dict[str, Any]:
        ...

    def validate(self) -> None:
        ...

    @classmethod
    def collect_parts(cls, args: Any) -> Tuple[Project, Profile]:
        ...

    @classmethod
    def from_args(cls, args: Any) -> 'RuntimeConfig':
        ...

    def get_metadata(self) -> ManifestMetadata:
        ...

    def _get_v2_config_paths(self, config: Dict[str, Any], path: Tuple[str, ...], paths: Optional[MutableSet[Tuple[str, ...]]] = None) -> frozenset:
        ...

    def _get_config_paths(self, config: Dict[str, Any], path: Tuple[str, ...] = (), paths: Optional[MutableSet[Tuple[str, ...]]] = None) -> frozenset:
        ...

    def get_resource_config_paths(self) -> Dict[str, frozenset]:
        ...

    def warn_for_unused_resource_config_paths(self, resource_fqns: Dict[str, frozenset], disabled: frozenset) -> None:
        ...

    def load_dependencies(self, base_only: bool = False) -> Dict[str, 'RuntimeConfig']:
        ...

    def clear_dependencies(self) -> None:
        ...

    def load_projects(self, paths: Iterable[str]) -> Iterator[Tuple[str, 'RuntimeConfig']]:
        ...

class UnsetCredentials(Credentials):
    ...

class UnsetProfile(Profile):
    ...

UNUSED_RESOURCE_CONFIGURATION_PATH_MESSAGE: str = 'Configuration paths exist in your dbt_project.yml file which do not apply to any resources.\nThere are {} unused configuration paths:\n{}\n'

def _is_config_used(path: Tuple[str, ...], fqns: frozenset) -> bool:
    ...
