import itertools
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableSet, Optional, Tuple, Type
import pytz
from dbt import tracking
from dbt.adapters.contracts.connection import AdapterRequiredConfig, Credentials, HasCredentials
from dbt.adapters.contracts.relation import ComponentName
from dbt.adapters.factory import get_include_paths, get_relation_class_by_name
from dbt.config.project import load_raw_project
from dbt.contracts.graph.manifest import ManifestMetadata
from dbt.contracts.project import Configuration
from dbt.events.types import UnusedResourceConfigPath
from dbt.exceptions import ConfigContractBrokenError, DbtProjectError, DbtRuntimeError, NonUniquePackageNameError, UninstalledPackagesFoundError
from dbt.flags import get_flags
from dbt_common.dataclass_schema import ValidationError
from dbt_common.events.functions import warn_or_error
from dbt_common.helper_types import DictDefaultEmptyStr, FQNPath, PathSet
from .profile import Profile
from .project import Project
from .renderer import DbtProjectYamlRenderer, ProfileRenderer

def load_project(project_root: Union[str, bool, pypi2nix.path.Path], version_check: Union[str, bool, pypi2nix.path.Path], profile: Union[str, bool], cli_vars: Union[None, str, bool]=None):
    project_renderer = DbtProjectYamlRenderer(profile, cli_vars)
    project = Project.from_project_root(project_root, project_renderer, verify_version=version_check)
    project.project_env_vars = project_renderer.ctx_obj.env_vars
    return project

def load_profile(project_root: Union[str, list[str], annofabcli.common.dataclasses.WaitOptions, None], cli_vars: Union[str, None, bool], profile_name_override: Union[None, str]=None, target_override: Union[None, str]=None, threads_override: Union[None, str]=None):
    raw_project = load_raw_project(project_root)
    raw_profile_name = raw_project.get('profile')
    profile_renderer = ProfileRenderer(cli_vars)
    profile_name = profile_renderer.render_value(raw_profile_name)
    profile = Profile.render(profile_renderer, profile_name, profile_name_override, target_override, threads_override)
    profile.profile_env_vars = profile_renderer.ctx_obj.env_vars
    return profile

def _project_quoting_dict(proj: str, profile: str) -> dict[, bool]:
    src = profile.credentials.translate_aliases(proj.quoting)
    result = {}
    for key in ComponentName:
        if key in src:
            value = src[key]
            if isinstance(value, bool):
                result[key] = value
    return result

@dataclass
class RuntimeConfig(Project, Profile, AdapterRequiredConfig):
    dependencies = None
    invoked_at = field(default_factory=lambda: datetime.now(pytz.UTC))

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def get_profile(cls: Union[str, dict[str, typing.Any], None], project_root: Union[str, typing.Callable], cli_vars: Union[str, typing.Callable], args: Any):
        return load_profile(project_root, cli_vars, args.profile, args.target, args.threads)

    @classmethod
    def from_parts(cls: Union[typing.Mapping, None, list[str], bob_emploi.frontend.api.project_pb2.Project], project: Union[dict[str, dict[str, typing.Any]], dict, bool, None], profile: Union[dict[str, dict[str, typing.Any]], dict, bool, None], args: Any, dependencies: Union[None, typing.Mapping, list[str], bob_emploi.frontend.api.project_pb2.Project]=None):
        """Instantiate a RuntimeConfig from its components.

        :param profile: A parsed dbt Profile.
        :param project: A parsed dbt Project.
        :param args: The parsed command-line arguments.
        :returns RuntimeConfig: The new configuration.
        """
        quoting = get_relation_class_by_name(profile.credentials.type).get_default_quote_policy().replace_dict(_project_quoting_dict(project, profile)).to_dict(omit_none=True)
        cli_vars = getattr(args, 'vars', {})
        log_cache_events = getattr(args, 'log_cache_events', profile.log_cache_events)
        return cls(project_name=project.project_name, version=project.version, project_root=project.project_root, model_paths=project.model_paths, macro_paths=project.macro_paths, seed_paths=project.seed_paths, test_paths=project.test_paths, analysis_paths=project.analysis_paths, docs_paths=project.docs_paths, asset_paths=project.asset_paths, target_path=project.target_path, snapshot_paths=project.snapshot_paths, clean_targets=project.clean_targets, log_path=project.log_path, packages_install_path=project.packages_install_path, packages_specified_path=project.packages_specified_path, quoting=quoting, models=project.models, on_run_start=project.on_run_start, on_run_end=project.on_run_end, dispatch=project.dispatch, seeds=project.seeds, snapshots=project.snapshots, dbt_version=project.dbt_version, packages=project.packages, manifest_selectors=project.manifest_selectors, selectors=project.selectors, query_comment=project.query_comment, sources=project.sources, data_tests=project.data_tests, unit_tests=project.unit_tests, metrics=project.metrics, semantic_models=project.semantic_models, saved_queries=project.saved_queries, exposures=project.exposures, vars=project.vars, config_version=project.config_version, unrendered=project.unrendered, project_env_vars=project.project_env_vars, restrict_access=project.restrict_access, profile_env_vars=profile.profile_env_vars, profile_name=profile.profile_name, target_name=profile.target_name, threads=profile.threads, credentials=profile.credentials, args=args, cli_vars=cli_vars, log_cache_events=log_cache_events, dependencies=dependencies, dbt_cloud=project.dbt_cloud, flags=project.flags)

    def new_project(self, project_root: Union[str, list[str], path.Path]):
        """Given a new project root, read in its project dictionary, supply the
        existing project's profile info, and create a new project file.

        :param project_root: A filepath to a dbt project.
        :raises DbtProfileError: If the profile is invalid.
        :raises DbtProjectError: If project is missing or invalid.
        :returns: The new configuration.
        """
        profile = Profile(**self.to_profile_info())
        profile.validate()
        renderer = DbtProjectYamlRenderer(profile)
        project = Project.from_project_root(project_root, renderer, verify_version=bool(getattr(self.args, 'VERSION_CHECK', True)))
        runtime_config = self.from_parts(project=project, profile=profile, args=deepcopy(self.args))
        runtime_config.quoting = deepcopy(self.quoting)
        return runtime_config

    def serialize(self):
        """Serialize the full configuration to a single dictionary. For any
        instance that has passed validate() (which happens in __init__), it
        matches the Configuration contract.

        Note that args are not serialized.

        :returns dict: The serialized configuration.
        """
        result = self.to_project_config(with_packages=True)
        result.update(self.to_profile_info(serialize_credentials=True))
        result['cli_vars'] = deepcopy(self.cli_vars)
        return result

    def validate(self) -> None:
        """Validate the configuration against its contract.

        :raises DbtProjectError: If the configuration fails validation.
        """
        try:
            Configuration.validate(self.serialize())
        except ValidationError as e:
            raise ConfigContractBrokenError(e) from e

    @classmethod
    def collect_parts(cls: Union[str, typing.Type], args: Any) -> tuple:
        project_root = args.project_dir if args.project_dir else os.getcwd()
        cli_vars = getattr(args, 'vars', {})
        profile = cls.get_profile(project_root, cli_vars, args)
        flags = get_flags()
        project = load_project(project_root, bool(flags.VERSION_CHECK), profile, cli_vars)
        return (project, profile)

    @classmethod
    def from_args(cls: Union[dict[str, typing.Any], typing.Callable, None], args: Any):
        """Given arguments, read in dbt_project.yml from the current directory,
        read in packages.yml if it exists, and use them to find the profile to
        load.

        :param args: The arguments as parsed from the cli.
        :raises DbtProjectError: If the project is invalid or missing.
        :raises DbtProfileError: If the profile is invalid or missing.
        :raises DbtValidationError: If the cli variables are invalid.
        """
        project, profile = cls.collect_parts(args)
        return cls.from_parts(project=project, profile=profile, args=args)

    def get_metadata(self) -> ManifestMetadata:
        return ManifestMetadata(project_name=self.project_name, project_id=self.hashed_name(), user_id=tracking.active_user.id if tracking.active_user else None, send_anonymous_usage_stats=get_flags().SEND_ANONYMOUS_USAGE_STATS if tracking.active_user else None, adapter_type=self.credentials.type)

    def _get_v2_config_paths(self, config: Union[dict[str, typing.Any], dict[str, str]], path: pathlib.Path, paths: norfs.fs.base.Path) -> set:
        for key, value in config.items():
            if isinstance(value, dict) and (not key.startswith('+')):
                self._get_config_paths(value, path + (key,), paths)
            else:
                paths.add(path)
        return frozenset(paths)

    def _get_config_paths(self, config: Union[dict[str, typing.Any], dict[str, str]], path: tuple=() -> set, paths=None):
        if paths is None:
            paths = set()
        for key, value in config.items():
            if isinstance(value, dict) and (not key.startswith('+')):
                self._get_v2_config_paths(value, path + (key,), paths)
            else:
                paths.add(path)
        return frozenset(paths)

    def get_resource_config_paths(self) -> dict[typing.Text, ]:
        """Return a dictionary with resource type keys whose values are
        lists of lists of strings, where each inner list of strings represents
        a configured path in the resource.
        """
        return {'models': self._get_config_paths(self.models), 'seeds': self._get_config_paths(self.seeds), 'snapshots': self._get_config_paths(self.snapshots), 'sources': self._get_config_paths(self.sources), 'data_tests': self._get_config_paths(self.data_tests), 'unit_tests': self._get_config_paths(self.unit_tests), 'metrics': self._get_config_paths(self.metrics), 'semantic_models': self._get_config_paths(self.semantic_models), 'saved_queries': self._get_config_paths(self.saved_queries), 'exposures': self._get_config_paths(self.exposures)}

    def warn_for_unused_resource_config_paths(self, resource_fqns: Union[str, typing.Mapping, bool], disabled: Union[tuple[str], dict[str, set[str]]]) -> None:
        """Return a list of lists of strings, where each inner list of strings
        represents a type + FQN path of a resource configuration that is not
        used.
        """
        disabled_fqns = frozenset((tuple(fqn) for fqn in disabled))
        resource_config_paths = self.get_resource_config_paths()
        unused_resource_config_paths = []
        for resource_type, config_paths in resource_config_paths.items():
            used_fqns = resource_fqns.get(resource_type, frozenset())
            fqns = used_fqns | disabled_fqns
            for config_path in config_paths:
                if not _is_config_used(config_path, fqns):
                    resource_path = '.'.join((i for i in (resource_type,) + config_path))
                    unused_resource_config_paths.append(resource_path)
        if len(unused_resource_config_paths) == 0:
            return
        warn_or_error(UnusedResourceConfigPath(unused_config_paths=unused_resource_config_paths))

    def load_dependencies(self, base_only: bool=False) -> Union[tuple[typing.Hashable], set[str], dict[, RuntimeConfig]]:
        if self.dependencies is None:
            all_projects = {self.project_name: self}
            internal_packages = get_include_paths(self.credentials.type)
            if base_only:
                project_paths = itertools.chain(internal_packages)
            else:
                count_packages_specified = len(self.packages.packages)
                count_packages_installed = len(tuple(self._get_project_directories()))
                if count_packages_specified > count_packages_installed:
                    raise UninstalledPackagesFoundError(count_packages_specified, count_packages_installed, self.packages_specified_path, self.packages_install_path)
                project_paths = itertools.chain(internal_packages, self._get_project_directories())
            for project_name, project in self.load_projects(project_paths):
                if project_name in all_projects:
                    raise NonUniquePackageNameError(project_name)
                all_projects[project_name] = project
            self.dependencies = all_projects
        return self.dependencies

    def clear_dependencies(self) -> None:
        self.dependencies = None

    def load_projects(self, paths: str) -> typing.Generator[tuple[typing.Union[str,list[str],Project]]]:
        for path in paths:
            try:
                project = self.new_project(str(path))
            except DbtProjectError as e:
                raise DbtProjectError(f'Failed to read package: {e}', result_type='invalid_project', path=path) from e
            else:
                yield (project.project_name, project)

    def _get_project_directories(self) -> typing.Generator:
        root = Path(self.project_root) / self.packages_install_path
        if root.exists():
            for path in root.iterdir():
                if path.is_dir() and (not path.name.startswith('__')):
                    yield path

class UnsetCredentials(Credentials):

    def __init__(self) -> None:
        super().__init__('', '')

    @property
    def type(self) -> None:
        return None

    @property
    def unique_field(self) -> None:
        return None

    def connection_info(self, *args, **kwargs) -> dict:
        return {}

    def _connection_keys(self) -> tuple:
        return ()

class UnsetProfile(Profile):

    def __init__(self) -> None:
        self.credentials = UnsetCredentials()
        self.profile_name = ''
        self.target_name = ''
        self.threads = -1

    def to_target_dict(self) -> DictDefaultEmptyStr:
        return DictDefaultEmptyStr({})

    def __getattribute__(self, name: str):
        if name in {'profile_name', 'target_name', 'threads'}:
            raise DbtRuntimeError(f'Error: disallowed attribute "{name}" - no profile!')
        return Profile.__getattribute__(self, name)
UNUSED_RESOURCE_CONFIGURATION_PATH_MESSAGE = 'Configuration paths exist in your dbt_project.yml file which do not apply to any resources.\nThere are {} unused configuration paths:\n{}\n'

def _is_config_used(path: Union[list[str], list[pathlib.Path], Path], fqns: Union[str, pathlib.Path]) -> bool:
    if fqns:
        for fqn in fqns:
            if len(path) <= len(fqn) and fqn[:len(path)] == path:
                return True
    return False