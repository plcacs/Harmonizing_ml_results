import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from typing import Any, ContextManager, Dict, FrozenSet, Iterator, Optional, Tuple

import yaml
import dbt.config
import dbt.exceptions
from dbt import flags
from dbt.constants import PACKAGES_FILE_NAME
from dbt.flags import set_from_args
from tests.unit.utils import normalize

INITIAL_ROOT: str = os.getcwd()


@contextmanager
def temp_cd(path: str) -> Iterator[None]:
    current_path: str = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current_path)


@contextmanager
def raises_nothing() -> Iterator[None]:
    yield


def empty_profile_renderer() -> Any:
    return dbt.config.renderer.ProfileRenderer({})


def empty_project_renderer() -> Any:
    return dbt.config.renderer.DbtProjectYamlRenderer()


model_config: Dict[str, Any] = {
    'my_package_name': {
        'enabled': True,
        'adwords': {
            'adwords_ads': {
                'materialized': 'table',
                'enabled': True,
                'schema': 'analytics'
            }
        },
        'snowplow': {
            'snowplow_sessions': {
                'sort': 'timestamp',
                'materialized': 'incremental',
                'dist': 'user_id',
                'unique_key': 'id'
            },
            'base': {
                'snowplow_events': {
                    'sort': ['timestamp', 'userid'],
                    'materialized': 'table',
                    'sort_type': 'interleaved',
                    'dist': 'userid'
                }
            }
        }
    }
}
model_fqns: FrozenSet[Tuple[str, ...]] = frozenset((
    ('my_package_name', 'snowplow', 'snowplow_sessions'),
    ('my_package_name', 'snowplow', 'base', 'snowplow_events'),
    ('my_package_name', 'adwords', 'adwords_ads')
))


class Args:
    def __init__(
        self,
        profiles_dir: Optional[str] = None,
        threads: Optional[int] = None,
        profile: Optional[str] = None,
        cli_vars: Optional[Dict[str, Any]] = None,
        version_check: Optional[bool] = None,
        project_dir: Optional[str] = None,
        target: Optional[str] = None
    ) -> None:
        self.profile: Optional[str] = profile
        self.threads: Optional[int] = threads
        self.target: Optional[str] = target
        if profiles_dir is not None:
            self.profiles_dir: str = profiles_dir
            flags.PROFILES_DIR = profiles_dir
        if cli_vars is not None:
            self.vars: Dict[str, Any] = cli_vars
        if version_check is not None:
            self.version_check: bool = version_check
        if project_dir is not None:
            self.project_dir: str = project_dir


class BaseConfigTest(unittest.TestCase):
    """Subclass this, and before calling the superclass setUp, set
    self.profiles_dir and self.project_dir.
    """

    def setUp(self) -> None:
        self.project_dir: str = normalize(tempfile.mkdtemp())
        self.default_project_data: Dict[str, Any] = {
            'version': '0.0.1',
            'name': 'my_test_project',
            'profile': 'default'
        }
        self.write_project(self.default_project_data)
        self.profiles_dir: str = normalize(tempfile.mkdtemp())
        self.default_profile_data: Dict[str, Any] = {
            'default': {
                'outputs': {
                    'postgres': {
                        'type': 'postgres',
                        'host': 'postgres-db-hostname',
                        'port': 5555,
                        'user': 'db_user',
                        'pass': 'db_pass',
                        'dbname': 'postgres-db-name',
                        'schema': 'postgres-schema',
                        'threads': 7
                    },
                    'with-vars': {
                        'type': "{{ env_var('env_value_type') }}",
                        'host': "{{ env_var('env_value_host') }}",
                        'port': "{{ env_var('env_value_port') | as_number }}",
                        'user': "{{ env_var('env_value_user') }}",
                        'pass': "{{ env_var('env_value_pass') }}",
                        'dbname': "{{ env_var('env_value_dbname') }}",
                        'schema': "{{ env_var('env_value_schema') }}"
                    },
                    'cli-and-env-vars': {
                        'type': "{{ env_var('env_value_type') }}",
                        'host': "{{ var('cli_value_host') }}",
                        'port': "{{ env_var('env_value_port') | as_number }}",
                        'user': "{{ env_var('env_value_user') }}",
                        'pass': "{{ env_var('env_value_pass') }}",
                        'dbname': "{{ env_var('env_value_dbname') }}",
                        'schema': "{{ env_var('env_value_schema') }}"
                    }
                },
                'target': 'postgres'
            },
            'other': {
                'outputs': {
                    'other-postgres': {
                        'type': 'postgres',
                        'host': 'other-postgres-db-hostname',
                        'port': 4444,
                        'user': 'other_db_user',
                        'pass': 'other_db_pass',
                        'dbname': 'other-postgres-db-name',
                        'schema': 'other-postgres-schema',
                        'threads': 2
                    }
                },
                'target': 'other-postgres'
            },
            'empty_profile_data': {}
        }
        self.write_profile(self.default_profile_data)
        self.args: Namespace = Namespace(
            profiles_dir=self.profiles_dir,
            cli_vars={},
            version_check=True,
            project_dir=self.project_dir,
            target=None,
            threads=None,
            profile=None
        )
        set_from_args(self.args, None)
        self.env_override: Dict[str, str] = {
            'env_value_type': 'postgres',
            'env_value_host': 'env-postgres-host',
            'env_value_port': '6543',
            'env_value_user': 'env-postgres-user',
            'env_value_pass': 'env-postgres-pass',
            'env_value_dbname': 'env-postgres-dbname',
            'env_value_schema': 'env-postgres-schema',
            'env_value_profile': 'default'
        }

    def assertRaisesOrReturns(
        self,
        exc: Optional[type[BaseException]]
    ) -> ContextManager[Any]:
        if exc is None:
            return raises_nothing()
        else:
            return self.assertRaises(exc)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.project_dir)
        except EnvironmentError:
            pass
        try:
            shutil.rmtree(self.profiles_dir)
        except EnvironmentError:
            pass

    def project_path(self, name: str) -> str:
        return os.path.join(self.project_dir, name)

    def profile_path(self, name: str) -> str:
        return os.path.join(self.profiles_dir, name)

    def write_project(self, project_data: Optional[Dict[str, Any]] = None) -> None:
        if project_data is None:
            project_data = self.project_data  # type: ignore[attr-defined]
        with open(self.project_path('dbt_project.yml'), 'w') as fp:
            yaml.dump(project_data, fp)

    def write_packages(self, package_data: Dict[str, Any]) -> None:
        with open(self.project_path('packages.yml'), 'w') as fp:
            yaml.dump(package_data, fp)

    def write_profile(self, profile_data: Optional[Dict[str, Any]] = None) -> None:
        if profile_data is None:
            profile_data = self.profile_data  # type: ignore[attr-defined]
        with open(self.profile_path('profiles.yml'), 'w') as fp:
            yaml.dump(profile_data, fp)

    def write_empty_profile(self) -> None:
        with open(self.profile_path('profiles.yml'), 'w') as fp:
            yaml.dump('', fp)


def project_from_config_norender(
    cfg: Dict[str, Any],
    packages: Optional[Dict[str, Any]] = None,
    project_root: str = '/invalid-root-path',
    verify_version: bool = False
) -> Any:
    if packages is None:
        packages = {}
    partial = dbt.config.project.PartialProject.from_dicts(
        project_root,
        project_dict=cfg,
        packages_dict=packages,
        selectors_dict={},
        verify_version=verify_version
    )
    partial.project_dict['project-root'] = project_root
    rendered = dbt.config.project.RenderComponents(
        project_dict=partial.project_dict,
        packages_dict=partial.packages_dict,
        selectors_dict=partial.selectors_dict
    )
    return partial.create_project(rendered)


def project_from_config_rendered(
    cfg: Dict[str, Any],
    packages: Optional[Dict[str, Any]] = None,
    project_root: str = '/invalid-root-path',
    verify_version: bool = False,
    packages_specified_path: str = PACKAGES_FILE_NAME
) -> Any:
    if packages is None:
        packages = {}
    partial = dbt.config.project.PartialProject.from_dicts(
        project_root,
        project_dict=cfg,
        packages_dict=packages,
        selectors_dict={},
        verify_version=verify_version,
        packages_specified_path=packages_specified_path
    )
    return partial.render(empty_project_renderer())