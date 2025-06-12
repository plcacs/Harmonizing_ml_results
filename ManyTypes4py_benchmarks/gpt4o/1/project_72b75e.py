import os
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union
import pytest
import yaml
import dbt.flags as flags
from dbt.adapters.base import BaseAdapter
from dbt.adapters.factory import get_adapter, get_adapter_by_type, register_adapter, reset_adapters
from dbt.config.runtime import RuntimeConfig
from dbt.context.providers import generate_runtime_macro_context
from dbt.events.logging import setup_event_logger
from dbt.mp_context import get_mp_context
from dbt.parser.manifest import ManifestLoader
from dbt.tests.util import TestProcessingException, get_connection, run_sql_with_adapter, write_file
from dbt_common.context import set_invocation_context
from dbt_common.events.event_manager_client import cleanup_event_logger
from dbt_common.exceptions import CompilationError, DbtDatabaseError
from dbt_common.tests import enable_test_caching

@pytest.fixture(scope='class')
def prefix() -> str:
    _randint = random.randint(0, 9999)
    _runtime_timedelta = datetime.utcnow() - datetime(1970, 1, 1, 0, 0, 0)
    _runtime = int(_runtime_timedelta.total_seconds() * 1000000.0) + _runtime_timedelta.microseconds
    prefix = f'test{_runtime}{_randint:04}'
    return prefix

@pytest.fixture(scope='class')
def unique_schema(request: Any, prefix: str) -> str:
    test_file = request.module.__name__
    test_file = test_file.split('.')[-1]
    unique_schema = f'{prefix}_{test_file}'
    return unique_schema

@pytest.fixture(scope='class')
def profiles_root(tmpdir_factory: Any) -> Path:
    return tmpdir_factory.mktemp('profile')

@pytest.fixture(scope='class')
def project_root(tmpdir_factory: Any) -> Path:
    project_root = tmpdir_factory.mktemp('project')
    print(f'\n=== Test project_root: {project_root}')
    return project_root

@pytest.fixture(scope='session')
def shared_data_dir(request: Any) -> str:
    return os.path.join(request.config.rootdir, 'tests', 'data')

@pytest.fixture(scope='module')
def test_data_dir(request: Any) -> str:
    return os.path.join(request.fspath.dirname, 'data')

@pytest.fixture(scope='class')
def dbt_profile_target() -> Dict[str, Union[str, int]]:
    return {'type': 'postgres', 'threads': 4, 'host': 'localhost', 'port': int(os.getenv('POSTGRES_TEST_PORT', 5432)), 'user': os.getenv('POSTGRES_TEST_USER', 'root'), 'pass': os.getenv('POSTGRES_TEST_PASS', 'password'), 'dbname': os.getenv('POSTGRES_TEST_DATABASE', 'dbt')}

@pytest.fixture(scope='class')
def profile_user(dbt_profile_target: Dict[str, Union[str, int]]) -> str:
    return dbt_profile_target['user']

@pytest.fixture(scope='class')
def profiles_config_update() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def dbt_profile_data(unique_schema: str, dbt_profile_target: Dict[str, Union[str, int]], profiles_config_update: Dict[str, Any]) -> Dict[str, Any]:
    profile = {'test': {'outputs': {'default': {}}, 'target': 'default'}}
    target = dbt_profile_target
    target['schema'] = unique_schema
    profile['test']['outputs']['default'] = target
    if profiles_config_update:
        profile.update(profiles_config_update)
    return profile

@pytest.fixture(scope='class')
def profiles_yml(profiles_root: Path, dbt_profile_data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    os.environ['DBT_PROFILES_DIR'] = str(profiles_root)
    write_file(yaml.safe_dump(dbt_profile_data), profiles_root, 'profiles.yml')
    yield dbt_profile_data
    del os.environ['DBT_PROFILES_DIR']

@pytest.fixture(scope='class')
def project_config_update() -> Union[Dict[str, Any], str]:
    return {}

@pytest.fixture(scope='class')
def dbt_project_yml(project_root: Path, project_config_update: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    project_config = {'name': 'test', 'profile': 'test', 'flags': {'send_anonymous_usage_stats': False}}
    if project_config_update:
        if isinstance(project_config_update, dict):
            project_config.update(project_config_update)
        elif isinstance(project_config_update, str):
            updates = yaml.safe_load(project_config_update)
            project_config.update(updates)
    write_file(yaml.safe_dump(project_config), project_root, 'dbt_project.yml')
    return project_config

@pytest.fixture(scope='class')
def dependencies() -> Union[Dict[str, Any], str]:
    return {}

@pytest.fixture(scope='class')
def dependencies_yml(project_root: Path, dependencies: Union[Dict[str, Any], str]) -> None:
    if dependencies:
        if isinstance(dependencies, str):
            data = dependencies
        else:
            data = yaml.safe_dump(dependencies)
        write_file(data, project_root, 'dependencies.yml')

@pytest.fixture(scope='class')
def packages() -> Union[Dict[str, Any], str]:
    return {}

@pytest.fixture(scope='class')
def packages_yml(project_root: Path, packages: Union[Dict[str, Any], str]) -> None:
    if packages:
        if isinstance(packages, str):
            data = packages
        else:
            data = yaml.safe_dump(packages)
        write_file(data, project_root, 'packages.yml')

@pytest.fixture(scope='class')
def selectors() -> Union[Dict[str, Any], str]:
    return {}

@pytest.fixture(scope='class')
def selectors_yml(project_root: Path, selectors: Union[Dict[str, Any], str]) -> None:
    if selectors:
        if isinstance(selectors, str):
            data = selectors
        else:
            data = yaml.safe_dump(selectors)
        write_file(data, project_root, 'selectors.yml')

@pytest.fixture(scope='class')
def clean_up_logging() -> None:
    cleanup_event_logger()

@pytest.fixture(scope='class')
def adapter(logs_dir: str, unique_schema: str, project_root: Path, profiles_root: Path, profiles_yml: Dict[str, Any], clean_up_logging: None, dbt_project_yml: Dict[str, Any]) -> Generator[BaseAdapter, None, None]:
    args = Namespace(profiles_dir=str(profiles_root), project_dir=str(project_root), target=None, profile=None, threads=None)
    flags.set_from_args(args, {})
    runtime_config = RuntimeConfig.from_args(args)
    register_adapter(runtime_config, get_mp_context())
    adapter = get_adapter(runtime_config)
    manifest = ManifestLoader.load_macros(runtime_config, adapter.connections.set_query_header, base_macros_only=True)
    adapter.set_macro_resolver(manifest)
    adapter.set_macro_context_generator(generate_runtime_macro_context)
    yield adapter
    adapter.cleanup_connections()
    reset_adapters()

def write_project_files(project_root: Path, dir_name: str, file_dict: Dict[str, Any]) -> None:
    path = project_root.mkdir(dir_name)
    if file_dict:
        write_project_files_recursively(path, file_dict)

def write_project_files_recursively(path: Path, file_dict: Dict[str, Any]) -> None:
    if type(file_dict) is not dict:
        raise TestProcessingException(f"File dict is not a dict: '{file_dict}' for path '{path}'")
    suffix_list = ['.sql', '.csv', '.md', '.txt', '.py']
    for name, value in file_dict.items():
        if name.endswith('.yml') or name.endswith('.yaml'):
            if isinstance(value, str):
                data = value
            else:
                data = yaml.safe_dump(value)
            write_file(data, path, name)
        elif name.endswith(tuple(suffix_list)):
            write_file(value, path, name)
        else:
            write_project_files_recursively(path.mkdir(name), value)

@pytest.fixture(scope='class')
def models() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def macros() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def properties() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def seeds() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def snapshots() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def tests() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def analyses() -> Dict[str, Any]:
    return {}

@pytest.fixture(scope='class')
def project_files(project_root: Path, models: Dict[str, Any], macros: Dict[str, Any], snapshots: Dict[str, Any], properties: Dict[str, Any], seeds: Dict[str, Any], tests: Dict[str, Any], analyses: Dict[str, Any], selectors_yml: None, dependencies_yml: None, packages_yml: None, dbt_project_yml: Dict[str, Any]) -> None:
    write_project_files(project_root, 'models', {**models, **properties})
    write_project_files(project_root, 'macros', macros)
    write_project_files(project_root, 'snapshots', snapshots)
    write_project_files(project_root, 'seeds', seeds)
    write_project_files(project_root, 'tests', tests)
    write_project_files(project_root, 'analyses', analyses)

@pytest.fixture(scope='class')
def logs_dir(request: Any, prefix: str) -> Generator[str, None, None]:
    dbt_log_dir = os.path.join(request.config.rootdir, 'logs', prefix)
    os.environ['DBT_LOG_PATH'] = str(dbt_log_dir)
    yield str(Path(dbt_log_dir))
    del os.environ['DBT_LOG_PATH']

@pytest.fixture(scope='class')
def test_config() -> Dict[str, Any]:
    return {}

class TestProjInfo:
    __test__ = False

    def __init__(self, project_root: Path, profiles_dir: Path, adapter_type: str, test_dir: str, shared_data_dir: str, test_data_dir: str, test_schema: str, database: str, test_config: Dict[str, Any]) -> None:
        self.project_root = project_root
        self.profiles_dir = profiles_dir
        self.adapter_type = adapter_type
        self.test_dir = test_dir
        self.shared_data_dir = shared_data_dir
        self.test_data_dir = test_data_dir
        self.test_schema = test_schema
        self.database = database
        self.test_config = test_config
        self.created_schemas = []

    @property
    def adapter(self) -> BaseAdapter:
        return get_adapter_by_type(self.adapter_type)

    def run_sql_file(self, sql_path: str, fetch: Optional[str] = None) -> None:
        with open(sql_path, 'r') as f:
            statements = f.read().split(';')
            for statement in statements:
                self.run_sql(statement, fetch)

    def run_sql(self, sql: str, fetch: Optional[str] = None) -> Any:
        return run_sql_with_adapter(self.adapter, sql, fetch=fetch)

    def create_test_schema(self, schema_name: Optional[str] = None) -> None:
        if schema_name is None:
            schema_name = self.test_schema
        with get_connection(self.adapter):
            relation = self.adapter.Relation.create(database=self.database, schema=schema_name)
            self.adapter.create_schema(relation)
            self.created_schemas.append(schema_name)

    def drop_test_schema(self) -> None:
        if self.adapter.get_macro_resolver() is None:
            manifest = ManifestLoader.load_macros(self.adapter.config, self.adapter.connections.set_query_header, base_macros_only=True)
            self.adapter.set_macro_resolver(manifest)
        with get_connection(self.adapter):
            for schema_name in self.created_schemas:
                relation = self.adapter.Relation.create(database=self.database, schema=schema_name)
                self.adapter.drop_schema(relation)
            self.created_schemas = []

    def get_tables_in_schema(self) -> Dict[str, str]:
        sql = "\n                select table_name,\n                        case when table_type = 'BASE TABLE' then 'table'\n                             when table_type = 'VIEW' then 'view'\n                             else table_type\n                        end as materialization\n                from information_schema.tables\n                where {}\n                order by table_name\n                "
        sql = sql.format("{} ilike '{}'".format('table_schema', self.test_schema))
        result = self.run_sql(sql, fetch='all')
        return {model_name: materialization for model_name, materialization in result}

@pytest.fixture(scope='class')
def environment() -> Dict[str, str]:
    return os.environ

@pytest.fixture(scope='class')
def initialization(environment: Dict[str, str]) -> None:
    set_invocation_context(environment)
    enable_test_caching()

@pytest.fixture(scope='class')
def project_setup(initialization: None, clean_up_logging: None, project_root: Path, profiles_root: Path, request: Any, unique_schema: str, profiles_yml: Dict[str, Any], adapter: BaseAdapter, shared_data_dir: str, test_data_dir: str, logs_dir: str, test_config: Dict[str, Any]) -> Generator[TestProjInfo, None, None]:
    log_flags = Namespace(LOG_PATH=logs_dir, LOG_FORMAT='json', LOG_FORMAT_FILE='json', USE_COLORS=False, USE_COLORS_FILE=False, LOG_LEVEL='info', LOG_LEVEL_FILE='debug', DEBUG=False, LOG_CACHE_EVENTS=False, QUIET=False, LOG_FILE_MAX_BYTES=1000000)
    setup_event_logger(log_flags)
    orig_cwd = os.getcwd()
    os.chdir(project_root)
    project = TestProjInfo(project_root=project_root, profiles_dir=profiles_root, adapter_type=adapter.type(), test_dir=request.fspath.dirname, shared_data_dir=shared_data_dir, test_data_dir=test_data_dir, test_schema=unique_schema, database=adapter.config.credentials.database, test_config=test_config)
    project.drop_test_schema()
    project.create_test_schema()
    yield project
    try:
        project.drop_test_schema()
    except (KeyError, AttributeError, CompilationError, DbtDatabaseError):
        pass
    os.chdir(orig_cwd)
    cleanup_event_logger()

@pytest.fixture(scope='class')
def project(project_setup: TestProjInfo, project_files: None) -> TestProjInfo:
    return project_setup
