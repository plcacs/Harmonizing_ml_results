import os
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Mapping
import pytest
import yaml
import dbt.flags as flags
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
def prefix() -> typing.Text:
    _randint = random.randint(0, 9999)
    _runtime_timedelta = datetime.utcnow() - datetime(1970, 1, 1, 0, 0, 0)
    _runtime = int(_runtime_timedelta.total_seconds() * 1000000.0) + _runtime_timedelta.microseconds
    prefix = f'test{_runtime}{_randint:04}'
    return prefix

@pytest.fixture(scope='class')
def unique_schema(request: str, prefix: str) -> typing.Text:
    test_file = request.module.__name__
    test_file = test_file.split('.')[-1]
    unique_schema = f'{prefix}_{test_file}'
    return unique_schema

@pytest.fixture(scope='class')
def profiles_root(tmpdir_factory: Union[typing.Callable, bool]) -> pathlib.Path:
    return tmpdir_factory.mktemp('profile')

@pytest.fixture(scope='class')
def project_root(tmpdir_factory: str) -> Union[str, pathlib.Path]:
    project_root = tmpdir_factory.mktemp('project')
    print(f'\n=== Test project_root: {project_root}')
    return project_root

@pytest.fixture(scope='session')
def shared_data_dir(request: Union[logging.LogRecord, str]) -> str:
    return os.path.join(request.config.rootdir, 'tests', 'data')

@pytest.fixture(scope='module')
def test_data_dir(request: Union[dict, str]) -> Union[str, pathlib.Path]:
    return os.path.join(request.fspath.dirname, 'data')

@pytest.fixture(scope='class')
def dbt_profile_target() -> dict[typing.Text, typing.Union[typing.Text,int]]:
    return {'type': 'postgres', 'threads': 4, 'host': 'localhost', 'port': int(os.getenv('POSTGRES_TEST_PORT', 5432)), 'user': os.getenv('POSTGRES_TEST_USER', 'root'), 'pass': os.getenv('POSTGRES_TEST_PASS', 'password'), 'dbname': os.getenv('POSTGRES_TEST_DATABASE', 'dbt')}

@pytest.fixture(scope='class')
def profile_user(dbt_profile_target: str) -> str:
    return dbt_profile_target['user']

@pytest.fixture(scope='class')
def profiles_config_update() -> dict:
    return {}

@pytest.fixture(scope='class')
def dbt_profile_data(unique_schema: Union[bool, str], dbt_profile_target: Any, profiles_config_update: dict[str, typing.Any]) -> dict[typing.Text, dict[typing.Text, typing.Union[dict[typing.Text, dict],typing.Text]]]:
    profile = {'test': {'outputs': {'default': {}}, 'target': 'default'}}
    target = dbt_profile_target
    target['schema'] = unique_schema
    profile['test']['outputs']['default'] = target
    if profiles_config_update:
        profile.update(profiles_config_update)
    return profile

@pytest.fixture(scope='class')
def profiles_yml(profiles_root: Union[str, pathlib.Path], dbt_profile_data: Union[str, dict, None]) -> typing.Generator[typing.Union[str,dict,None]]:
    os.environ['DBT_PROFILES_DIR'] = str(profiles_root)
    write_file(yaml.safe_dump(dbt_profile_data), profiles_root, 'profiles.yml')
    yield dbt_profile_data
    del os.environ['DBT_PROFILES_DIR']

@pytest.fixture(scope='class')
def project_config_update() -> dict:
    return {}

@pytest.fixture(scope='class')
def dbt_project_yml(project_root: Union[str, dict[str, typing.Any], bool], project_config_update: Union[dict[str, typing.Any], dict, bool]) -> dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, bool]]]:
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
def dependencies() -> dict:
    return {}

@pytest.fixture(scope='class')
def dependencies_yml(project_root: Union[str, typing.Sequence[str], list[str], int], dependencies: Union[waterbutler.providers.bitbuckepath.BitbucketPath, waterbutler.providers.gitlab.path.GitLabPath, Path]) -> None:
    if dependencies:
        if isinstance(dependencies, str):
            data = dependencies
        else:
            data = yaml.safe_dump(dependencies)
        write_file(data, project_root, 'dependencies.yml')

@pytest.fixture(scope='class')
def packages() -> dict:
    return {}

@pytest.fixture(scope='class')
def packages_yml(project_root: str, packages: Union[str, None, list[str]]) -> None:
    if packages:
        if isinstance(packages, str):
            data = packages
        else:
            data = yaml.safe_dump(packages)
        write_file(data, project_root, 'packages.yml')

@pytest.fixture(scope='class')
def selectors() -> dict:
    return {}

@pytest.fixture(scope='class')
def selectors_yml(project_root: str, selectors: Union[str, None, list[str]]) -> None:
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
def adapter(logs_dir: Union[bool, typing.Callable[typing.Any, None], None, str], unique_schema: Union[bool, typing.Callable[typing.Any, None], None, str], project_root: Union[str, util.photolib.workspace.Workspace, pathlib.Path, None], profiles_root: Union[str, util.photolib.workspace.Workspace, pathlib.Path, None], profiles_yml: Union[bool, typing.Callable[typing.Any, None], None, str], clean_up_logging: Union[bool, typing.Callable[typing.Any, None], None, str], dbt_project_yml: Union[bool, typing.Callable[typing.Any, None], None, str]) -> typing.Generator:
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

def write_project_files(project_root: Union[str, pathlib.Path, dict[str, str]], dir_name: Union[str, pathlib.Path, dict[str, str]], file_dict: Union[str, typing.Mapping]) -> None:
    path = project_root.mkdir(dir_name)
    if file_dict:
        write_project_files_recursively(path, file_dict)

def write_project_files_recursively(path: Union[bool, typing.Iterable[str], str, pathlib.Path], file_dict: Union[dict[str, typing.Any], argparse.Namespace]) -> None:
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
def models() -> dict:
    return {}

@pytest.fixture(scope='class')
def macros() -> dict:
    return {}

@pytest.fixture(scope='class')
def properties() -> dict:
    return {}

@pytest.fixture(scope='class')
def seeds() -> dict:
    return {}

@pytest.fixture(scope='class')
def snapshots() -> dict:
    return {}

@pytest.fixture(scope='class')
def tests() -> dict:
    return {}

@pytest.fixture(scope='class')
def analyses() -> dict:
    return {}

@pytest.fixture(scope='class')
def project_files(project_root: Union[str, dict[str, typing.Any], pathlib.Path], models: Union[str, dict[str, typing.Any], bool], macros: Union[str, dict[str, typing.Any], list[str]], snapshots: Union[str, dict[str, typing.Any], bool], properties: Union[str, dict[str, typing.Any], bool], seeds: Union[str, bool, dict[str, typing.Any]], tests: Union[str, list[str], dict[str, typing.Any]], analyses: Union[str, dict[str, typing.Any], list[str]], selectors_yml: Union[bool, pathlib.Path, sideeye.config.Configuration], dependencies_yml: Union[bool, pathlib.Path, sideeye.config.Configuration], packages_yml: Union[bool, pathlib.Path, sideeye.config.Configuration], dbt_project_yml: Union[bool, pathlib.Path, sideeye.config.Configuration]) -> None:
    write_project_files(project_root, 'models', {**models, **properties})
    write_project_files(project_root, 'macros', macros)
    write_project_files(project_root, 'snapshots', snapshots)
    write_project_files(project_root, 'seeds', seeds)
    write_project_files(project_root, 'tests', tests)
    write_project_files(project_root, 'analyses', analyses)

@pytest.fixture(scope='class')
def logs_dir(request: str, prefix: str) -> typing.Generator[str]:
    dbt_log_dir = os.path.join(request.config.rootdir, 'logs', prefix)
    os.environ['DBT_LOG_PATH'] = str(dbt_log_dir)
    yield str(Path(dbt_log_dir))
    del os.environ['DBT_LOG_PATH']

@pytest.fixture(scope='class')
def test_config() -> dict:
    return {}

class TestProjInfo:
    __test__ = False

    def __init__(self, project_root: Union[str, bool, dict], profiles_dir: Union[str, None, int], adapter_type: Union[str, pypi2nix.path.Path, dict[str, typing.Any]], test_dir: Union[str, None], shared_data_dir: Union[str, None], test_data_dir: Union[str, None], test_schema: Union[str, None, dict[str, typing.Any]], database: Union[str, bool, pathlib.Path], test_config: Union[str, pypi2nix.path.Path, dict[str, typing.Any]]) -> None:
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
    def adapter(self) -> typing.Generator:
        return get_adapter_by_type(self.adapter_type)

    def run_sql_file(self, sql_path: Union[str, path.Path], fetch: Union[None, str, bool]=None) -> None:
        with open(sql_path, 'r') as f:
            statements = f.read().split(';')
            for statement in statements:
                self.run_sql(statement, fetch)

    def run_sql(self, sql: Union[str, IConnection, None], fetch: Union[None, str, IConnection]=None):
        return run_sql_with_adapter(self.adapter, sql, fetch=fetch)

    def create_test_schema(self, schema_name: Union[None, str]=None) -> None:
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

    def get_tables_in_schema(self) -> dict[str, str]:
        sql = "\n                select table_name,\n                        case when table_type = 'BASE TABLE' then 'table'\n                             when table_type = 'VIEW' then 'view'\n                             else table_type\n                        end as materialization\n                from information_schema.tables\n                where {}\n                order by table_name\n                "
        sql = sql.format("{} ilike '{}'".format('table_schema', self.test_schema))
        result = self.run_sql(sql, fetch='all')
        return {model_name: materialization for model_name, materialization in result}

@pytest.fixture(scope='class')
def environment():
    return os.environ

@pytest.fixture(scope='class')
def initialization(environment: dict) -> None:
    set_invocation_context(environment)
    enable_test_caching()

@pytest.fixture(scope='class')
def project_setup(initialization: Union[bool, str, None], clean_up_logging: Union[bool, str, None], project_root: Union[str, bool, list[str]], profiles_root: Union[dict[str, typing.Any], bool, None], request: Union[dict[str, typing.Any], bool, None], unique_schema: Union[dict[str, typing.Any], bool, None], profiles_yml: Union[bool, str, None], adapter: Union[dict[str, typing.Any], bool, None], shared_data_dir: Union[dict[str, typing.Any], bool, None], test_data_dir: Union[dict[str, typing.Any], bool, None], logs_dir: Union[str, bool, typing.Sequence[str]], test_config: Union[dict[str, typing.Any], bool, None]) -> typing.Generator[TestProjInfo]:
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
def project(project_setup: str, project_files: str) -> str:
    return project_setup