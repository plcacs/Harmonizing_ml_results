import os
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Callable, Generator, Optional, Tuple, Union

import pytest
import yaml
import dbt.flags as flags
from dbt.tests.util import rm_file, run_dbt, run_dbt_and_capture, write_file


@pytest.fixture(scope='class')
def profiles_yml(profiles_root: str, dbt_profile_data: Dict[str, Any]) -> Dict[str, Any]:
    write_file(yaml.safe_dump(dbt_profile_data), profiles_root, 'profiles.yml')
    return dbt_profile_data


@pytest.fixture(scope='class')
def profiles_home_root() -> str:
    return os.path.join(os.path.expanduser('~'), '.dbt')


@pytest.fixture(scope='class')
def profiles_env_root(tmpdir_factory: pytest.TempPathFactory) -> str:
    path = tmpdir_factory.mktemp('profile_env')
    return str(path).lower()


@pytest.fixture(scope='class')
def profiles_flag_root(tmpdir_factory: pytest.TempPathFactory) -> Any:
    return tmpdir_factory.mktemp('profile_flag')


@pytest.fixture(scope='class')
def profiles_project_root(project: Any) -> str:
    return project.project_root


@pytest.fixture(scope='class')
def cwd() -> str:
    return os.getcwd()


@pytest.fixture(scope='class')
def cwd_parent(cwd: str) -> str:
    return os.path.dirname(cwd)


@pytest.fixture(scope='class')
def cwd_child() -> Path:
    return Path(os.getcwd()) / 'macros'


@pytest.fixture
def write_profiles_yml(request: pytest.FixtureRequest) -> Callable[[str, Dict[str, Any]], None]:
    def _write_profiles_yml(profiles_dir: str, dbt_profile_contents: Dict[str, Any]) -> None:
        def cleanup() -> None:
            rm_file(Path(profiles_dir) / 'profiles.yml')
        request.addfinalizer(cleanup)
        write_file(yaml.safe_dump(dbt_profile_contents), profiles_dir, 'profiles.yml')
    return _write_profiles_yml


@contextmanager
def environ(env: Dict[str, str]) -> Generator[None, None, None]:
    """Temporarily set environment variables inside the context manager and
    fully restore previous environment afterwards
    """
    original_env: Dict[str, Optional[str]] = {key: os.getenv(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


class TestProfilesMayNotExist:
    def test_debug(self, project: Any) -> None:
        run_dbt(['debug', '--profiles-dir', 'does_not_exist'], expect_pass=None)

    def test_deps(self, project: Any) -> None:
        run_dbt(['deps', '--profiles-dir', 'does_not_exist'])


class TestProfiles:
    def dbt_debug(self, project_dir_cli_arg: Optional[str] = None, 
                 profiles_dir_cli_arg: Optional[str] = None) -> Tuple[int, str]:
        flags.set_from_args(Namespace(), {})
        command: list[str] = ['debug']
        if project_dir_cli_arg:
            command.extend(['--project-dir', str(project_dir_cli_arg)])
        if profiles_dir_cli_arg:
            command.extend(['--profiles-dir', str(profiles_dir_cli_arg)])
        return run_dbt_and_capture(command, expect_pass=None)

    @pytest.mark.parametrize('project_dir_cli_arg, working_directory', [(None, 'cwd'), (None, 'cwd_child'), ('cwd', 'cwd_parent')])
    def test_profiles(self, project_dir_cli_arg: Optional[str], working_directory: Optional[str], 
                     write_profiles_yml: Callable[[str, Dict[str, Any]], None], 
                     dbt_profile_data: Dict[str, Any], profiles_home_root: str, 
                     profiles_project_root: str, profiles_flag_root: Any, 
                     profiles_env_root: str, request: pytest.FixtureRequest) -> None:
        """Verify priority order to search for profiles.yml configuration.

        Reverse priority order:
        1. HOME directory
        2. DBT_PROFILES_DIR environment variable
        3. --profiles-dir command-line argument

        Specification later in this list will take priority over earlier ones, even when both are provided.
        """
        if project_dir_cli_arg is not None:
            project_dir_cli_arg = request.getfixturevalue(project_dir_cli_arg)
        if working_directory is not None:
            working_directory = request.getfixturevalue(working_directory)
        if working_directory is not None:
            os.chdir(working_directory)
        _, stdout = self.dbt_debug(project_dir_cli_arg)
        assert f'Using profiles.yml file at {profiles_home_root}' in stdout
        env_vars = {'DBT_PROFILES_DIR': profiles_env_root}
        with environ(env_vars):
            _, stdout = self.dbt_debug(project_dir_cli_arg)
            assert f'Using profiles.yml file at {profiles_env_root}' in stdout
            _, stdout = self.dbt_debug(project_dir_cli_arg, profiles_dir_cli_arg=profiles_flag_root)
            assert f'Using profiles.yml file at {profiles_flag_root}' in stdout
