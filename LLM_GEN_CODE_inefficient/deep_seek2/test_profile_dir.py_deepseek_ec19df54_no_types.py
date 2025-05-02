import os
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import pytest
import yaml
import dbt.flags as flags
from dbt.tests.util import rm_file, run_dbt, run_dbt_and_capture, write_file

@pytest.fixture(scope='class')
def profiles_yml(profiles_root, dbt_profile_data):
    write_file(yaml.safe_dump(dbt_profile_data), profiles_root, 'profiles.yml')
    return dbt_profile_data

@pytest.fixture(scope='class')
def profiles_home_root():
    return os.path.join(os.path.expanduser('~'), '.dbt')

@pytest.fixture(scope='class')
def profiles_env_root(tmpdir_factory):
    path = tmpdir_factory.mktemp('profile_env')
    return str(path).lower()

@pytest.fixture(scope='class')
def profiles_flag_root(tmpdir_factory):
    return tmpdir_factory.mktemp('profile_flag')

@pytest.fixture(scope='class')
def profiles_project_root(project):
    return project.project_root

@pytest.fixture(scope='class')
def cwd():
    return os.getcwd()

@pytest.fixture(scope='class')
def cwd_parent(cwd):
    return os.path.dirname(cwd)

@pytest.fixture(scope='class')
def cwd_child():
    return Path(os.getcwd()) / 'macros'

@pytest.fixture
def write_profiles_yml(request):

    def _write_profiles_yml(profiles_dir, dbt_profile_contents):

        def cleanup():
            rm_file(Path(profiles_dir) / 'profiles.yml')
        request.addfinalizer(cleanup)
        write_file(yaml.safe_dump(dbt_profile_contents), profiles_dir, 'profiles.yml')
    return _write_profiles_yml

@contextmanager
def environ(env):
    original_env = {key: os.getenv(key) for key in env}
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

    def test_debug(self, project):
        run_dbt(['debug', '--profiles-dir', 'does_not_exist'], expect_pass=None)

    def test_deps(self, project):
        run_dbt(['deps', '--profiles-dir', 'does_not_exist'])

class TestProfiles:

    def dbt_debug(self, project_dir_cli_arg=None, profiles_dir_cli_arg=None):
        flags.set_from_args(Namespace(), {})
        command: List[str] = ['debug']
        if project_dir_cli_arg:
            command.extend(['--project-dir', str(project_dir_cli_arg)])
        if profiles_dir_cli_arg:
            command.extend(['--profiles-dir', str(profiles_dir_cli_arg)])
        return run_dbt_and_capture(command, expect_pass=None)

    @pytest.mark.parametrize('project_dir_cli_arg, working_directory', [(None, 'cwd'), (None, 'cwd_child'), ('cwd', 'cwd_parent')])
    def test_profiles(self, project_dir_cli_arg, working_directory, write_profiles_yml, dbt_profile_data, profiles_home_root, profiles_project_root, profiles_flag_root, profiles_env_root, request):
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