from pathlib import Path
from shutil import rmtree
from tarfile import TarFile
import tarfile
import textwrap
from typing import Any, Optional, Union, List

import filecmp
import shutil
import toml
import yaml
from click import ClickException
from click.testing import CliRunner, Result
from kedro.framework.cli.micropkg import _get_sdist_name, safe_extract
from kedro.framework.project import settings
from unittest.mock import Mock

PIPELINE_NAME: str = 'my_pipeline'


def call_pipeline_create(cli: Any, metadata: Any, pipeline_name: str = PIPELINE_NAME) -> None:
    result: Result = CliRunner().invoke(cli, ['pipeline', 'create', pipeline_name], obj=metadata)
    assert result.exit_code == 0


def call_micropkg_package(cli: Any, metadata: Any, alias: Optional[str] = None,
                          destination: Optional[Union[str, Path]] = None,
                          pipeline_name: str = PIPELINE_NAME) -> None:
    options: List[str] = ['--alias', alias] if alias else []
    options += ['--destination', str(destination)] if destination else []
    result: Result = CliRunner().invoke(cli, ['micropkg', 'package', f'pipelines.{pipeline_name}', *options], obj=metadata)
    assert result.exit_code == 0, result.output


def call_pipeline_delete(cli: Any, metadata: Any, pipeline_name: str = PIPELINE_NAME) -> None:
    result: Result = CliRunner().invoke(cli, ['pipeline', 'delete', '-y', pipeline_name], obj=metadata)
    assert result.exit_code == 0


import pytest


@pytest.fixture
def temp_dir_with_context_manager(tmp_path: Path) -> Any:
    mock_temp_dir: Any = Mock()
    mock_temp_dir.__enter__ = Mock(return_value=tmp_path)
    mock_temp_dir.__exit__ = Mock(return_value=None)
    return mock_temp_dir


@pytest.mark.usefixtures('chdir_to_dummy_project', 'cleanup_dist')
class TestMicropkgPullCommand:

    def assert_package_files_exist(self, source_path: Path) -> None:
        assert {f.name for f in source_path.iterdir()} == {'__init__.py', 'nodes.py', 'pipeline.py'}

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias, destination', [(None, None), ('aliased', None), ('aliased', 'pipelines'), (None, 'pipelines')])
    def test_pull_local_sdist(self, fake_project_cli: Any, fake_repo_path: Path, fake_package_path: Path,
                              env: Optional[str], alias: Optional[str], fake_metadata: Any) -> None:
        """Test for pulling a valid sdist file locally."""
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        call_pipeline_delete(fake_project_cli, fake_metadata)
        source_path: Path = fake_package_path / 'pipelines' / PIPELINE_NAME
        config_path: Path = fake_repo_path / settings.CONF_SOURCE / 'base' / 'pipelines' / PIPELINE_NAME
        test_path: Path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        assert not source_path.exists()
        assert not test_path.exists()
        assert not config_path.exists()
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        options: List[str] = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        options += ['--destination', destination] if destination else []
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), *options], obj=fake_metadata)
        assert result.exit_code == 0, result.output
        assert 'pulled and unpacked' in result.output
        pipeline_name: str = alias or PIPELINE_NAME
        destination = destination or Path()
        source_dest: Path = fake_package_path / destination / pipeline_name
        test_dest: Path = fake_repo_path / 'tests' / destination / pipeline_name
        config_env: str = env or 'base'
        params_config: Path = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert params_config.is_file()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias, destination', [(None, None), ('aliased', None), ('aliased', 'pipelines'), (None, 'pipelines')])
    def test_pull_local_sdist_compare(self, fake_project_cli: Any, fake_repo_path: Path, fake_package_path: Path,
                                      env: Optional[str], alias: Optional[str], destination: Optional[str],
                                      fake_metadata: Any) -> None:
        """Test for pulling a valid sdist file locally, unpack it
        into another location and check that unpacked files
        are identical to the ones in the original modular pipeline.
        """
        pipeline_name_local: str = 'another_pipeline'
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata, alias=pipeline_name_local)
        source_path: Path = fake_package_path / 'pipelines' / PIPELINE_NAME
        test_path: Path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        source_params_config: Path = fake_repo_path / settings.CONF_SOURCE / 'base' / f'parameters_{PIPELINE_NAME}.yml'
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=pipeline_name_local, version='0.1')
        assert sdist_file.is_file()
        options: List[str] = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        options += ['--destination', destination] if destination else []
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), *options], obj=fake_metadata)
        assert result.exit_code == 0, result.output
        assert 'pulled and unpacked' in result.output
        pipeline_name_final: str = alias or pipeline_name_local
        destination = destination or Path()
        source_dest: Path = fake_package_path / destination / pipeline_name_final
        test_dest: Path = fake_repo_path / 'tests' / destination / pipeline_name_final
        config_env: str = env or 'base'
        dest_params_config: Path = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name_final}.yml'
        assert not filecmp.dircmp(source_path, source_dest).diff_files
        assert not filecmp.dircmp(test_path, test_dest).diff_files
        assert source_params_config.read_bytes() == dest_params_config.read_bytes()

    def test_micropkg_pull_same_alias_package_name(self, fake_project_cli: Any, fake_repo_path: Path,
                                                   fake_package_path: Path, fake_metadata: Any) -> None:
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        pipeline_name: str = PIPELINE_NAME
        destination: str = 'tools'
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), '--destination', destination, '--alias', pipeline_name], obj=fake_metadata)
        assert result.exit_code == 0, result.stderr
        assert 'pulled and unpacked' in result.output
        source_dest: Path = fake_package_path / destination / pipeline_name
        test_dest: Path = fake_repo_path / 'tests' / destination / pipeline_name
        config_env: str = 'base'
        params_config: Path = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert params_config.is_file()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    def test_micropkg_pull_nested_destination(self, fake_project_cli: Any, fake_repo_path: Path,
                                                fake_package_path: Path, fake_metadata: Any) -> None:
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        pipeline_name: str = PIPELINE_NAME
        destination: str = 'pipelines/nested'
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), '--destination', destination, '--alias', pipeline_name], obj=fake_metadata)
        assert result.exit_code == 0, result.stderr
        assert 'pulled and unpacked' in result.output
        source_dest: Path = fake_package_path / destination / pipeline_name
        test_dest: Path = fake_repo_path / 'tests' / destination / pipeline_name
        config_env: str = 'base'
        params_config: Path = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert params_config.is_file()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    def test_micropkg_alias_refactors_imports(self, fake_project_cli: Any, fake_package_path: Path,
                                              fake_repo_path: Path, fake_metadata: Any) -> None:
        call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_file: Path = fake_package_path / 'pipelines' / PIPELINE_NAME / 'pipeline.py'
        import_stmt: str = f'import {fake_metadata.package_name}.pipelines.{PIPELINE_NAME}.nodes'
        with pipeline_file.open('a') as f:
            f.write(import_stmt)
        package_alias: str = 'alpha'
        pull_alias: str = 'beta'
        pull_destination: str = 'pipelines/lib'
        call_micropkg_package(cli=fake_project_cli, metadata=fake_metadata, alias=package_alias)
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=package_alias, version='0.1')
        CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file)], obj=fake_metadata)
        CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), '--alias', pull_alias, '--destination', pull_destination], obj=fake_metadata)
        pull: str = f'pipelines.lib.{pull_alias}'
        for alias_used in (package_alias, pull):
            alias_path: Path = Path(*alias_used.split('.'))
            path: Path = fake_package_path / alias_path / 'pipeline.py'
            file_content: str = path.read_text()
            expected_stmt: str = f'import {fake_metadata.package_name}.{alias_used}.nodes'
            assert expected_stmt in file_content

    def test_micropkg_pull_from_aliased_pipeline_conflicting_name(self, fake_project_cli: Any,
                                                                    fake_package_path: Path, fake_repo_path: Path,
                                                                    fake_metadata: Any) -> None:
        package_name: str = fake_metadata.package_name
        call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_file: Path = fake_package_path / 'pipelines' / PIPELINE_NAME / 'pipeline.py'
        import_stmt: str = f'import {package_name}.pipelines.{PIPELINE_NAME}.nodes'
        with pipeline_file.open('a') as f:
            f.write(import_stmt)
        call_micropkg_package(cli=fake_project_cli, metadata=fake_metadata, alias=package_name)
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=package_name, version='0.1')
        assert sdist_file.is_file()
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file)], obj=fake_metadata)
        assert result.exit_code == 0, result.output
        path: Path = fake_package_path / package_name / 'pipeline.py'
        file_content: str = path.read_text()
        expected_stmt: str = f'import {package_name}.{package_name}.nodes'
        assert expected_stmt in file_content

    def test_micropkg_pull_as_aliased_pipeline_conflicting_name(self, fake_project_cli: Any,
                                                                fake_package_path: Path, fake_repo_path: Path,
                                                                fake_metadata: Any) -> None:
        package_name: str = fake_metadata.package_name
        call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_file: Path = fake_package_path / 'pipelines' / PIPELINE_NAME / 'pipeline.py'
        import_stmt: str = f'import {package_name}.pipelines.{PIPELINE_NAME}.nodes'
        with pipeline_file.open('a') as f:
            f.write(import_stmt)
        call_micropkg_package(fake_project_cli, fake_metadata)
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), '--alias', package_name], obj=fake_metadata)
        assert result.exit_code == 0, result.output
        path: Path = fake_package_path / package_name / 'pipeline.py'
        file_content: str = path.read_text()
        expected_stmt: str = f'import {package_name}.{package_name}.nodes'
        assert expected_stmt in file_content

    def test_pull_sdist_fs_args(self, fake_project_cli: Any, fake_repo_path: Path,
                                mocker: Any, tmp_path: Path, fake_metadata: Any) -> None:
        """Test for pulling a sdist file with custom fs_args specified."""
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        call_pipeline_delete(fake_project_cli, fake_metadata)
        fs_args_config: Path = tmp_path / 'fs_args_config.yml'
        with fs_args_config.open(mode='w') as f:
            yaml.dump({'fs_arg_1': 1, 'fs_arg_2': {'fs_arg_2_nested_1': 2}}, f)
        mocked_filesystem = mocker.patch('fsspec.filesystem')
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        options: List[str] = ['--fs-args', str(fs_args_config)]
        CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), *options])
        mocked_filesystem.assert_called_once_with('file', fs_arg_1=1, fs_arg_2={'fs_arg_2_nested_1': 2})

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias', [None, 'alias_path'])
    def test_pull_tests_missing(self, fake_project_cli: Any, fake_repo_path: Path,
                                fake_package_path: Path, env: Optional[str],
                                alias: Optional[str], fake_metadata: Any) -> None:
        """Test for pulling a valid sdist file locally,
        but `tests` directory is missing from the sdist file.
        """
        call_pipeline_create(fake_project_cli, fake_metadata)
        test_path: Path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        shutil.rmtree(test_path)
        assert not test_path.exists()
        call_micropkg_package(fake_project_cli, fake_metadata)
        call_pipeline_delete(fake_project_cli, fake_metadata)
        source_path: Path = fake_package_path / 'pipelines' / PIPELINE_NAME
        source_params_config: Path = fake_repo_path / settings.CONF_SOURCE / 'base' / f'parameters_{PIPELINE_NAME}.yml'
        assert not source_path.exists()
        assert not source_params_config.exists()
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        options: List[str] = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), *options], obj=fake_metadata)
        assert result.exit_code == 0
        pipeline_name: str = alias or PIPELINE_NAME
        source_dest: Path = fake_package_path / pipeline_name
        test_dest: Path = fake_repo_path / 'tests' / 'pipelines' / pipeline_name
        config_env: str = env or 'base'
        params_config: Path = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert params_config.is_file()
        assert not test_dest.exists()

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias', [None, 'alias_path'])
    def test_pull_config_missing(self, fake_project_cli: Any, fake_repo_path: Path,
                                 fake_package_path: Path, env: Optional[str],
                                 alias: Optional[str], fake_metadata: Any) -> None:
        """
        Test for pulling a valid sdist file locally, but `config` directory is missing
        from the sdist file.
        """
        call_pipeline_create(fake_project_cli, fake_metadata)
        source_params_config: Path = fake_repo_path / settings.CONF_SOURCE / 'base' / f'parameters_{PIPELINE_NAME}.yml'
        source_params_config.unlink()
        call_micropkg_package(fake_project_cli, fake_metadata)
        call_pipeline_delete(fake_project_cli, fake_metadata)
        source_path: Path = fake_package_path / 'pipelines' / PIPELINE_NAME
        test_path: Path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        assert not source_path.exists()
        assert not test_path.exists()
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        options: List[str] = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), *options], obj=fake_metadata)
        assert result.exit_code == 0
        pipeline_name: str = alias or PIPELINE_NAME
        source_dest: Path = fake_package_path / pipeline_name
        test_dest: Path = fake_repo_path / 'tests' / pipeline_name
        config_env: str = env or 'base'
        dest_params_config: Path = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert not dest_params_config.exists()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias', [None, 'alias_path'])
    def test_pull_from_pypi(self, fake_project_cli: Any, fake_repo_path: Path,
                            mocker: Any, tmp_path: Path, fake_package_path: Path,
                            env: Optional[str], alias: Optional[str], fake_metadata: Any,
                            temp_dir_with_context_manager: Any) -> None:
        """
        Test for pulling a valid sdist file from pypi.
        """
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata, destination=tmp_path)
        version: str = '0.1'
        sdist_file: Path = tmp_path / _get_sdist_name(name=PIPELINE_NAME, version=version)
        assert sdist_file.is_file()
        call_pipeline_delete(fake_project_cli, fake_metadata)
        source_path: Path = fake_package_path / 'pipelines' / PIPELINE_NAME
        test_path: Path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        source_params_config: Path = fake_repo_path / settings.CONF_SOURCE / 'base' / f'parameters_{PIPELINE_NAME}.yml'
        assert not source_path.exists()
        assert not test_path.exists()
        assert not source_params_config.exists()
        python_call_mock = mocker.patch('kedro.framework.cli.micropkg.python_call')
        mocker.patch('kedro.framework.cli.micropkg.tempfile.TemporaryDirectory', return_value=temp_dir_with_context_manager)

        class _FakeWheelMetadata:
            def get_all(self, name: str, failobj: Any = None) -> List[Any]:
                return []
        mocker.patch('kedro.framework.cli.micropkg.project_wheel_metadata', return_value=_FakeWheelMetadata())
        options: List[str] = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        package_name: str = 'my-pipeline'
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', package_name, *options], obj=fake_metadata)
        assert result.exit_code == 0
        assert 'pulled and unpacked' in result.output
        python_call_mock.assert_called_once_with('pip', ['download', '--no-deps', '--no-binary', ':all:', '--dest', str(tmp_path), package_name])
        pipeline_name_final: str = alias or PIPELINE_NAME
        source_dest: Path = fake_package_path / pipeline_name_final
        test_dest: Path = fake_repo_path / 'tests' / pipeline_name_final
        config_env: str = env or 'base'
        dest_params_config: Path = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name_final}.yml'
        self.assert_package_files_exist(source_dest)
        assert dest_params_config.is_file()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    def test_invalid_pull_from_pypi(self, fake_project_cli: Any, mocker: Any, tmp_path: Path,
                                    fake_metadata: Any, temp_dir_with_context_manager: Any) -> None:
        """
        Test for pulling package from pypi, and it cannot be found.
        """
        pypi_error_message: str = 'ERROR: Could not find a version that satisfies the requirement'
        python_call_mock = mocker.patch('kedro.framework.cli.micropkg.python_call', side_effect=ClickException(pypi_error_message))
        mocker.patch('kedro.framework.cli.micropkg.tempfile.TemporaryDirectory', return_value=temp_dir_with_context_manager)
        invalid_pypi_name: str = 'non_existent'
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', invalid_pypi_name], obj=fake_metadata)
        assert result.exit_code
        python_call_mock.assert_called_once_with('pip', ['download', '--no-deps', '--no-binary', ':all:', '--dest', str(tmp_path), invalid_pypi_name])
        assert pypi_error_message in result.stdout

    def test_pull_from_pypi_more_than_one_sdist_file(self, fake_project_cli: Any, mocker: Any, tmp_path: Path,
                                                     fake_metadata: Any, temp_dir_with_context_manager: Any) -> None:
        """
        Test for pulling a sdist file with `pip download`, but there are more than one sdist
        file to unzip.
        """
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata, destination=tmp_path)
        call_micropkg_package(fake_project_cli, fake_metadata, alias='another', destination=tmp_path)
        mocker.patch('kedro.framework.cli.micropkg.python_call')
        mocker.patch('kedro.framework.cli.micropkg.tempfile.TemporaryDirectory', return_value=temp_dir_with_context_manager)
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', PIPELINE_NAME], obj=fake_metadata)
        assert result.exit_code
        assert 'Error: More than 1 or no sdist files found:' in result.output

    def test_pull_unsupported_protocol_by_fsspec(self, fake_project_cli: Any, fake_metadata: Any,
                                                 tmp_path: Path, mocker: Any, temp_dir_with_context_manager: Any) -> None:
        protocol: str = 'unsupported'
        exception_message: str = f'Protocol not known: {protocol}'
        error_message: str = 'Error: More than 1 or no sdist files found:'
        package_path: str = f'{protocol}://{PIPELINE_NAME}'
        python_call_mock = mocker.patch('kedro.framework.cli.micropkg.python_call')
        filesystem_mock = mocker.patch('fsspec.filesystem', side_effect=ValueError(exception_message))
        mocker.patch('kedro.framework.cli.micropkg.tempfile.TemporaryDirectory', return_value=temp_dir_with_context_manager)
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', package_path], obj=fake_metadata)
        assert result.exit_code
        filesystem_mock.assert_called_once_with(protocol)
        python_call_mock.assert_called_once_with('pip', ['download', '--no-deps', '--no-binary', ':all:', '--dest', str(tmp_path), package_path])
        assert exception_message in result.output
        assert "Trying to use 'pip download'..." in result.output
        assert error_message in result.output

    def test_micropkg_pull_invalid_sdist(self, fake_project_cli: Any, fake_repo_path: Path,
                                           fake_metadata: Any, tmp_path: Path) -> None:
        """
        Test for pulling an invalid sdist file locally with more than one package.
        """
        error_message: str = 'Invalid sdist was extracted: exactly one directory was expected'
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        with tarfile.open(sdist_file, 'r:gz') as tar:
            safe_extract(tar, tmp_path)
        extra_project: Path = tmp_path / f'{PIPELINE_NAME}-0.1_extra'
        extra_project.mkdir()
        (extra_project / 'README.md').touch()
        sdist_file.unlink()
        with tarfile.open(sdist_file, 'w:gz') as tar:
            for fn in tmp_path.iterdir():
                tar.add(fn, arcname=fn.relative_to(tmp_path))
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file)], obj=fake_metadata)
        assert result.exit_code == 1
        assert error_message in result.stdout

    def test_micropkg_pull_invalid_package_contents(self, fake_project_cli: Any, fake_repo_path: Path,
                                                      fake_metadata: Any, tmp_path: Path) -> None:
        """
        Test for pulling an invalid sdist file locally with more than one package.
        """
        error_message: str = 'Invalid package contents: exactly one package was expected'
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        sdist_file: Path = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        with tarfile.open(sdist_file, 'r:gz') as tar:
            safe_extract(tar, tmp_path)
        extra_package: Path = tmp_path / f'{PIPELINE_NAME}-0.1' / f'{PIPELINE_NAME}_extra'
        extra_package.mkdir()
        (extra_package / '__init__.py').touch()
        sdist_file.unlink()
        with tarfile.open(sdist_file, 'w:gz') as tar:
            for fn in tmp_path.iterdir():
                tar.add(fn, arcname=fn.relative_to(tmp_path))
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file)], obj=fake_metadata)
        assert result.exit_code == 1
        assert error_message in result.stdout

    @pytest.mark.parametrize('tar_members,path_name', [(['../tarmember', 'tarmember'], 'destination'),
                                                         (['tarmember', '../tarmember'], 'destination')])
    def test_path_traversal(self, tar_members: List[str], path_name: str) -> None:
        """Test for checking path traversal attempt in tar file"""
        tar = Mock()
        tar.getmembers.return_value = [tarfile.TarInfo(name=tar_name) for tar_name in tar_members]
        path: Path = Path(path_name)
        with pytest.raises(Exception, match='Failed to safely extract tar file.'):
            safe_extract(tar, path)


@pytest.mark.usefixtures('chdir_to_dummy_project', 'cleanup_dist', 'cleanup_pyproject_toml')
class TestMicropkgPullFromManifest:

    def test_micropkg_pull_all(self, fake_repo_path: Path, fake_project_cli: Any,
                                fake_metadata: Any, mocker: Any) -> None:
        from kedro.framework.cli import micropkg
        spy = mocker.spy(micropkg, '_pull_package')
        pyproject_toml: Path = fake_repo_path / 'pyproject.toml'
        sdist_file_template: str = str(fake_repo_path / 'dist' / _get_sdist_name('{}', '0.1'))
        project_toml_str: str = textwrap.dedent(
            f'''
            [tool.kedro.micropkg.pull]
            "{sdist_file_template.format('first')}" = {{alias = "dp", destination = "pipelines"}}
            "{sdist_file_template.format('second')}" = {{alias = "ds", destination = "pipelines", env = "local"}}
            "{sdist_file_template.format('third')}" = {{}}
            '''
        )
        with pyproject_toml.open(mode='a') as file:
            file.write(project_toml_str)
        for name in ('first', 'second', 'third'):
            call_pipeline_create(fake_project_cli, fake_metadata, pipeline_name=name)
            call_micropkg_package(fake_project_cli, fake_metadata, pipeline_name=name)
            call_pipeline_delete(fake_project_cli, fake_metadata, pipeline_name=name)
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', '--all'], obj=fake_metadata)
        assert result.exit_code == 0
        assert 'Micro-packages pulled and unpacked!' in result.output
        assert spy.call_count == 3
        build_config = toml.loads(project_toml_str)
        pull_manifest = build_config['tool']['kedro']['micropkg']['pull']
        for sdist_file, pull_specs in pull_manifest.items():
            expected_call = mocker.call(sdist_file, fake_metadata, **pull_specs)
            assert expected_call in spy.call_args_list

    def test_micropkg_pull_all_empty_toml(self, fake_repo_path: Path, fake_project_cli: Any,
                                          fake_metadata: Any, mocker: Any) -> None:
        from kedro.framework.cli import micropkg
        spy = mocker.spy(micropkg, '_pull_package')
        pyproject_toml: Path = fake_repo_path / 'pyproject.toml'
        with pyproject_toml.open(mode='a') as file:
            file.write('\n[tool.kedro.micropkg.pull]\n')
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', '--all'], obj=fake_metadata)
        assert result.exit_code == 0
        expected_message: str = "Nothing to pull. Please update the 'pyproject.toml' package manifest section."
        assert expected_message in result.output
        assert not spy.called

    def test_invalid_toml(self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any) -> None:
        pyproject_toml: Path = fake_repo_path / 'pyproject.toml'
        with pyproject_toml.open(mode='a') as file:
            file.write('what/toml?')
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', '--all'], obj=fake_metadata)
        assert result.exit_code
        assert isinstance(result.exception, toml.TomlDecodeError)

    def test_micropkg_pull_no_arg_provided(self, fake_project_cli: Any, fake_metadata: Any) -> None:
        result: Result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull'], obj=fake_metadata)
        assert result.exit_code
        expected_message: str = "Please specify a package path or add '--all' to pull all micro-packages in the 'pyproject.toml' package manifest section."
        assert expected_message in result.output