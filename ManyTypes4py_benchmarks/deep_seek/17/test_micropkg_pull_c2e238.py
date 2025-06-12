import filecmp
import shutil
import tarfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import Mock
import pytest
import toml
import yaml
from click import ClickException
from click.testing import CliRunner
from kedro.framework.cli.micropkg import _get_sdist_name, safe_extract
from kedro.framework.project import settings
from kedro.framework.startup import ProjectMetadata

PIPELINE_NAME = 'my_pipeline'

@pytest.fixture
def temp_dir_with_context_manager(tmp_path: Path) -> Mock:
    mock_temp_dir = Mock()
    mock_temp_dir.__enter__ = Mock(return_value=tmp_path)
    mock_temp_dir.__exit__ = Mock(return_value=None)
    return mock_temp_dir

def call_pipeline_create(cli: Any, metadata: ProjectMetadata, pipeline_name: str = PIPELINE_NAME) -> None:
    result = CliRunner().invoke(cli, ['pipeline', 'create', pipeline_name], obj=metadata)
    assert result.exit_code == 0

def call_micropkg_package(
    cli: Any,
    metadata: ProjectMetadata,
    alias: Optional[str] = None,
    destination: Optional[Union[str, Path]] = None,
    pipeline_name: str = PIPELINE_NAME
) -> None:
    options = ['--alias', alias] if alias else []
    options += ['--destination', str(destination)] if destination else []
    result = CliRunner().invoke(
        cli,
        ['micropkg', 'package', f'pipelines.{pipeline_name}', *options],
        obj=metadata
    )
    assert result.exit_code == 0, result.output

def call_pipeline_delete(cli: Any, metadata: ProjectMetadata, pipeline_name: str = PIPELINE_NAME) -> None:
    result = CliRunner().invoke(cli, ['pipeline', 'delete', '-y', pipeline_name], obj=metadata)
    assert result.exit_code == 0

@pytest.mark.usefixtures('chdir_to_dummy_project', 'cleanup_dist')
class TestMicropkgPullCommand:

    def assert_package_files_exist(self, source_path: Path) -> None:
        assert {f.name for f in source_path.iterdir()} == {'__init__.py', 'nodes.py', 'pipeline.py'}

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias, destination', [(None, None), ('aliased', None), ('aliased', 'pipelines'), (None, 'pipelines')])
    def test_pull_local_sdist(
        self,
        fake_project_cli: Any,
        fake_repo_path: Path,
        fake_package_path: Path,
        env: Optional[str],
        alias: Optional[str],
        destination: Optional[str],
        fake_metadata: ProjectMetadata
    ) -> None:
        """Test for pulling a valid sdist file locally."""
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        call_pipeline_delete(fake_project_cli, fake_metadata)
        source_path = fake_package_path / 'pipelines' / PIPELINE_NAME
        config_path = fake_repo_path / settings.CONF_SOURCE / 'base' / 'pipelines' / PIPELINE_NAME
        test_path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        assert not source_path.exists()
        assert not test_path.exists()
        assert not config_path.exists()
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        options = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        options += ['--destination', destination] if destination else []
        result = CliRunner().invoke(
            fake_project_cli,
            ['micropkg', 'pull', str(sdist_file), *options],
            obj=fake_metadata
        )
        assert result.exit_code == 0, result.output
        assert 'pulled and unpacked' in result.output
        pipeline_name = alias or PIPELINE_NAME
        destination = destination or Path()
        source_dest = fake_package_path / destination / pipeline_name
        test_dest = fake_repo_path / 'tests' / destination / pipeline_name
        config_env = env or 'base'
        params_config = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert params_config.is_file()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias, destination', [(None, None), ('aliased', None), ('aliased', 'pipelines'), (None, 'pipelines')])
    def test_pull_local_sdist_compare(
        self,
        fake_project_cli: Any,
        fake_repo_path: Path,
        fake_package_path: Path,
        env: Optional[str],
        alias: Optional[str],
        destination: Optional[str],
        fake_metadata: ProjectMetadata
    ) -> None:
        """Test for pulling a valid sdist file locally, unpack it
        into another location and check that unpacked files
        are identical to the ones in the original modular pipeline.
        """
        pipeline_name = 'another_pipeline'
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata, alias=pipeline_name)
        source_path = fake_package_path / 'pipelines' / PIPELINE_NAME
        test_path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        source_params_config = fake_repo_path / settings.CONF_SOURCE / 'base' / f'parameters_{PIPELINE_NAME}.yml'
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=pipeline_name, version='0.1')
        assert sdist_file.is_file()
        options = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        options += ['--destination', destination] if destination else []
        result = CliRunner().invoke(
            fake_project_cli,
            ['micropkg', 'pull', str(sdist_file), *options],
            obj=fake_metadata
        )
        assert result.exit_code == 0, result.output
        assert 'pulled and unpacked' in result.output
        pipeline_name = alias or pipeline_name
        destination = destination or Path()
        source_dest = fake_package_path / destination / pipeline_name
        test_dest = fake_repo_path / 'tests' / destination / pipeline_name
        config_env = env or 'base'
        dest_params_config = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        assert not filecmp.dircmp(source_path, source_dest).diff_files
        assert not filecmp.dircmp(test_path, test_dest).diff_files
        assert source_params_config.read_bytes() == dest_params_config.read_bytes()

    def test_micropkg_pull_same_alias_package_name(
        self,
        fake_project_cli: Any,
        fake_repo_path: Path,
        fake_package_path: Path,
        fake_metadata: ProjectMetadata
    ) -> None:
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        pipeline_name = PIPELINE_NAME
        destination = 'tools'
        result = CliRunner().invoke(
            fake_project_cli,
            ['micropkg', 'pull', str(sdist_file), '--destination', destination, '--alias', pipeline_name],
            obj=fake_metadata
        )
        assert result.exit_code == 0, result.stderr
        assert 'pulled and unpacked' in result.output
        source_dest = fake_package_path / destination / pipeline_name
        test_dest = fake_repo_path / 'tests' / destination / pipeline_name
        config_env = 'base'
        params_config = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert params_config.is_file()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    def test_micropkg_pull_nested_destination(
        self,
        fake_project_cli: Any,
        fake_repo_path: Path,
        fake_package_path: Path,
        fake_metadata: ProjectMetadata
    ) -> None:
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        pipeline_name = PIPELINE_NAME
        destination = 'pipelines/nested'
        result = CliRunner().invoke(
            fake_project_cli,
            ['micropkg', 'pull', str(sdist_file), '--destination', destination, '--alias', pipeline_name],
            obj=fake_metadata
        )
        assert result.exit_code == 0, result.stderr
        assert 'pulled and unpacked' in result.output
        source_dest = fake_package_path / destination / pipeline_name
        test_dest = fake_repo_path / 'tests' / destination / pipeline_name
        config_env = 'base'
        params_config = fake_repo_path / settings.CONF_SOURCE / config_env / f'parameters_{pipeline_name}.yml'
        self.assert_package_files_exist(source_dest)
        assert params_config.is_file()
        actual_test_files = {f.name for f in test_dest.iterdir()}
        expected_test_files = {'__init__.py', 'test_pipeline.py'}
        assert actual_test_files == expected_test_files

    def test_micropkg_alias_refactors_imports(
        self,
        fake_project_cli: Any,
        fake_package_path: Path,
        fake_repo_path: Path,
        fake_metadata: ProjectMetadata
    ) -> None:
        call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_file = fake_package_path / 'pipelines' / PIPELINE_NAME / 'pipeline.py'
        import_stmt = f'import {fake_metadata.package_name}.pipelines.{PIPELINE_NAME}.nodes'
        with pipeline_file.open('a') as f:
            f.write(import_stmt)
        package_alias = 'alpha'
        pull_alias = 'beta'
        pull_destination = 'pipelines/lib'
        call_micropkg_package(cli=fake_project_cli, metadata=fake_metadata, alias=package_alias)
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=package_alias, version='0.1')
        CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file)], obj=fake_metadata)
        CliRunner().invoke(
            fake_project_cli,
            ['micropkg', 'pull', str(sdist_file), '--alias', pull_alias, '--destination', pull_destination],
            obj=fake_metadata
        )
        pull = f'pipelines.lib.{pull_alias}'
        for alias in (package_alias, pull):
            alias_path = Path(*alias.split('.'))
            path = fake_package_path / alias_path / 'pipeline.py'
            file_content = path.read_text()
            expected_stmt = f'import {fake_metadata.package_name}.{alias}.nodes'
            assert expected_stmt in file_content

    def test_micropkg_pull_from_aliased_pipeline_conflicting_name(
        self,
        fake_project_cli: Any,
        fake_package_path: Path,
        fake_repo_path: Path,
        fake_metadata: ProjectMetadata
    ) -> None:
        package_name = fake_metadata.package_name
        call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_file = fake_package_path / 'pipelines' / PIPELINE_NAME / 'pipeline.py'
        import_stmt = f'import {package_name}.pipelines.{PIPELINE_NAME}.nodes'
        with pipeline_file.open('a') as f:
            f.write(import_stmt)
        call_micropkg_package(cli=fake_project_cli, metadata=fake_metadata, alias=package_name)
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=package_name, version='0.1')
        assert sdist_file.is_file()
        result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file)], obj=fake_metadata)
        assert result.exit_code == 0, result.output
        path = fake_package_path / package_name / 'pipeline.py'
        file_content = path.read_text()
        expected_stmt = f'import {package_name}.{package_name}.nodes'
        assert expected_stmt in file_content

    def test_micropkg_pull_as_aliased_pipeline_conflicting_name(
        self,
        fake_project_cli: Any,
        fake_package_path: Path,
        fake_repo_path: Path,
        fake_metadata: ProjectMetadata
    ) -> None:
        package_name = fake_metadata.package_name
        call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_file = fake_package_path / 'pipelines' / PIPELINE_NAME / 'pipeline.py'
        import_stmt = f'import {package_name}.pipelines.{PIPELINE_NAME}.nodes'
        with pipeline_file.open('a') as f:
            f.write(import_stmt)
        call_micropkg_package(cli=fake_project_cli, metadata=fake_metadata)
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        result = CliRunner().invoke(
            fake_project_cli,
            ['micropkg', 'pull', str(sdist_file), '--alias', package_name],
            obj=fake_metadata
        )
        assert result.exit_code == 0, result.output
        path = fake_package_path / package_name / 'pipeline.py'
        file_content = path.read_text()
        expected_stmt = f'import {package_name}.{package_name}.nodes'
        assert expected_stmt in file_content

    def test_pull_sdist_fs_args(
        self,
        fake_project_cli: Any,
        fake_repo_path: Path,
        mocker: Mock,
        tmp_path: Path,
        fake_metadata: ProjectMetadata
    ) -> None:
        """Test for pulling a sdist file with custom fs_args specified."""
        call_pipeline_create(fake_project_cli, fake_metadata)
        call_micropkg_package(fake_project_cli, fake_metadata)
        call_pipeline_delete(fake_project_cli, fake_metadata)
        fs_args_config = tmp_path / 'fs_args_config.yml'
        with fs_args_config.open(mode='w') as f:
            yaml.dump({'fs_arg_1': 1, 'fs_arg_2': {'fs_arg_2_nested_1': 2}}, f)
        mocked_filesystem = mocker.patch('fsspec.filesystem')
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        options = ['--fs-args', str(fs_args_config)]
        CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', str(sdist_file), *options])
        mocked_filesystem.assert_called_once_with('file', fs_arg_1=1, fs_arg_2={'fs_arg_2_nested_1': 2})

    @pytest.mark.parametrize('env', [None, 'local'])
    @pytest.mark.parametrize('alias', [None, 'alias_path'])
    def test_pull_tests_missing(
        self,
        fake_project_cli: Any,
        fake_repo_path: Path,
        fake_package_path: Path,
        env: Optional[str],
        alias: Optional[str],
        fake_metadata: ProjectMetadata
    ) -> None:
        """Test for pulling a valid sdist file locally,
        but `tests` directory is missing from the sdist file.
        """
        call_pipeline_create(fake_project_cli, fake_metadata)
        test_path = fake_repo_path / 'tests' / 'pipelines' / PIPELINE_NAME
        shutil.rmtree(test_path)
        assert not test_path.exists()
        call_micropkg_package(fake_project_cli, fake_metadata)
        call_pipeline_delete(fake_project_cli, fake_metadata)
        source_path = fake_package_path / 'pipelines' / PIPELINE_NAME
        source_params_config = fake_repo_path / settings.CONF_SOURCE / 'base' / f'parameters_{PIPELINE_NAME}.yml'
        assert not source_path.exists()
        assert not source_params_config.exists()
        sdist_file = fake_repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        options = ['-e', env] if env else []
        options += ['--alias', alias] if alias else []
        result = CliRunner().invoke(
            fake_project_cli,
            ['micropkg', 'pull', str(sdist_file), *options