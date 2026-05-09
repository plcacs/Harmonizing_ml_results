import pytest
from pathlib import Path
from unittest.mock import Mock
import tarfile
from click import ClickException
from click.testing import CliRunner
from kedro.framework.cli.micropkg import _get_sdist_name, safe_extract
from kedro.framework.project import settings
from toml import loads

class TestMicropkgPullCommand:
    @pytest.mark.usefixtures('chdir_to_dummy_project', 'cleanup_dist', 'cleanup_pyproject_toml')
    def test_micropkg_pull_all(
        self,
        fake_repo_path: Path,
        fake_project_cli: CliRunner,
        fake_metadata: object,
        mocker: pytest.Mock,
    ) -> None:
        # ...
        spy = mocker.spy(micropkg, '_pull_package')
        # ...
        result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', '--all'], obj=fake_metadata)
        # ...
        assert spy.call_count == 3
        build_config = loads(project_toml_str)
        pull_manifest = build_config['tool']['kedro']['micropkg']['pull']
        for sdist_file, pull_specs in pull_manifest.items():
            expected_call = mocker.call(sdist_file, fake_metadata, **pull_specs)
            assert expected_call in spy.call_args_list

    def test_micropkg_pull_all_empty_toml(
        self,
        fake_repo_path: Path,
        fake_project_cli: CliRunner,
        fake_metadata: object,
        mocker: pytest.Mock,
    ) -> None:
        # ...
        spy = mocker.spy(micropkg, '_pull_package')
        # ...
        result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', '--all'], obj=fake_metadata)
        # ...
        assert not spy.called

    def test_invalid_toml(
        self,
        fake_repo_path: Path,
        fake_project_cli: CliRunner,
        fake_metadata: object,
    ) -> None:
        # ...
        result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull', '--all'], obj=fake_metadata)
        # ...
        assert isinstance(result.exception, toml.TomlDecodeError)

    def test_micropkg_pull_no_arg_provided(
        self,
        fake_project_cli: CliRunner,
        fake_metadata: object,
    ) -> None:
        # ...
        result = CliRunner().invoke(fake_project_cli, ['micropkg', 'pull'], obj=fake_metadata)
        # ...
        expected_message = "Please specify a package path or add '--all' to pull all micro-packages in the 'pyproject.toml' package manifest section."
        assert expected_message in result.output
