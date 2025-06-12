import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def call_mock(mocker: Any) -> MagicMock:
    return mocker.patch('kedro.framework.cli.project.call')


@pytest.fixture
def fake_copyfile(mocker: Any) -> MagicMock:
    return mocker.patch('shutil.copyfile')


@pytest.mark.usefixtures('chdir_to_dummy_project')
class TestIpythonCommand:

    def test_happy_path(
        self,
        call_mock: MagicMock,
        fake_project_cli: Any,
        fake_repo_path: Path,
        fake_metadata: Any
    ) -> None:
        result = CliRunner().invoke(fake_project_cli, ['ipython', '--random-arg', 'value'], obj=fake_metadata)
        assert not result.exit_code, result.stdout
        call_mock.assert_called_once_with(['ipython', '--ext', 'kedro.ipython', '--random-arg', 'value'])

    @pytest.mark.parametrize('env_flag,env', [('--env', 'base'), ('-e', 'local')])
    def test_env(
        self,
        env_flag: str,
        env: str,
        fake_project_cli: Any,
        mocker: Any,
        fake_metadata: Any
    ) -> None:
        """This tests starting ipython with specific env."""
        mock_environ = mocker.patch('os.environ', {})
        result = CliRunner().invoke(fake_project_cli, ['ipython', env_flag, env], obj=fake_metadata)
        assert not result.exit_code, result.stdout
        assert mock_environ['KEDRO_ENV'] == env

    def test_fail_no_ipython(
        self,
        fake_project_cli: Any,
        mocker: Any
    ) -> None:
        mocker.patch.dict('sys.modules', {'IPython': None})
        result = CliRunner().invoke(fake_project_cli, ['ipython'])
        assert result.exit_code
        error = (
            "Module 'IPython' not found. Make sure to install required project dependencies by running the "
            "'pip install -r requirements.txt' command first."
        )
        assert error in result.output


@pytest.mark.usefixtures('chdir_to_dummy_project')
class TestPackageCommand:

    def test_happy_path(
        self,
        call_mock: MagicMock,
        fake_project_cli: Any,
        mocker: Any,
        fake_repo_path: Path,
        fake_metadata: Any
    ) -> None:
        result = CliRunner().invoke(fake_project_cli, ['package'], obj=fake_metadata)
        assert not result.exit_code, result.stdout
        call_mock.assert_has_calls([
            mocker.call(
                [sys.executable, '-m', 'build', '--wheel', '--outdir', 'dist'],
                cwd=str(fake_repo_path)
            ),
            mocker.call([
                'tar',
                '--exclude=local/*.yml',
                '-czf',
                f'dist/conf-{fake_metadata.package_name}.tar.gz',
                f'--directory={fake_metadata.project_path}',
                'conf'
            ])
        ])

    def test_no_pyproject_toml(
        self,
        call_mock: MagicMock,
        fake_project_cli: Any,
        mocker: Any,
        fake_repo_path: Path,
        fake_metadata: Any
    ) -> None:
        (fake_metadata.project_path / 'pyproject.toml').unlink(missing_ok=True)
        result = CliRunner().invoke(fake_project_cli, ['package'], obj=fake_metadata)
        assert not result.exit_code, result.stdout
        call_mock.assert_has_calls([
            mocker.call(
                [sys.executable, '-m', 'build', '--wheel', '--outdir', '../dist'],
                cwd=str(fake_metadata.source_dir)
            ),
            mocker.call([
                'tar',
                '--exclude=local/*.yml',
                '-czf',
                f'dist/conf-{fake_metadata.package_name}.tar.gz',
                f'--directory={fake_metadata.project_path}',
                'conf'
            ])
        ])
