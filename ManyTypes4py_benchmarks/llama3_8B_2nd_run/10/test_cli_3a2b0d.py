from collections import namedtuple
from itertools import cycle
from os import rename
from pathlib import Path
from unittest.mock import MagicMock, patch
import click
import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf
from pytest import fixture, mark, raises, warns
from kedro import KedroDeprecationWarning
from kedro import __version__ as version
from kedro.framework.cli import load_entry_points
from kedro.framework.cli.cli import KedroCLI, _init_plugins, cli, global_commands, project_commands
from kedro.framework.cli.utils import CommandCollection, KedroCliError, _clean_pycache, find_run_command, forward_command, get_pkg_version
from kedro.runner import ParallelRunner, SequentialRunner

@fixture
def requirements_file(tmp_path: Path) -> Path:
    body = '\n'.join(['SQLAlchemy>=1.2.0, <2.0', 'pandas==0.23.0', 'toposort']) + '\n'
    reqs_file = tmp_path / 'requirements.txt'
    reqs_file.write_text(body)
    yield reqs_file

@fixture
def fake_session(mocker: pytest.Mock) -> KedroSession:
    mock_session_create = mocker.patch.object(KedroSession, 'create')
    mocked_session = mock_session_create.return_value.__enter__.return_value
    return mocked_session

class TestCliCommands:
    @pytest.mark.parametrize("cli_arg, expected_extra_params", [
        ("foo=bar", {"foo": "bar"}),
        ("foo=123.45, bar=1a, baz=678. ,qux=1e-2,quux=0,quuz=", {"foo": 123.45, "bar": "1a", "baz": 678.0, "qux": 0.01, "quux": 0, "quuz": None}),
        ("foo=bar, baz=fizz=buzz", {"foo": "bar", "baz": "fizz=buzz"}),
        ("foo=fizz=buzz", {"foo": "fizz=buzz"}),
        ("foo=bar, baz=https://example.com", {"foo": "bar", "baz": "https://example.com"}),
        ("foo=bar, foo=fizz buzz", {"foo": "fizz buzz"}),
        ("foo.nested=bar", {"foo": {"nested": "bar"}}),
        ("foo.nested=123.45", {"foo": {"nested": 123.45}}),
        ("foo.nested_1.double_nest=123.45,foo.nested_2=1a", {"foo": {"nested_1": {"double_nest": 123.45}, "nested_2": "1a"}}),
    ])
    def test_run_extra_params(self, cli_arg: str, expected_extra_params: dict, fake_project_cli: KedroCLI, fake_metadata: KedroSession, mocker: pytest.Mock) -> None:
        mock_session_create = mocker.patch.object(KedroSession, 'create')
        result = CliRunner().invoke(fake_project_cli, ['run', '--params', cli_arg], obj=fake_metadata)
        assert not result.exit_code
        mock_session_create.assert_called_once_with(env=mocker.ANY, conf_source=None, extra_params=expected_extra_params)

    @pytest.mark.parametrize("bad_arg", ["bad", "foo=bar,bad"])
    def test_bad_extra_params(self, fake_project_cli: KedroCLI, fake_metadata: KedroSession, bad_arg: str) -> None:
        result = CliRunner().invoke(fake_project_cli, ['run', '--params', bad_arg], obj=fake_metadata)
        assert result.exit_code
        assert "Item `bad` must contain a key and a value separated by `=`." in result.stdout

    @pytest.mark.parametrize("bad_arg", ["=", "=value", " =value"])
    def test_bad_params_key(self, fake_project_cli: KedroCLI, fake_metadata: KedroSession, bad_arg: str) -> None:
        result = CliRunner().invoke(fake_project_cli, ['run', '--params', bad_arg], obj=fake_metadata)
        assert result.exit_code
        assert "Parameter key cannot be an empty string" in result.stdout
