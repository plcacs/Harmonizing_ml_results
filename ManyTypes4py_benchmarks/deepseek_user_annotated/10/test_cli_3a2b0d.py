from collections import namedtuple
from itertools import cycle
from os import rename
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner, Result
from omegaconf import OmegaConf
from pytest import fixture, mark, raises, warns

from kedro import KedroDeprecationWarning
from kedro import __version__ as version
from kedro.framework.cli import load_entry_points
from kedro.framework.cli.cli import (
    KedroCLI,
    _init_plugins,
    cli,
    global_commands,
    project_commands,
)
from kedro.framework.cli.utils import (
    CommandCollection,
    KedroCliError,
    _clean_pycache,
    find_run_command,
    forward_command,
    get_pkg_version,
)
from kedro.framework.session import KedroSession
from kedro.runner import ParallelRunner, SequentialRunner


@click.group(name="stub_cli")
def stub_cli() -> None:
    """Stub CLI group description."""
    print("group callback")


@stub_cli.command(name="stub_command")
def stub_command() -> None:
    print("command callback")


@forward_command(stub_cli, name="forwarded_command")
def forwarded_command(args: List[str], **kwargs: Any) -> None:
    print("fred", args)


@forward_command(stub_cli, name="forwarded_help", forward_help=True)
def forwarded_help(args: List[str], **kwargs: Any) -> None:
    print("fred", args)


@forward_command(stub_cli)
def unnamed(args: List[str], **kwargs: Any) -> None:
    print("fred", args)


@fixture
def requirements_file(tmp_path: Path) -> Path:
    body = "\n".join(["SQLAlchemy>=1.2.0, <2.0", "pandas==0.23.0", "toposort"]) + "\n"
    reqs_file = tmp_path / "requirements.txt"
    reqs_file.write_text(body)
    yield reqs_file


@fixture
def fake_session(mocker: Any) -> MagicMock:
    mock_session_create = mocker.patch.object(KedroSession, "create")
    mocked_session = mock_session_create.return_value.__enter__.return_value
    return mocked_session


class TestCliCommands:
    def test_cli(self) -> None:
        """Run `kedro` without arguments."""
        result = CliRunner().invoke(cli, [])

        assert result.exit_code == 0
        assert "kedro" in result.output

    def test_print_version(self) -> None:
        """Check that `kedro --version` and `kedro -V` outputs contain
        the current package version."""
        result = CliRunner().invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert version in result.output

        result_abr = CliRunner().invoke(cli, ["-V"])
        assert result_abr.exit_code == 0
        assert version in result_abr.output

    def test_info_contains_plugin_versions(self, entry_point: Any) -> None:
        entry_point.dist.version = "1.0.2"
        entry_point.module = "bob.fred"

        result = CliRunner().invoke(cli, ["info"])
        assert result.exit_code == 0
        assert (
            "bob: 1.0.2 (entry points:cli_hooks,global,hooks,init,line_magic,project,starters)"
            in result.output
        )

        entry_point.load.assert_not_called()

    def test_info_only_kedro_telemetry_plugin_installed(self) -> None:
        result = CliRunner().invoke(cli, ["info"])
        assert result.exit_code == 0

        split_result = result.output.strip().split("\n")
        assert "Installed plugins" in split_result[-2]
        assert "kedro_telemetry" in split_result[-1]

    def test_help(self) -> None:
        """Check that `kedro --help` returns a valid help message."""
        result = CliRunner().invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "kedro" in result.output

        result = CliRunner().invoke(cli, ["-h"])
        assert result.exit_code == 0
        assert "-h, --help     Show this message and exit." in result.output


class TestCommandCollection:
    def test_found(self) -> None:
        """Test calling existing command."""
        cmd_collection = CommandCollection(("Commands", [cli, stub_cli]))
        result = CliRunner().invoke(cmd_collection, ["stub_command"])
        assert result.exit_code == 0
        assert "group callback" not in result.output
        assert "command callback" in result.output

    def test_found_reverse(self) -> None:
        """Test calling existing command."""
        cmd_collection = CommandCollection(("Commands", [stub_cli, cli]))
        result = CliRunner().invoke(cmd_collection, ["stub_command"])
        assert result.exit_code == 0
        assert "group callback" in result.output
        assert "command callback" in result.output

    def test_not_found(self) -> None:
        """Test calling nonexistent command."""
        cmd_collection = CommandCollection(("Commands", [cli, stub_cli]))
        result = CliRunner().invoke(cmd_collection, ["not_found"])
        assert result.exit_code == 2
        assert "No such command" in result.output
        assert "Did you mean one of these" not in result.output

    def test_not_found_closest_match(self, mocker: Any) -> None:
        """Check that calling a nonexistent command with a close match returns the close match"""
        patched_difflib = mocker.patch(
            "kedro.framework.cli.utils.difflib.get_close_matches",
            return_value=["suggestion_1", "suggestion_2"],
        )

        cmd_collection = CommandCollection(("Commands", [cli, stub_cli]))
        result = CliRunner().invoke(cmd_collection, ["not_found"])

        patched_difflib.assert_called_once_with(
            "not_found", mocker.ANY, mocker.ANY, mocker.ANY
        )

        assert result.exit_code == 2
        assert "No such command" in result.output
        assert "Did you mean one of these?" in result.output
        assert "suggestion_1" in result.output
        assert "suggestion_2" in result.output

    def test_not_found_closet_match_singular(self, mocker: Any) -> None:
        """Check that calling a nonexistent command with a close match has the proper wording"""
        patched_difflib = mocker.patch(
            "kedro.framework.cli.utils.difflib.get_close_matches",
            return_value=["suggestion_1"],
        )

        cmd_collection = CommandCollection(("Commands", [cli, stub_cli]))
        result = CliRunner().invoke(cmd_collection, ["not_found"])

        patched_difflib.assert_called_once_with(
            "not_found", mocker.ANY, mocker.ANY, mocker.ANY
        )

        assert result.exit_code == 2
        assert "No such command" in result.output
        assert "Did you mean this?" in result.output
        assert "suggestion_1" in result.output

    def test_help(self) -> None:
        """Check that help output includes stub_cli group description."""
        cmd_collection = CommandCollection(("Commands", [cli, stub_cli]))
        result = CliRunner().invoke(cmd_collection, [])
        assert result.exit_code == 0
        assert "Stub CLI group description" in result.output
        assert "Kedro is a CLI" in result.output


class TestForwardCommand:
    def test_regular(self) -> None:
        """Test forwarded command invocation."""
        result = CliRunner().invoke(stub_cli, ["forwarded_command", "bob"])
        assert result.exit_code == 0, result.output
        assert "bob" in result.output
        assert "fred" in result.output
        assert "--help" not in result.output
        assert "forwarded_command" not in result.output

    def test_unnamed(self) -> None:
        """Test forwarded command invocation."""
        result = CliRunner().invoke(stub_cli, ["unnamed", "bob"])
        assert result.exit_code == 0, result.output
        assert "bob" in result.output
        assert "fred" in result.output
        assert "--help" not in result.output
        assert "forwarded_command" not in result.output

    def test_help(self) -> None:
        """Test help output for the command with help flags not forwarded."""
        result = CliRunner().invoke(stub_cli, ["forwarded_command", "bob", "--help"])
        assert result.exit_code == 0, result.output
        assert "bob" not in result.output
        assert "fred" not in result.output
        assert "--help" in result.output
        assert "forwarded_command" in result.output

    def test_forwarded_help(self) -> None:
        """Test help output for the command with forwarded help flags."""
        result = CliRunner().invoke(stub_cli, ["forwarded_help", "bob", "--help"])
        assert result.exit_code == 0, result.output
        assert "bob" in result.output
        assert "fred" in result.output
        assert "--help" in result.output
        assert "forwarded_help" not in result.output


class TestCliUtils:
    def test_get_pkg_version(self, requirements_file: Path) -> None:
        """Test get_pkg_version(), which extracts package version
        from the provided requirements file."""
        sa_version = "SQLAlchemy>=1.2.0, <2.0"
        assert get_pkg_version(requirements_file, "SQLAlchemy") == sa_version
        assert get_pkg_version(requirements_file, "pandas") == "pandas==0.23.0"
        assert get_pkg_version(requirements_file, "toposort") == "toposort"
        with raises(KedroCliError):
            get_pkg_version(requirements_file, "nonexistent")
        with raises(KedroCliError):
            non_existent_file = str(requirements_file) + "-nonexistent"
            get_pkg_version(non_existent_file, "pandas")

    def test_get_pkg_version_deprecated(self, requirements_file: Path) -> None:
        with warns(
            KedroDeprecationWarning,
            match=r"\`get_pkg_version\(\)\` has been deprecated",
        ):
            _ = get_pkg_version(requirements_file, "pandas")

    def test_clean_pycache(self, tmp_path: Path, mocker: Any) -> None:
        """Test `clean_pycache` utility function"""
        source = Path(tmp_path)
        pycache2 = Path(source / "nested1" / "nested2" / "__pycache__").resolve()
        pycache2.mkdir(parents=True)
        pycache1 = Path(source / "nested1" / "__pycache__").resolve()
        pycache1.mkdir()
        pycache = Path(source / "__pycache__").resolve()
        pycache.mkdir()

        mocked_rmtree = mocker.patch("shutil.rmtree")
        _clean_pycache(source)

        expected_calls = [
            mocker.call(pycache, ignore_errors=True),
            mocker.call(pycache1, ignore_errors=True),
            mocker.call(pycache2, ignore_errors=True),
        ]
        assert mocked_rmtree.mock_calls == expected_calls

    def test_find_run_command_non_existing_project(self) -> None:
        with pytest.raises(ModuleNotFoundError, match="No module named 'fake_project'"):
            _ = find_run_command("fake_project")

    def test_find_run_command_with_clipy(
        self, fake_metadata: Any, fake_repo_path: Path, fake_project_cli: Any, mocker: Any
    ) -> None:
        mocker.patch("kedro.framework.cli.cli._is_project", return_value=True)
        mocker.patch(
            "kedro.framework.cli.cli.bootstrap_project", return_value=fake_metadata
        )

        mock_project_cli = MagicMock(spec=[fake_repo_path / "cli.py"])
        mock_project_cli.cli = MagicMock(spec=["cli"])
        mock_project_cli.run = MagicMock(spec=["run"])
        mocker.patch(
            "kedro.framework.cli.utils.importlib.import_module",
            return_value=mock_project_cli,
        )

        run = find_run_command(fake_metadata.package_name)
        assert run is mock_project_cli.run

    def test_find_run_command_no_clipy(
        self, fake_metadata: Any, fake_repo_path: Path, mocker: Any
    ) -> None:
        mocker.patch("kedro.framework.cli.cli._is_project", return_value=True)
        mocker.patch(
            "kedro.framework.cli.cli.bootstrap_project", return_value=fake_metadata
        )
        mock_project_cli = MagicMock(spec=[fake_repo_path / "cli.py"])
        mocker.patch(
            "kedro.framework.cli.utils.importlib.import_module",
            return_value=mock_project_cli,
        )

        with raises(KedroCliError, match="Cannot load commands from"):
            _ = find_run_command(fake_metadata.package_name)

    def test_find_run_command_use_plugin_run(
        self, fake_metadata: Any, fake_repo_path: Path, mocker: Any
    ) -> None:
        mock_plugin = MagicMock(spec=["plugins"])
        mock_command = MagicMock(name="run_command")
        mock_plugin.commands = {"run": mock_command}
        mocker.patch(
            "kedro.framework.cli.utils.load_entry_points", return_value=[mock_plugin]
        )

        mocker.patch("kedro.framework.cli.cli._is_project", return_value=True)
        mocker.patch(
            "kedro.framework.cli.cli.bootstrap_project", return_value=fake_metadata
        )
        mocker.patch(
            "kedro.framework.cli.cli.importlib.import_module",
            side_effect=ModuleNotFoundError("dummy_package.cli"),
        )

        run = find_run_command(fake_metadata.package_name)
        assert run == mock_command

    def test_find_run_command_use_default_run(self, fake_metadata: Any, mocker: Any) -> None:
        mocker.patch("kedro.framework.cli.cli._is_project", return_value=True)
        mocker.patch(
            "kedro.framework.cli.cli.bootstrap_project", return_value=fake_metadata
        )
        mocker.patch(
            "kedro.framework.cli.cli.importlib.import_module",
            side_effect=ModuleNotFoundError("dummy_package.cli"),
        )
        run = find_run_command(fake_metadata.package_name)
        assert run.help == "Run the pipeline."


class TestEntryPoints:
    def test_project_groups(self, entry_points: Any, entry_point: Any) -> None:
        entry_point.load.return_value = "groups"
        groups = load_entry_points("project")
        assert groups == ["groups"]
        entry_points.return_value.select.assert_called_once_with(
            group="kedro.project_commands"
        )

    def test_project_error_is_caught(
        self, entry_points: Any, entry_point: Any, caplog: Any
    ) -> None:
        entry_point.load.side_effect = Exception()
        entry_point.module = "project"
        load_entry_points("project")
        assert "Failed to load project commands" in caplog.text
        entry_points.return_value.select.assert_called_once_with(
            group="kedro.project_commands"
        )

    def test_global_groups(self, entry_points: Any, entry_point: Any) -> None:
        entry_point.load.return_value = "groups"
        groups = load_entry_points("global")
        assert groups == ["groups"]
        entry_points.return_value.select.assert_called_once_with(
            group="kedro.global_commands"
        )

    def test_global_error_is_caught(
        self, entry_points: Any, entry_point: Any, caplog: Any
    ) -> None:
        entry_point.load.side_effect = Exception()
        entry_point.module = "global"
        load_entry_points("global")
        assert "Failed to load global commands" in caplog.text
        entry_points.return_value.select.assert_called_once_with(
            group="kedro.global_commands"
        )

    def test_init(self, entry_points: Any, entry_point: Any) -> None:
        _init_plugins()
        entry_points.return_value.select.assert_called_once_with(group="kedro.init")
        entry_point.load().assert_called_once_with()

    def test_init_error_is_caught(self, entry_points: Any, entry_point: Any) -> None:
        entry_point.load.return_value.side_effect = Exception()
        with raises(Exception):
            _init_plugins()
        entry_points.return_value.select.assert_called_once_with(group="kedro.init")


class TestKedroCLI:
    def test_project_commands_no_clipy(self, mocker: Any, fake_metadata: Any) -> None:
        mocker.patch("kedro.framework.cli.cli._is_project", return_value=True)
        mocker.patch(
            "kedro.framework.cli.cli.bootstrap_project", return_value=fake_metadata
        )
        mocker.patch(
            "kedro.framework.cli.cli.importlib.import_module",
            side_effect=cycle([ModuleNotFoundError()]),
        )
        kedro_cli = KedroCLI(fake_metadata.project_path)
        # There is only one `LazyGroup` for project commands
        assert len(kedro_cli.project_groups) == 1
        assert kedro_cli.project_groups == [project_commands]
        # Assert