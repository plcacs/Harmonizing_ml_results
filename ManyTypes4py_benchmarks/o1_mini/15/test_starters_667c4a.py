"""This module contains unit test for the cli command 'kedro new'"""
from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pytest
import requests
import toml
import yaml
from click.testing import CliRunner
from cookiecutter.exceptions import RepositoryCloneFailed
from kedro import __version__ as version
from kedro.framework.cli.starters import (
    _OFFICIAL_STARTER_SPECS_DICT,
    TEMPLATE_PATH,
    KedroStarterSpec,
    _convert_tool_short_names_to_numbers,
    _fetch_validate_parse_config_from_user_prompts,
    _get_latest_starters_version,
    _kedro_version_equal_or_lower_to_starters,
    _make_cookiecutter_args_and_fetch_template,
    _parse_tools_input,
    _parse_yes_no_to_bool,
    _select_checkout_branch_for_cookiecutter,
    _validate_tool_selection,
)
FILES_IN_TEMPLATE_WITH_NO_TOOLS: int = 15


@pytest.fixture
def chdir_to_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def mock_determine_repo_dir(mocker: pytest.MockFixture) -> pytest.Mock:
    return mocker.patch(
        "cookiecutter.repository.determine_repo_dir",
        return_value=(str(TEMPLATE_PATH), None),
    )


@pytest.fixture
def mock_cookiecutter(mocker: pytest.MockFixture) -> pytest.Mock:
    return mocker.patch("cookiecutter.main.cookiecutter")


@pytest.fixture
def patch_cookiecutter_args(mocker: pytest.MockFixture) -> None:
    mocker.patch(
        "kedro.framework.cli.starters._make_cookiecutter_args_and_fetch_template",
        side_effect=mock_make_cookiecutter_args_and_fetch_template,
    )


def mock_make_cookiecutter_args_and_fetch_template(
    *args: Any, **kwargs: Any
) -> Tuple[Dict[str, Any], str]:
    cookiecutter_args, starter_path = _make_cookiecutter_args_and_fetch_template(
        *args, **kwargs
    )
    cookiecutter_args["checkout"] = "main"
    return (cookiecutter_args, starter_path)


def _clean_up_project(project_dir: Path) -> None:
    if project_dir.is_dir():
        shutil.rmtree(str(project_dir), ignore_errors=True)


def _write_yaml(filepath: Path, config: Dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    yaml_str: str = yaml.dump(config)
    filepath.write_text(yaml_str)


def _make_cli_prompt_input(
    tools: str = "none",
    project_name: str = "",
    example_pipeline: str = "no",
    repo_name: str = "",
    python_package: str = "",
) -> str:
    return "\n".join([project_name, tools, example_pipeline, repo_name, python_package])


def _make_cli_prompt_input_without_tools(
    project_name: str = "", repo_name: str = "", python_package: str = ""
) -> str:
    return "\n".join([project_name, repo_name, python_package])


def _make_cli_prompt_input_without_name(
    tools: str = "none",
    repo_name: str = "",
    python_package: str = "",
) -> str:
    return "\n".join([tools, repo_name, python_package])


def _get_expected_files(tools: str, example_pipeline: str) -> int:
    tools_template_files: Dict[str, int] = {
        "1": 0,
        "2": 3,
        "3": 1,
        "4": 2,
        "5": 8,
        "6": 2,
        "7": 0,
    }
    tools_list: List[str] = _parse_tools_input(tools)
    example_pipeline_bool: bool = _parse_yes_no_to_bool(example_pipeline)  # type: ignore
    expected_files: int = FILES_IN_TEMPLATE_WITH_NO_TOOLS
    for tool in tools_list:
        expected_files += tools_template_files[tool]
    if example_pipeline_bool and "5" not in tools_list:
        expected_files += tools_template_files["5"]
    example_files_count: List[int] = [3, 2, 6]
    if example_pipeline_bool:
        expected_files += sum(example_files_count)
        expected_files += 4 if "7" in tools_list else 0
        expected_files += 1 if "2" in tools_list else 0
    return expected_files


def _assert_requirements_ok(
    result: pytest.runner.CallInfo,
    tools: str = "none",
    repo_name: str = "new-kedro-project",
    output_dir: str = ".",
) -> None:
    assert result.exit_code == 0, result.output
    root_path: Path = (Path(output_dir) / repo_name).resolve()
    assert "Congratulations!" in result.output
    assert (
        f"has been created in the directory \n{root_path}"
        in result.output
    )
    pyproject_file_path: Path = root_path / "pyproject.toml"
    tools_list: List[str] = _parse_tools_input(tools)
    if "1" in tools_list:
        pyproject_config: Dict[str, Any] = toml.load(pyproject_file_path)
        expected: Dict[str, Any] = {
            "tool": {
                "ruff": {
                    "line-length": 88,
                    "show-fixes": True,
                    "select": ["F", "W", "E", "I", "UP", "PL", "T201"],
                    "ignore": ["E501"],
                    "format": {"docstring-code-format": True},
                }
            }
        }
        assert expected["tool"]["ruff"] == pyproject_config["tool"]["ruff"]
        assert (
            "ruff~=0.1.8" in pyproject_config["project"]["optional-dependencies"]["dev"]
        )
    if "2" in tools_list:
        pyproject_config: Dict[str, Any] = toml.load(pyproject_file_path)
        expected: Dict[str, Any] = {
            "pytest": {
                "ini_options": {
                    "addopts": "--cov-report term-missing --cov src/new_kedro_project -ra"
                }
            },
            "coverage": {
                "report": {
                    "fail_under": 0,
                    "show_missing": True,
                    "exclude_lines": ["pragma: no cover", "raise NotImplementedError"],
                }
            },
        }
        assert expected["pytest"] == pyproject_config["tool"]["pytest"]
        assert expected["coverage"] == pyproject_config["tool"]["coverage"]
        assert (
            "pytest-cov~=3.0"
            in pyproject_config["project"]["optional-dependencies"]["dev"]
        )
        assert (
            "pytest-mock>=1.7.1, <2.0"
            in pyproject_config["project"]["optional-dependencies"]["dev"]
        )
        assert (
            "pytest~=7.2" in pyproject_config["project"]["optional-dependencies"]["dev"]
        )
    if "4" in tools_list:
        pyproject_config: Dict[str, Any] = toml.load(pyproject_file_path)
        expected: Dict[str, Any] = {
            "optional-dependencies": {
                "docs": [
                    "docutils<0.21",
                    "sphinx>=5.3,<7.3",
                    "sphinx_rtd_theme==2.0.0",
                    "nbsphinx==0.8.1",
                    "sphinx-autodoc-typehints==1.20.2",
                    "sphinx_copybutton==0.5.2",
                    "ipykernel>=5.3, <7.0",
                    "Jinja2<3.2.0",
                    "myst-parser>=1.0,<2.1",
                ]
            }
        }
        assert (
            expected["optional-dependencies"]["docs"]
            == pyproject_config["project"]["optional-dependencies"]["docs"]
        )


def _assert_template_ok(
    result: pytest.runner.CallInfo,
    tools: str = "none",
    project_name: str = "New Kedro Project",
    example_pipeline: str = "no",
    repo_name: str = "new-kedro-project",
    python_package: str = "new_kedro_project",
    kedro_version: str = version,
    output_dir: str = ".",
) -> None:
    assert result.exit_code == 0, result.output
    full_path: Path = (Path(output_dir) / repo_name).resolve()
    assert "Congratulations!" in result.output
    assert (
        f"Your project '{project_name}' has been created in the directory \n{full_path}"
        in result.output
    )
    if "y" in example_pipeline.lower():
        assert (
            "It has been created with an example pipeline." in result.output
        )
    else:
        assert (
            "It has been created with an example pipeline." not in result.output
        )
    generated_files: List[Path] = [
        p
        for p in full_path.rglob("*")
        if p.is_file() and p.name != ".DS_Store"
    ]
    assert len(generated_files) == _get_expected_files(tools, example_pipeline)
    assert full_path.exists()
    assert (full_path / ".gitignore").is_file()
    assert project_name in (full_path / "README.md").read_text(
        encoding="utf-8"
    )
    assert "KEDRO" in (full_path / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert kedro_version in (
        full_path / "requirements.txt"
    ).read_text(encoding="utf-8")
    assert (
        (full_path / "src" / python_package / "__init__.py").is_file()
    )


def _assert_name_ok(result: pytest.runner.CallInfo, project_name: str = "New Kedro Project") -> None:
    assert result.exit_code == 0, result.output
    assert "Congratulations!" in result.output
    assert (
        f"Your project '{project_name}' has been created in the directory"
        in result.output
    )


def test_starter_list(fake_kedro_cli: Callable[..., Any]) -> None:
    """Check that `kedro starter list` prints out all starter aliases."""
    result: pytest.runner.CallInfo = CliRunner().invoke(fake_kedro_cli, ["starter", "list"])
    assert result.exit_code == 0, result.output
    for alias in _OFFICIAL_STARTER_SPECS_DICT:
        assert alias in result.output


def test_starter_list_with_starter_plugin(
    fake_kedro_cli: Callable[..., Any],
    entry_point: Any,
) -> None:
    """Check that `kedro starter list` prints out the plugin starters."""
    entry_point.load.return_value = [
        KedroStarterSpec("valid_starter", "valid_path")
    ]
    entry_point.module = "valid_starter_module"
    result: pytest.runner.CallInfo = CliRunner().invoke(fake_kedro_cli, ["starter", "list"])
    assert result.exit_code == 0, result.output
    assert "valid_starter_module" in result.output


@pytest.mark.parametrize(
    "specs,expected",
    [
        (
            [{"alias": "valid_starter", "template_path": "valid_path"}],
            "should be a 'KedroStarterSpec'",
        ),
        (
            [
                KedroStarterSpec("duplicate", "duplicate"),
                KedroStarterSpec("duplicate", "duplicate"),
            ],
            "has been ignored as it is already defined by",
        ),
    ],
)
def test_starter_list_with_invalid_starter_plugin(
    fake_kedro_cli: Callable[..., Any],
    entry_point: Any,
    specs: List[Union[Dict[str, str], KedroStarterSpec]],
    expected: str,
) -> None:
    """Check that `kedro starter list` prints out the plugin starters."""
    entry_point.load.return_value = specs
    entry_point.module = "invalid_starter"
    result: pytest.runner.CallInfo = CliRunner().invoke(fake_kedro_cli, ["starter", "list"])
    assert result.exit_code == 0, result.output
    assert expected in result.output


class TestParseToolsInput:
    @pytest.mark.parametrize(
        "input,expected",
        [
            ("1", ["1"]),
            ("1,2,3", ["1", "2", "3"]),
            ("2-4", ["2", "3", "4"]),
            ("3-3", ["3"]),
            ("all", ["1", "2", "3", "4", "5", "6", "7"]),
            ("none", []),
        ],
    )
    def test_parse_tools_valid(
        self, input: str, expected: List[str]
    ) -> None:
        result: List[str] = _parse_tools_input(input)
        assert result == expected

    @pytest.mark.parametrize("input", ["5-2", "3-1"])
    def test_parse_tools_invalid_range(
        self, input: str, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit):
            _parse_tools_input(input)
        message: str = (
            f"'{input}' is an invalid range for project tools.\n"
            "Please ensure range values go from smaller to larger."
        )
        captured = capsys.readouterr()
        assert message in captured.err

    @pytest.mark.parametrize(
        "input,right_border",
        [("3-9", "9"), ("3-10000", "10000")],
    )
    def test_parse_tools_range_too_high(
        self, input: str, right_border: str, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit):
            _parse_tools_input(input)
        message: str = (
            f"'{right_border}' is not a valid selection.\n"
            "Please select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        )
        captured = capsys.readouterr()
        assert message in captured.err

    @pytest.mark.parametrize(
        "input,last_invalid",
        [("0,3,5", "0"), ("1,3,8", "8"), ("0-4", "0")],
    )
    def test_parse_tools_invalid_selection(
        self, input: str, last_invalid: str, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit):
            selected: List[str] = _parse_tools_input(input)
            _validate_tool_selection(selected)
        message: str = (
            f"'{last_invalid}' is not a valid selection.\n"
            "Please select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        )
        captured = capsys.readouterr()
        assert message in captured.err


@pytest.mark.usefixtures("chdir_to_tmp")
class TestNewFromUserPromptsValid:
    """Tests for running `kedro new` interactively."""

    def test_default(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test new project creation using default New Kedro Project options."""
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new"], input=_make_cli_prompt_input()
        )
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    def test_custom_project_name(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(project_name="My Project"),
        )
        _assert_template_ok(
            result,
            project_name="My Project",
            repo_name="my-project",
            python_package="my_project",
        )
        _clean_up_project(Path("./my-project"))

    def test_custom_project_name_with_hyphen_and_underscore_and_number(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(project_name="My-Project_ 1"),
        )
        _assert_template_ok(
            result,
            project_name="My-Project_ 1",
            repo_name="my-project--1",
            python_package="my_project__1",
        )
        _clean_up_project(Path("./my-project--1"))

    def test_no_prompts(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        (Path("template") / "prompts.yml").unlink()
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "--starter", "template"]
        )
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    def test_empty_prompts(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        _write_yaml(Path("template") / "prompts.yml", {})
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "--starter", "template"]
        )
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    @pytest.mark.parametrize(
        "regex,valid_value",
        [
            (r"^\w+(-*\w+)*$", "my-value"),
            (r"^[A-Z_]+", "MY-VALUE"),
            (r"^\d+$", "123"),
        ],
    )
    def test_custom_prompt_valid_input(
        self,
        fake_kedro_cli: Callable[..., Any],
        regex: str,
        valid_value: str,
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        _write_yaml(
            Path("template") / "prompts.yml",
            {
                "project_name": {"title": "Project Name"},
                "custom_value": {
                    "title": "Custom Value",
                    "regex_validator": regex,
                },
            },
        )
        custom_input: str = "\n".join([valid_value, "My Project"])
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "template"],
            input=custom_input,
        )
        _assert_template_ok(
            result,
            project_name="My Project",
            repo_name="my-project",
            python_package="my_project",
        )
        _clean_up_project(Path("./my-project"))

    def test_custom_prompt_for_essential_variable(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        _write_yaml(
            Path("template") / "prompts.yml",
            {
                "project_name": {"title": "Project Name"},
                "repo_name": {
                    "title": "Custom Repo Name",
                    "regex_validator": r"^[a-zA-Z_]\w{1,}$",
                },
            },
        )
        custom_input: str = "\n".join(["My Project", "my_custom_repo"])
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "template"],
            input=custom_input,
        )
        _assert_template_ok(
            result,
            project_name="My Project",
            repo_name="my_custom_repo",
            python_package="my_project",
        )
        _clean_up_project(Path("./my_custom_repo"))

    def test_fetch_validate_parse_config_from_user_prompts_without_context(
        self,
    ) -> None:
        required_prompts: Dict[str, Any] = {}
        message: str = "No cookiecutter context available."
        with pytest.raises(Exception, match=message):
            _fetch_validate_parse_config_from_user_prompts(
                prompts=required_prompts, cookiecutter_context=None
            )


@pytest.mark.usefixtures("chdir_to_tmp")
class TestNewFromUserPromptsInvalid:
    def test_fail_if_dir_exists(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Check the error if the output directory already exists."""
        Path("new-kedro-project").mkdir()
        (Path("new-kedro-project") / "empty_file").touch()
        old_contents: List[Path] = list(Path("new-kedro-project").iterdir())
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-v"], input=_make_cli_prompt_input()
        )
        assert list(Path("new-kedro-project").iterdir()) == old_contents
        assert result.exit_code != 0
        assert "directory already exists" in result.output

    def test_cookiecutter_exception_if_no_verbose(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Check if the original cookiecutter exception is present in the output
        if no verbose flag is provided."""
        Path("new-kedro-project").mkdir()
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new"], input=_make_cli_prompt_input()
        )
        assert "cookiecutter.exceptions" in result.output

    def test_prompt_no_title(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        _write_yaml(Path("template") / "prompts.yml", {"repo_name": {}})
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "--starter", "template"]
        )
        assert result.exit_code != 0
        assert (
            "Each prompt must have a title field to be valid" in result.output
        )

    def test_prompt_bad_yaml(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        (Path("template") / "prompts.yml").write_text(
            "invalid\tyaml", encoding="utf-8"
        )
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "--starter", "template"]
        )
        assert result.exit_code != 0
        assert "Failed to generate project: could not load prompts.yml" in result.output

    def test_invalid_project_name_special_characters(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(project_name="My $Project!"),
        )
        assert result.exit_code != 0
        assert (
            "is an invalid value for project name.\n"
            "It must contain only alphanumeric symbols"
            in result.output
        )

    def test_invalid_project_name_too_short(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(project_name="P"),
        )
        assert result.exit_code != 0
        assert (
            "is an invalid value for project name.\n"
            "It must contain only alphanumeric symbols"
            in result.output
        )

    def test_custom_prompt_invalid_input(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        _write_yaml(
            Path("template") / "prompts.yml",
            {
                "project_name": {"title": "Project Name"},
                "custom_value": {
                    "title": "Custom Value",
                    "regex_validator": r"^\w+(-*\w+)*$",
                },
            },
        )
        custom_input: str = "\n".join(["My Project", "My Project"])
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "template"],
            input=custom_input,
        )
        assert result.exit_code != 0
        assert "'My Project' is an invalid value" in result.output


@pytest.mark.usefixtures("chdir_to_tmp")
class TestNewFromConfigFileValid:
    """Test `kedro new` with config file provided."""

    def test_required_keys_only(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-v", "--config", "config.yml"]
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./my-project"))

    def test_custom_required_keys(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "Project X",
            "example_pipeline": "no",
            "repo_name": "projectx",
            "python_package": "proj_x",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-v", "--config", "config.yml"]
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./projectx"))

    def test_custom_kedro_version(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
            "kedro_version": "my_version",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-v", "--config", "config.yml"]
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./my-project"))

    def test_custom_output_dir(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
            "output_dir": "my_output_dir",
        }
        _write_yaml(Path("config.yml"), config)
        Path("my_output_dir").mkdir()
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-v", "--config", "config.yml"]
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./my-project"))

    def test_extra_keys_allowed(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(
            Path("config.yml"),
            {**config, "extra_key": "my_extra_key"},
        )
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-v", "--config", "config.yml"]
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./my-project"))

    def test_no_prompts(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        config: Dict[str, Any] = {
            "project_name": "My Project",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        shutil.copytree(TEMPLATE_PATH, "template")
        (Path("template") / "prompts.yml").unlink()
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "template", "--config", "config.yml"],
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./my-project"))

    def test_empty_prompts(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        config: Dict[str, Any] = {
            "project_name": "My Project",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        shutil.copytree(TEMPLATE_PATH, "template")
        _write_yaml(Path("template") / "prompts.yml", {})
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "template", "--config", "config.yml"],
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./my-project"))

    def test_config_with_no_tools_example(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "project_name": "My Project",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-v", "--config", "config.yml"]
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./my-project"))


@pytest.mark.usefixtures("chdir_to_tmp")
class TestNewFromConfigFileInvalid:
    def test_output_dir_does_not_exist(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Check the error if the output directory is invalid."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
            "output_dir": "does_not_exist",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "-c", "config.yml"],
        )
        assert result.exit_code != 0
        assert "is not a valid output directory." in result.output

    def test_config_missing_key(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Check the error if keys are missing from config file."""
        config: Dict[str, Any] = {
            "tools": "none",
            "example_pipeline": "no",
            "python_package": "my_project",
            "repo_name": "my-project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "-c", "config.yml"],
        )
        assert result.exit_code != 0
        assert "project_name not found in config file" in result.output

    def test_config_does_not_exist(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Check the error if the config file does not exist."""
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-c", "missing.yml"]
        )
        assert result.exit_code != 0
        assert "Path 'missing.yml' does not exist" in result.output

    def test_config_empty(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Check the error if the config file is empty."""
        Path("config.yml").touch()
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli, ["new", "-c", "config.yml"]
        )
        assert result.exit_code != 0
        assert "Config file is empty" in result.output

    def test_config_bad_yaml(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Check the error if config YAML is invalid."""
        Path("config.yml").write_text("invalid\tyaml", encoding="utf-8")
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "-c", "config.yml"],
        )
        assert result.exit_code != 0
        assert "Failed to generate project: could not load config" in result.output

    def test_invalid_project_name_special_characters(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "My $Project!",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml"],
        )
        assert result.exit_code != 0
        assert (
            "is an invalid value for project name. It must contain only alphanumeric symbols, "
            "spaces, underscores and hyphens and be at least 2 characters long"
            in result.output
        )

    def test_invalid_project_name_too_short(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "P",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml"],
        )
        assert result.exit_code != 0
        assert (
            "is an invalid value for project name. It must contain only alphanumeric symbols, "
            "spaces, underscores and hyphens and be at least 2 characters long"
            in result.output
        )


@pytest.mark.usefixtures("chdir_to_tmp")
class TestNewWithStarterValid:
    def test_absolute_path(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--starter", str(Path("./template").resolve())],
            input=_make_cli_prompt_input(),
        )
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    def test_relative_path(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--starter", "template"],
            input=_make_cli_prompt_input(),
        )
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    def test_relative_path_directory(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--starter", ".", "--directory", "template"],
            input=_make_cli_prompt_input(),
        )
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    def test_alias(
        self,
        fake_kedro_cli: Callable[..., Any],
        mock_determine_repo_dir: pytest.Mock,
        mock_cookiecutter: pytest.Mock,
    ) -> None:
        CliRunner().invoke(
            fake_kedro_cli, ["new", "--starter", "spaceflights-pandas"], input=_make_cli_prompt_input()
        )
        kwargs: Dict[str, Any] = {
            "template": "git+https://github.com/kedro-org/kedro-starters.git",
            "directory": "spaceflights-pandas",
        }
        starters_version: Optional[str] = mock_determine_repo_dir.call_args[1].pop(
            "checkout", None
        )
        assert starters_version in [version, "main"]
        assert kwargs.items() <= mock_determine_repo_dir.call_args[1].items()
        assert kwargs.items() <= mock_cookiecutter.call_args[1].items()

    def test_alias_custom_checkout(
        self,
        fake_kedro_cli: Callable[..., Any],
        mock_determine_repo_dir: pytest.Mock,
        mock_cookiecutter: pytest.Mock,
    ) -> None:
        CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "spaceflights-pandas", "--checkout", "my_checkout"],
            input=_make_cli_prompt_input(),
        )
        kwargs: Dict[str, Any] = {
            "template": "git+https://github.com/kedro-org/kedro-starters.git",
            "checkout": "my_checkout",
            "directory": "spaceflights-pandas",
        }
        assert kwargs.items() <= mock_determine_repo_dir.call_args[1].items()
        assert kwargs.items() <= mock_cookiecutter.call_args[1].items()

    def test_git_repo(
        self,
        fake_kedro_cli: Callable[..., Any],
        mock_determine_repo_dir: pytest.Mock,
        mock_cookiecutter: pytest.Mock,
    ) -> None:
        CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "git+https://github.com/fake/fake.git"],
            input=_make_cli_prompt_input(),
        )
        kwargs: Dict[str, Optional[str]] = {
            "template": "git+https://github.com/fake/fake.git",
            "directory": None,
        }
        assert kwargs.items() <= mock_determine_repo_dir.call_args[1].items()
        del kwargs["directory"]
        assert kwargs.items() <= mock_cookiecutter.call_args[1].items()

    def test_git_repo_custom_checkout(
        self,
        fake_kedro_cli: Callable[..., Any],
        mock_determine_repo_dir: pytest.Mock,
        mock_cookiecutter: pytest.Mock,
    ) -> None:
        CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--starter",
                "git+https://github.com/fake/fake.git",
                "--checkout",
                "my_checkout",
            ],
            input=_make_cli_prompt_input(),
        )
        kwargs: Dict[str, Any] = {
            "template": "git+https://github.com/fake/fake.git",
            "checkout": "my_checkout",
            "directory": None,
        }
        assert kwargs.items() <= mock_determine_repo_dir.call_args[1].items()
        del kwargs["directory"]
        assert kwargs.items() <= mock_cookiecutter.call_args[1].items()

    def test_git_repo_custom_directory(
        self,
        fake_kedro_cli: Callable[..., Any],
        mock_determine_repo_dir: pytest.Mock,
        mock_cookiecutter: pytest.Mock,
    ) -> None:
        CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--starter",
                "git+https://github.com/fake/fake.git",
                "--directory",
                "my_directory",
            ],
            input=_make_cli_prompt_input(),
        )
        kwargs: Dict[str, Any] = {
            "template": "git+https://github.com/fake/fake.git",
            "directory": "my_directory",
        }
        assert kwargs.items() <= mock_determine_repo_dir.call_args[1].items()
        assert kwargs.items() <= mock_cookiecutter.call_args[1].items()

    def test_no_hint(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--starter", str(Path("./template").resolve())],
            input=_make_cli_prompt_input(),
        )
        assert (
            "To skip the interactive flow you can run `kedro new` with\nkedro new --name=<your-project-name> --tools=<your-project-tools> --example=<yes/no>"
            not in result.output
        )
        assert "You have selected" not in result.output
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))


class TestNewWithStarterInvalid:
    def test_invalid_starter(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--starter", "invalid"],
            input=_make_cli_prompt_input(),
        )
        assert result.exit_code != 0
        assert "Kedro project template not found at invalid" in result.output

    @pytest.mark.parametrize(
        "starter,repo",
        [
            (
                "spaceflights-pandas",
                "https://github.com/kedro-org/kedro-starters.git",
            ),
            (
                "git+https://github.com/fake/fake.git",
                "https://github.com/fake/fake.git",
            ),
        ],
    )
    def test_invalid_checkout(
        self,
        starter: str,
        repo: str,
        fake_kedro_cli: Callable[..., Any],
        mocker: pytest.MockFixture,
    ) -> None:
        mocker.patch(
            "cookiecutter.repository.determine_repo_dir",
            side_effect=RepositoryCloneFailed,
        )
        mock_ls_remote: Any = mocker.patch("git.cmd.Git").return_value.ls_remote
        mock_ls_remote.return_value = "tag1\ntag2"
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "-v",
                "--starter",
                starter,
                "--checkout",
                "invalid",
            ],
            input=_make_cli_prompt_input(),
        )
        assert result.exit_code != 0
        assert (
            "Specified tag invalid. The following tags are available: tag1, tag2"
            in result.output
        )
        mock_ls_remote.assert_called_with("--tags", repo)


class TestFlagsNotAllowed:
    def test_checkout_flag_without_starter(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--checkout", "some-checkout"],
            input=_make_cli_prompt_input(),
        )
        assert result.exit_code != 0
        assert (
            "Cannot use the --checkout flag without a --starter value."
            in result.output
        )

    def test_directory_flag_without_starter(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--directory", "some-directory"],
            input=_make_cli_prompt_input(),
        )
        assert result.exit_code != 0
        assert (
            "Cannot use the --directory flag without a --starter value."
            in result.output
        )

    def test_directory_flag_with_starter_alias(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--starter",
                "spaceflights-pandas",
                "--directory",
                "some-dir",
            ],
            input=_make_cli_prompt_input(),
        )
        assert result.exit_code != 0
        assert (
            "Cannot use the --directory flag with a --starter alias"
            in result.output
        )

    def test_starter_flag_with_tools_flag(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--tools", "all", "--starter", "spaceflights-pandas"],
            input=_make_cli_prompt_input(),
        )
        assert result.exit_code != 0
        assert (
            "Cannot use the --starter flag with the --example and/or --tools flag."
            in result.output
        )

    def test_starter_flag_with_example_flag(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--starter", "spaceflights-pandas", "--example", "no"],
            input=_make_cli_prompt_input(),
        )
        assert result.exit_code != 0
        assert (
            "Cannot use the --starter flag with the --example and/or --tools flag."
            in result.output
        )


@pytest.mark.usefixtures("chdir_to_tmp", "patch_cookiecutter_args")
class TestToolsAndExampleFromUserPrompts:
    @pytest.mark.parametrize(
        "tools",
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "2,3,4",
            "3-5",
            "1,2,4-6",
            "1,2,4-6,7",
            "4-6,7",
            "1, 2 ,4 - 6, 7",
            "1-3, 5-7",
            "all",
            "1, 2, 3",
            "  1,  2, 3  ",
            "ALL",
            "none",
            "",
        ],
    )
    @pytest.mark.parametrize("example_pipeline", ["Yes", "No"])
    def test_valid_tools_and_example(
        self,
        fake_kedro_cli: Callable[..., Any],
        tools: str,
        example_pipeline: str,
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(tools=tools, example_pipeline=example_pipeline),
        )
        _assert_template_ok(
            result,
            tools=tools,
            example_pipeline=example_pipeline,
        )
        _assert_requirements_ok(result, tools=tools)
        if tools not in ("none", ""):
            assert "You have selected the following project tools:" in result.output
        else:
            assert "You have selected no project tools" in result.output
        assert (
            "To skip the interactive flow you can run `kedro new` with\n"
            "kedro new --name=<your-project-name> --tools=<your-project-tools> --example=<yes/no>"
            in result.output
        )
        _clean_up_project(Path("./new-kedro-project"))

    @pytest.mark.parametrize("bad_input", ["bad input", "-1", "3-", "1 1"])
    def test_invalid_tools(
        self, fake_kedro_cli: Callable[..., Any], bad_input: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(tools=bad_input),
        )
        assert result.exit_code != 0
        assert "is an invalid value for project tools." in result.output
        assert (
            "Invalid input. Please select valid options for project tools using comma-separated values, ranges, or 'all/none'.\n"
            in result.output
        )

    @pytest.mark.parametrize(
        "input,last_invalid",
        [
            ("0,3,5", "0"),
            ("1,3,9", "9"),
            ("0-4", "0"),
            ("99", "99"),
            ("3-9", "9"),
            ("3-10000", "10000"),
        ],
    )
    def test_invalid_tools_selection(
        self, fake_kedro_cli: Callable[..., Any], input: str, last_invalid: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(tools=input),
        )
        assert result.exit_code != 0
        message: str = (
            f"'{last_invalid}' is not a valid selection.\n"
            "Please select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        )
        assert message in result.output

    @pytest.mark.parametrize("input", ["5-2", "3-1"])
    def test_invalid_tools_range(
        self, fake_kedro_cli: Callable[..., Any], input: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(tools=input),
        )
        assert result.exit_code != 0
        message: str = (
            f"'{input}' is an invalid range for project tools.\n"
            "Please ensure range values go from smaller to larger."
        )
        assert message in result.output

    @pytest.mark.parametrize(
        "example_pipeline",
        ["y", "n", "N", "YEs", "   yEs   "],
    )
    def test_valid_example(
        self, fake_kedro_cli: Callable[..., Any], example_pipeline: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(example_pipeline=example_pipeline),
        )
        _assert_template_ok(result, example_pipeline=example_pipeline)
        _clean_up_project(Path("./new-kedro-project"))

    @pytest.mark.parametrize("bad_input", ["bad input", "Not", "ye", "True", "t"])
    def test_invalid_example(
        self, fake_kedro_cli: Callable[..., Any], bad_input: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new"],
            input=_make_cli_prompt_input(example_pipeline=bad_input),
        )
        assert result.exit_code != 0
        assert (
            f"'{bad_input}' is an invalid value for example pipeline."
            in result.output
        )
        assert (
            "It must contain only y, n, YES, or NO (case insensitive).\n"
            in result.output
        )


@pytest.mark.usefixtures("chdir_to_tmp", "patch_cookiecutter_args")
class TestToolsAndExampleFromConfigFile:
    @pytest.mark.parametrize(
        "tools,example_pipeline",
        [
            (
                "lint",
                "Yes",
            ),
            (
                "test",
                "Yes",
            ),
            (
                "tests",
                "Yes",
            ),
            (
                "log",
                "Yes",
            ),
            (
                "logs",
                "Yes",
            ),
            (
                "docs",
                "Yes",
            ),
            (
                "doc",
                "Yes",
            ),
            (
                "data",
                "Yes",
            ),
            (
                "pyspark",
                "Yes",
            ),
            (
                "viz",
                "Yes",
            ),
            (
                "tests,logs,doc",
                "Yes",
            ),
            (
                "test,data,lint",
                "Yes",
            ),
            (
                "log,docs,data,test,lint",
                "Yes",
            ),
            (
                "log, docs, data, test, lint",
                "Yes",
            ),
            (
                "log,       docs,     data,   test,     lint",
                "Yes",
            ),
            (
                "1-3, 5-7",
                "Yes",
            ),
            (
                "all",
                "Yes",
            ),
            (
                "LINT",
                "Yes",
            ),
            (
                "ALL",
                "Yes",
            ),
            (
                "TEST, LOG, DOCS",
                "Yes",
            ),
            (
                "test, DATA, liNt",
                "Yes",
            ),
            (
                "none",
                "Yes",
            ),
            (
                "NONE",
                "Yes",
            ),
            (
                "lint",
                "No",
            ),
            (
                "test",
                "No",
            ),
            (
                "tests",
                "No",
            ),
            (
                "log",
                "No",
            ),
            (
                "logs",
                "No",
            ),
            (
                "docs",
                "No",
            ),
            (
                "doc",
                "No",
            ),
            (
                "data",
                "No",
            ),
            (
                "pyspark",
                "No",
            ),
            (
                "viz",
                "No",
            ),
            (
                "tests,logs,doc",
                "No",
            ),
            (
                "test,data,lint",
                "No",
            ),
            (
                "log,docs,data,test,lint",
                "No",
            ),
            (
                "log, docs, data, test, lint",
                "No",
            ),
            (
                "log,       docs,     data,   test,     lint",
                "No",
            ),
            (
                "1-3, 5-7",
                "No",
            ),
            (
                "all",
                "No",
            ),
            (
                "LINT",
                "No",
            ),
            (
                "ALL",
                "No",
            ),
            (
                "TEST, LOG, DOCS",
                "No",
            ),
            (
                "test, DATA, liNt",
                "No",
            ),
            (
                "none",
                "No",
            ),
            (
                "NONE",
                "No",
            ),
        ],
    )
    def test_valid_tools_and_example(
        self,
        fake_kedro_cli: Callable[..., Any],
        tools: str,
        example_pipeline: str,
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": tools,
            "project_name": "New Kedro Project",
            "example_pipeline": example_pipeline,
            "repo_name": "new-kedro-project",
            "python_package": "new_kedro_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml"],
        )
        tools_converted: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=tools
        )
        tools_converted_str: str = ",".join(tools_converted) if tools_converted else "none"
        _assert_template_ok(
            result,
            tools=tools_converted_str,
            example_pipeline=example_pipeline,
        )
        _assert_requirements_ok(
            result,
            tools=tools_converted_str,
            repo_name="new-kedro-project",
        )
        if tools not in ("none", "NONE", ""):
            assert "You have selected the following project tools:" in result.output
        else:
            assert "You have selected no project tools" in result.output
        assert (
            "To skip the interactive flow you can run `kedro new` with\n"
            "kedro new --name=<your-project-name> --tools=<your-project-tools> --example=<yes/no>"
            not in result.output
        )
        _clean_up_project(Path("./new-kedro-project"))

    @pytest.mark.parametrize("bad_input", ["bad input", "0,3,5", "1,3,9", "llint", "tes", "test, lin"])
    def test_invalid_tools(
        self, fake_kedro_cli: Callable[..., Any], bad_input: str
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": bad_input,
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml"],
        )
        assert result.exit_code != 0
        assert (
            "Please select from the available tools: lint, test, log, docs, data, pyspark, viz, all, none"
            in result.output
        )

    def test_invalid_tools_with_starter(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        """Test project created from config with tools and example used with --starter"""
        config: Dict[str, Any] = {
            "tools": "all",
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        shutil.copytree(TEMPLATE_PATH, "template")
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml", "--starter=template"],
        )
        assert result.exit_code != 0
        assert (
            "The --starter flag can not be used with `example_pipeline` and/or `tools` keys "
            "in the config file.\n"
            in result.output
        )

    @pytest.mark.parametrize(
        "input",
        ["lint,all", "test,none", "all,none"],
    )
    def test_invalid_tools_flag_combination(
        self, fake_kedro_cli: Callable[..., Any], input: str
    ) -> None:
        config: Dict[str, Any] = {
            "tools": input,
            "project_name": "My Project",
            "example_pipeline": "no",
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml"],
        )
        assert result.exit_code != 0
        assert (
            "Tools options 'all' and 'none' cannot be used with other options"
            in result.output
        )

    @pytest.mark.parametrize(
        "example_pipeline",
        ["y", "n", "N", "YEs", "   yEs   "],
    )
    def test_valid_example(
        self, fake_kedro_cli: Callable[..., Any], example_pipeline: str
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "New Kedro Project",
            "example_pipeline": example_pipeline,
            "repo_name": "new-kedro-project",
            "python_package": "new_kedro_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml"],
        )
        _assert_template_ok(result, **config)
        _clean_up_project(Path("./new-kedro-project"))

    @pytest.mark.parametrize("bad_input", ["bad input", "Not", "ye", "True", "t"])
    def test_invalid_example(
        self, fake_kedro_cli: Callable[..., Any], bad_input: str
    ) -> None:
        """Test project created from config."""
        config: Dict[str, Any] = {
            "tools": "none",
            "project_name": "My Project",
            "example_pipeline": bad_input,
            "repo_name": "my-project",
            "python_package": "my_project",
        }
        _write_yaml(Path("config.yml"), config)
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "-v", "--config", "config.yml"],
        )
        assert result.exit_code != 0
        assert (
            f"'{bad_input}' is an invalid value for example pipeline."
            in result.output
        )
        assert (
            "It must contain only y, n, YES, or NO (case insensitive).\n"
            in result.output
        )


@pytest.mark.usefixtures("chdir_to_tmp", "patch_cookiecutter_args")
class TestToolsAndExampleFromCLI:
    @pytest.mark.parametrize(
        "tools,example_pipeline",
        [
            (
                "lint",
                "Yes",
            ),
            (
                "test",
                "Yes",
            ),
            (
                "tests",
                "Yes",
            ),
            (
                "log",
                "Yes",
            ),
            (
                "logs",
                "Yes",
            ),
            (
                "docs",
                "Yes",
            ),
            (
                "doc",
                "Yes",
            ),
            (
                "data",
                "Yes",
            ),
            (
                "pyspark",
                "Yes",
            ),
            (
                "viz",
                "Yes",
            ),
            (
                "tests,logs,doc",
                "Yes",
            ),
            (
                "test,data,lint",
                "Yes",
            ),
            (
                "log,docs,data,test,lint",
                "Yes",
            ),
            (
                "log, docs, data, test, lint",
                "Yes",
            ),
            (
                "log,       docs,     data,   test,     lint",
                "Yes",
            ),
            (
                "1-3, 5-7",
                "Yes",
            ),
            (
                "all",
                "Yes",
            ),
            (
                "LINT",
                "Yes",
            ),
            (
                "ALL",
                "Yes",
            ),
            (
                "TEST, LOG, DOCS",
                "Yes",
            ),
            (
                "test, DATA, liNt",
                "Yes",
            ),
            (
                "none",
                "Yes",
            ),
            (
                "NONE",
                "Yes",
            ),
            (
                "lint",
                "No",
            ),
            (
                "test",
                "No",
            ),
            (
                "tests",
                "No",
            ),
            (
                "log",
                "No",
            ),
            (
                "logs",
                "No",
            ),
            (
                "docs",
                "No",
            ),
            (
                "doc",
                "No",
            ),
            (
                "data",
                "No",
            ),
            (
                "pyspark",
                "No",
            ),
            (
                "viz",
                "No",
            ),
            (
                "tests,logs,doc",
                "No",
            ),
            (
                "test,data,lint",
                "No",
            ),
            (
                "log,docs,data,test,lint",
                "No",
            ),
            (
                "log, docs, data, test, lint",
                "No",
            ),
            (
                "log,       docs,     data,   test,     lint",
                "No",
            ),
            (
                "1-3, 5-7",
                "No",
            ),
            (
                "all",
                "No",
            ),
            (
                "LINT",
                "No",
            ),
            (
                "ALL",
                "No",
            ),
            (
                "TEST, LOG, DOCS",
                "No",
            ),
            (
                "test, DATA, liNt",
                "No",
            ),
            (
                "none",
                "No",
            ),
            (
                "NONE",
                "No",
            ),
        ],
    )
    def test_valid_tools_flag(
        self,
        fake_kedro_cli: Callable[..., Any],
        tools: str,
        example_pipeline: str,
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--tools", tools, "--example", example_pipeline],
            input=_make_cli_prompt_input_without_tools(),
        )
        tools_converted: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=tools
        )
        tools_converted_str: str = ",".join(tools_converted) if tools_converted else "none"
        _assert_template_ok(
            result,
            tools=tools_converted_str,
            example_pipeline=example_pipeline,
        )
        _assert_requirements_ok(
            result,
            tools=tools_converted_str,
            repo_name="new-kedro-project",
        )
        if tools_converted_str not in ("none", "NONE"):
            assert "You have selected the following project tools:" in result.output
        else:
            assert "You have selected no project tools" in result.output
        assert (
            "To skip the interactive flow you can run `kedro new` with\n"
            "kedro new --name=<your-project-name> --tools=<your-project-tools> --example=<yes/no>"
            in result.output
        )
        _clean_up_project(Path("./new-kedro-project"))

    def test_invalid_tools_flag(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--tools", "bad_input"],
            input=_make_cli_prompt_input_without_tools(),
        )
        assert result.exit_code != 0
        assert (
            "Please select from the available tools: lint, test, log, docs, data, pyspark, viz, all, none"
            in result.output
        )

    @pytest.mark.parametrize(
        "tools",
        ["lint,all", "test,none", "all,none"],
    )
    def test_invalid_tools_flag_combination(
        self, fake_kedro_cli: Callable[..., Any], tools: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--tools", tools],
            input=_make_cli_prompt_input_without_tools(),
        )
        assert result.exit_code != 0
        assert (
            "Tools options 'all' and 'none' cannot be used with other options"
            in result.output
        )

    def test_flags_skip_interactive_flow(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--name",
                "New Kedro Project",
                "--tools",
                "none",
                "--example",
                "no",
            ],
        )
        assert result.exit_code == 0
        assert (
            "To skip the interactive flow you can run `kedro new` with\n"
            "kedro new --name=<your-project-name> --tools=<your-project-tools> --example=<yes/no>"
            not in result.output
        )
        _clean_up_project(Path("./new-kedro-project"))


@pytest.mark.usefixtures("chdir_to_tmp", "patch_cookiecutter_args")
class TestNameFromCLI:
    @pytest.mark.parametrize(
        "name",
        [
            "readable_name",
            "Readable Name",
            "Readable-name",
            "readable_name_12233",
            "123ReadableName",
        ],
    )
    def test_valid_names(
        self, fake_kedro_cli: Callable[..., Any], name: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--name", name],
            input=_make_cli_prompt_input_without_name(),
        )
        repo_name: str = name.lower().replace(" ", "_").replace("-", "_")
        assert result.exit_code == 0
        _assert_name_ok(result, project_name=name)
        _clean_up_project(Path("./" + repo_name))

    @pytest.mark.parametrize(
        "name",
        ["bad_name$%!", "Bad.Name", ""],
    )
    def test_invalid_names(
        self, fake_kedro_cli: Callable[..., Any], name: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            ["new", "--name", name],
            input=_make_cli_prompt_input_without_name(),
        )
        assert result.exit_code != 0
        assert (
            "is an invalid value for project name. It must contain only alphanumeric symbols, "
            "spaces, underscores and hyphens and be at least 2 characters long"
            in result.output
        )


class TestParseYesNoToBools:
    @pytest.mark.parametrize(
        "input",
        ["yes", "YES", "y", "Y", "yEs"],
    )
    def parse_yes_no_to_bool_responds_true(
        self, input: str
    ) -> None:
        assert _parse_yes_no_to_bool(input) is True

    @pytest.mark.parametrize(
        "input",
        ["no", "NO", "n", "N", "No", ""],
    )
    def parse_yes_no_to_bool_responds_false(
        self, input: str
    ) -> None:
        assert _parse_yes_no_to_bool(input) is False

    def parse_yes_no_to_bool_responds_none(self) -> None:
        assert _parse_yes_no_to_bool(None) is None


class TestValidateSelection:
    def test_validate_tool_selection_valid(self) -> None:
        tools: List[str] = ["1", "2", "3", "4"]
        assert _validate_tool_selection(tools) is None

    def test_validate_tool_selection_invalid_single_tool(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tools: List[str] = ["8"]
        with pytest.raises(SystemExit):
            _validate_tool_selection(tools)
        message: str = (
            "is not a valid selection.\n"
            "Please select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        )
        captured = capsys.readouterr()
        assert message in captured.err

    def test_validate_tool_selection_invalid_multiple_tools(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tools: List[str] = ["8", "10", "15"]
        with pytest.raises(SystemExit):
            _validate_tool_selection(tools)
        message: str = (
            "is not a valid selection.\n"
            "Please select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        )
        captured = capsys.readouterr()
        assert message in captured.err

    def test_validate_tool_selection_mix_valid_invalid_tools(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tools: List[str] = ["1", "8", "3", "15"]
        with pytest.raises(SystemExit):
            _validate_tool_selection(tools)
        message: str = (
            "is not a valid selection.\n"
            "Please select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        )
        captured = capsys.readouterr()
        assert message in captured.err

    def test_validate_tool_selection_empty_list(
        self,
    ) -> None:
        tools: List[str] = []
        assert _validate_tool_selection(tools) is None


class TestConvertToolNamesToNumbers:
    def test_convert_tool_short_names_to_numbers_with_valid_tools(
        self,
    ) -> None:
        selected_tools: str = "lint,test,docs"
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == ["1", "2", "4"]

    def test_convert_tool_short_names_to_numbers_with_empty_string(
        self,
    ) -> None:
        selected_tools: str = ""
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == []

    def test_convert_tool_short_names_to_numbers_with_none_string(
        self,
    ) -> None:
        selected_tools: str = "none"
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == []

    def test_convert_tool_short_names_to_numbers_with_all_string(
        self,
    ) -> None:
        result: List[str] = _convert_tool_short_names_to_numbers("all")
        assert result == ["1", "2", "3", "4", "5", "6", "7"]

    def test_convert_tool_short_names_to_numbers_with_mixed_valid_invalid_tools(
        self,
    ) -> None:
        selected_tools: str = "lint,invalid_tool,docs"
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == ["1", "4"]

    def test_convert_tool_short_names_to_numbers_with_whitespace(
        self,
    ) -> None:
        selected_tools: str = " lint , test , docs "
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == ["1", "2", "4"]

    def test_convert_tool_short_names_to_numbers_with_case_insensitive_tools(
        self,
    ) -> None:
        selected_tools: str = "Lint,TEST,Docs"
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == ["1", "2", "4"]

    def test_convert_tool_short_names_to_numbers_with_invalid_tools(
        self,
    ) -> None:
        selected_tools: str = "invalid_tool1,invalid_tool2"
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == []

    def test_convert_tool_short_names_to_numbers_with_duplicates(
        self,
    ) -> None:
        selected_tools: str = "lint,test,tests"
        result: List[str] = _convert_tool_short_names_to_numbers(
            selected_tools=selected_tools
        )
        assert result == ["1", "2"]


@pytest.mark.usefixtures("chdir_to_tmp")
class TestTelemetryCLIFlag:
    @pytest.mark.parametrize(
        "input",
        ["yes", "YES", "y", "Y", "yEs"],
    )
    def test_flag_value_is_yes(
        self, fake_kedro_cli: Callable[..., Any], input: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--name",
                "New Kedro Project",
                "--tools",
                "none",
                "--example",
                "no",
                "--telemetry",
                input,
            ],
        )
        repo_name: str = "new-kedro-project"
        assert result.exit_code == 0
        telemetry_file_path: Path = Path(repo_name + "/.telemetry")
        assert telemetry_file_path.exists()
        with telemetry_file_path.open("r") as file:
            file_content: str = file.read()
            target_string: str = "consent: true"
        assert target_string in file_content
        _clean_up_project(Path("./" + repo_name))

    @pytest.mark.parametrize(
        "input",
        ["no", "NO", "n", "N", "No"],
    )
    def test_flag_value_is_no(
        self, fake_kedro_cli: Callable[..., Any], input: str
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--name",
                "New Kedro Project",
                "--tools",
                "none",
                "--example",
                "no",
                "--telemetry",
                input,
            ],
        )
        repo_name: str = "new-kedro-project"
        assert result.exit_code == 0
        telemetry_file_path: Path = Path(repo_name + "/.telemetry")
        assert telemetry_file_path.exists()
        with telemetry_file_path.open("r") as file:
            file_content: str = file.read()
            target_string: str = "consent: false"
        assert target_string in file_content
        _clean_up_project(Path("./" + repo_name))

    def test_flag_value_is_invalid(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--name",
                "New Kedro Project",
                "--tools",
                "none",
                "--example",
                "no",
                "--telemetry",
                "wrong",
            ],
        )
        repo_name: str = "new-kedro-project"
        assert result.exit_code == 2
        assert "'wrong' is not one of 'yes', 'no', 'y', 'n'" in result.output
        telemetry_file_path: Path = Path(repo_name + "/.telemetry")
        assert not telemetry_file_path.exists()

    def test_flag_is_absent(
        self, fake_kedro_cli: Callable[..., Any]
    ) -> None:
        result: pytest.runner.CallInfo = CliRunner().invoke(
            fake_kedro_cli,
            [
                "new",
                "--name",
                "New Kedro Project",
                "--tools",
                "none",
                "--example",
                "no",
            ],
        )
        repo_name: str = "new-kedro-project"
        assert result.exit_code == 0
        telemetry_file_path: Path = Path(repo_name + "/.telemetry")
        assert not telemetry_file_path.exists()


@pytest.fixture
def mock_env_vars(mocker: pytest.MockFixture) -> None:
    """Fixture to mock environment variables"""
    mocker.patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"}, clear=True)


class TestGetLatestStartersVersion:
    def test_get_latest_starters_version(
        self, mock_env_vars: None, requests_mock: Any
    ) -> None:
        """Test _get_latest_starters_version when KEDRO_STARTERS_VERSION is not set"""
        latest_version: str = "1.2.3"
        requests_mock.get(
            "https://api.github.com/repos/kedro-org/kedro-starters/releases/latest",
            json={"tag_name": latest_version},
            status_code=200,
        )
        result: str = _get_latest_starters_version()
        assert result == latest_version
        assert os.getenv("KEDRO_STARTERS_VERSION") == latest_version

    def test_get_latest_starters_version_with_env_set(
        self, mocker: pytest.MockFixture
    ) -> None:
        """Test _get_latest_starters_version when KEDRO_STARTERS_VERSION is already set"""
        expected_version: str = "1.2.3"
        mocker.patch.dict(os.environ, {"KEDRO_STARTERS_VERSION": expected_version})
        result: str = _get_latest_starters_version()
        assert result == expected_version

    def test_get_latest_starters_version_error(
        self,
        mock_env_vars: None,
        requests_mock: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test _get_latest_starters_version when the request raises an exception"""
        with caplog.at_level(logging.ERROR):
            requests_mock.get(
                "https://api.github.com/repos/kedro-org/kedro-starters/releases/latest",
                exc=requests.exceptions.RequestException("Request failed"),
            )
            result: str = _get_latest_starters_version()
            assert result == ""
            assert (
                "Error fetching kedro-starters latest release version: Request failed"
                in caplog.text
            )


class TestStartersVersionComparison:
    def test_kedro_version_equal_or_lower_to_starters_equal(
        self, mocker: pytest.MockFixture
    ) -> None:
        """Test when version is equal to starters_version"""
        mocker.patch(
            "kedro.framework.cli.starters._get_latest_starters_version",
            return_value="1.2.3",
        )
        test_version: str = "1.2.3"
        result: bool = _kedro_version_equal_or_lower_to_starters(test_version)
        assert result is True

    def test_kedro_version_equal_or_lower_to_starters_lower(
        self, mocker: pytest.MockFixture
    ) -> None:
        """Test when version is lower than starters_version"""
        mocker.patch(
            "kedro.framework.cli.starters._get_latest_starters_version",
            return_value="1.3.0",
        )
        test_version: str = "1.2.3"
        result: bool = _kedro_version_equal_or_lower_to_starters(test_version)
        assert result is True

    def test_kedro_version_equal_or_lower_to_starters_higher(
        self, mocker: pytest.MockFixture
    ) -> None:
        """Test when version is higher than starters_version"""
        mocker.patch(
            "kedro.framework.cli.starters._get_latest_starters_version",
            return_value="1.2.3",
        )
        test_version: str = "1.2.4"
        result: bool = _kedro_version_equal_or_lower_to_starters(test_version)
        assert result is False


class TestFetchCookiecutterArgsWhenKedroVersionDifferentFromStarters:
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {"tools": [], "example_pipeline": "False", "output_dir": "/my/output/dir"}

    def test_kedro_version_match_starters_true(
        self,
        config: Dict[str, Any],
        mocker: pytest.MockFixture,
    ) -> None:
        mocker.patch(
            "kedro.framework.cli.starters._kedro_version_equal_or_lower_to_starters",
            return_value=True,
        )
        config["tools"] = ["PySpark", "Kedro Viz"]
        checkout: str = version
        directory: str = "my-directory"
        template_path: str = "my/template/path"
        expected_args: Dict[str, Any] = {
            "output_dir": "/my/output/dir",
            "no_input": True,
            "extra_context": config,
            "checkout": version,
            "directory": "spaceflights-pyspark-viz",
        }
        expected_path: str = "git+https://github.com/kedro-org/kedro-starters.git"
        result_args: Dict[str, Any]
        result_path: str
        (
            result_args,
            result_path,
        ) = _make_cookiecutter_args_and_fetch_template(
            config, checkout, directory, template_path
        )
        assert result_args == expected_args
        assert result_path == expected_path

    def test_kedro_version_match_starters_false(
        self,
        config: Dict[str, Any],
        mocker: pytest.MockFixture,
    ) -> None:
        mocker.patch(
            "kedro.framework.cli.starters._kedro_version_equal_or_lower_to_starters",
            return_value=False,
        )
        config["tools"] = ["PySpark", "Kedro Viz"]
        checkout: str = "my-checkout"
        directory: str = "my-directory"
        template_path: str = "my/template/path"
        expected_args: Dict[str, Any] = {
            "output_dir": "/my/output/dir",
            "no_input": True,
            "extra_context": config,
            "checkout": "my-checkout",
            "directory": "spaceflights-pyspark-viz",
        }
        expected_path: str = "git+https://github.com/kedro-org/kedro-starters.git"
        result_args: Dict[str, Any]
        result_path: str
        (
            result_args,
            result_path,
        ) = _make_cookiecutter_args_and_fetch_template(
            config, checkout, directory, template_path
        )
        assert result_args == expected_args
        assert result_path == expected_path

    def test_no_tools_and_example_pipeline(
        self,
        config: Dict[str, Any],
        mocker: pytest.MockFixture,
    ) -> None:
        mocker.patch(
            "kedro.framework.cli.starters._kedro_version_equal_or_lower_to_starters",
            return_value=False,
        )
        config["tools"] = []
        config["example_pipeline"] = "True"
        checkout: str = "main"
        directory: str = ""
        template_path: str = "my/template/path"
        expected_args: Dict[str, Any] = {
            "output_dir": "/my/output/dir",
            "no_input": True,
            "extra_context": config,
            "checkout": "main",
            "directory": "spaceflights-pandas",
        }
        expected_path: str = "git+https://github.com/kedro-org/kedro-starters.git"
        result_args: Dict[str, Any]
        result_path: str
        (
            result_args,
            result_path,
        ) = _make_cookiecutter_args_and_fetch_template(
            config, checkout, directory, template_path
        )
        assert result_args == expected_args
        assert result_path == expected_path

    def test_no_tools_no_example_pipeline(
        self,
        config: Dict[str, Any],
        mocker: pytest.MockFixture,
    ) -> None:
        mocker.patch(
            "kedro.framework.cli.starters._kedro_version_equal_or_lower_to_starters",
            return_value=False,
        )
        config["tools"] = []
        config["example_pipeline"] = "False"
        checkout: str = "main"
        directory: str = ""
        template_path: str = "my/template/path"
        expected_args: Dict[str, Any] = {
            "output_dir": "/my/output/dir",
            "no_input": True,
            "extra_context": config,
            "checkout": "main",
        }
        expected_path: str = "my/template/path"
        result_args: Dict[str, Any]
        result_path: str
        (
            result_args,
            result_path,
        ) = _make_cookiecutter_args_and_fetch_template(
            config, checkout, directory, template_path
        )
        assert result_args == expected_args
        assert result_path == expected_path


class TestSelectCheckoutBranch:
    def test_select_checkout_branch_with_checkout(
        self,
    ) -> None:
        checkout: Optional[str] = "user_selected_branch"
        result: Optional[str] = _select_checkout_branch_for_cookiecutter(
            checkout=checkout
        )
        assert result == checkout

    def test_select_checkout_branch_kedro_version_equal_or_lower(
        self, mocker: pytest.MockFixture
    ) -> None:
        mocker.patch(
            "kedro.framework.cli.starters._kedro_version_equal_or_lower_to_starters",
            return_value=True,
        )
        checkout: Optional[str] = None
        result: Optional[str] = _select_checkout_branch_for_cookiecutter(
            checkout=checkout
        )
        assert result == version

    def test_select_checkout_branch_main(
        self, mocker: pytest.MockFixture
    ) -> None:
        mocker.patch(
            "kedro.framework.cli.starters._kedro_version_equal_or_lower_to_starters",
            return_value=False,
        )
        checkout: Optional[str] = None
        result: Optional[str] = _select_checkout_branch_for_cookiecutter(
            checkout=checkout
        )
        assert result == "main"
