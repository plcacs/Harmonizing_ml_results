"""This module contains unit test for the cli command 'kedro new'"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
def mock_determine_repo_dir(mocker: Any) -> Any:
    return mocker.patch(
        "cookiecutter.repository.determine_repo_dir",
        return_value=(str(TEMPLATE_PATH), None),
    )


@pytest.fixture
def mock_cookiecutter(mocker: Any) -> Any:
    return mocker.patch("cookiecutter.main.cookiecutter")


@pytest.fixture
def patch_cookiecutter_args(mocker: Any) -> None:
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
    cookiecutter_args["checkout"] = "main"  # Force the checkout to be "main"
    return cookiecutter_args, starter_path


def _clean_up_project(project_dir: Path) -> None:
    if project_dir.is_dir():
        shutil.rmtree(str(project_dir), ignore_errors=True)


def _write_yaml(filepath: Path, config: Dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    yaml_str = yaml.dump(config)
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
    tools: str = "none", repo_name: str = "", python_package: str = ""
) -> str:
    return "\n".join([tools, repo_name, python_package])


def _get_expected_files(tools: str, example_pipeline: str) -> int:
    tools_template_files: Dict[str, int] = {
        "1": 0,  # Linting does not add any files
        "2": 3,  # If Testing is selected, we add 2 init.py files and 1 test_run.py
        "3": 1,  # If Logging is selected, we add logging.py
        "4": 2,  # If Documentation is selected, we add conf.py and index.rst
        "5": 8,  # If Data Structure is selected, we add 8 .gitkeep files
        "6": 2,  # If PySpark is selected, we add spark.yml and hooks.py
        "7": 0,  # Kedro Viz does not add any files
    }  # files added to template by each tool
    tools_list: List[str] = _parse_tools_input(tools)
    example_pipeline_bool: bool = _parse_yes_no_to_bool(example_pipeline)
    expected_files: int = FILES_IN_TEMPLATE_WITH_NO_TOOLS

    for tool in tools_list:
        expected_files = expected_files + tools_template_files[tool]
    # If example pipeline was chosen we don't need to delete /data folder
    if example_pipeline_bool and "5" not in tools_list:
        expected_files += tools_template_files["5"]
    example_files_count: List[int] = [
        3,  # Raw data files
        2,  # Parameters_ .yml files
        6,  # .py files in pipelines folder
    ]
    if example_pipeline_bool:  # If example option is chosen
        expected_files += sum(example_files_count)
        expected_files += (
            4 if "7" in tools_list else 0
        )  # add 3 .py and 1 parameters files in reporting for Viz
        expected_files += (
            1 if "2" in tools_list else 0
        )  # add 1 test file if tests is chosen in tools

    return expected_files


def _assert_requirements_ok(
    result: Any,
    tools: str = "none",
    repo_name: str = "new-kedro-project",
    output_dir: str = ".",
) -> None:
    assert result.exit_code == 0, result.output

    root_path: Path = (Path(output_dir) / repo_name).resolve()

    assert "Congratulations!" in result.output
    assert f"has been created in the directory \n{root_path}" in result.output

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
        pyproject_config = toml.load(pyproject_file_path)
        expected = {
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
        pyproject_config = toml.load(pyproject_file_path)
        expected = {
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
    result: Any,
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
        assert "It has been created with an example pipeline." in result.output
    else:
        assert "It has been created with an example pipeline." not in result.output

    generated_files: List[Path] = [
        p for p in full_path.rglob("*") if p.is_file() and p.name != ".DS_Store"
    ]

    assert len(generated_files) == _get_expected_files(tools, example_pipeline)
    assert full_path.exists()
    assert (full_path / ".gitignore").is_file()
    assert project_name in (full_path / "README.md").read_text(encoding="utf-8")
    assert "KEDRO" in (full_path / ".gitignore").read_text(encoding="utf-8")
    assert kedro_version in (full_path / "requirements.txt").read_text(encoding="utf-8")
    assert (full_path / "src" / python_package / "__init__.py").is_file()


def _assert_name_ok(
    result: Any,
    project_name: str = "New Kedro Project",
) -> None:
    assert result.exit_code == 0, result.output
    assert "Congratulations!" in result.output
    assert (
        f"Your project '{project_name}' has been created in the directory"
        in result.output
    )


def test_starter_list(fake_kedro_cli: Any) -> None:
    """Check that `kedro starter list` prints out all starter aliases."""
    result = CliRunner().invoke(fake_kedro_cli, ["starter", "list"])

    assert result.exit_code == 0, result.output
    for alias in _OFFICIAL_STARTER_SPECS_DICT:
        assert alias in result.output


def test_starter_list_with_starter_plugin(
    fake_kedro_cli: Any, entry_point: Any
) -> None:
    """Check that `kedro starter list` prints out the plugin starters."""
    entry_point.load.return_value = [KedroStarterSpec("valid_starter", "valid_path")]
    entry_point.module = "valid_starter_module"
    result = CliRunner().invoke(fake_kedro_cli, ["starter", "list"])
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
    fake_kedro_cli: Any, entry_point: Any, specs: Any, expected: str
) -> None:
    """Check that `kedro starter list` prints out the plugin starters."""
    entry_point.load.return_value = specs
    entry_point.module = "invalid_starter"
    result = CliRunner().invoke(fake_kedro_cli, ["starter", "list"])
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
    def test_parse_tools_valid(self, input: str, expected: List[str]) -> None:
        result = _parse_tools_input(input)
        assert result == expected

    @pytest.mark.parametrize(
        "input",
        ["5-2", "3-1"],
    )
    def test_parse_tools_invalid_range(self, input: str, capsys: Any) -> None:
        with pytest.raises(SystemExit):
            _parse_tools_input(input)
        message = f"'{input}' is an invalid range for project tools.\nPlease ensure range values go from smaller to larger."
        assert message in capsys.readouterr().err

    @pytest.mark.parametrize(
        "input,right_border",
        [("3-9", "9"), ("3-10000", "10000")],
    )
    def test_parse_tools_range_too_high(
        self, input: str, right_border: str, capsys: Any
    ) -> None:
        with pytest.raises(SystemExit):
            _parse_tools_input(input)
        message = f"'{right_border}' is not a valid selection.\nPlease select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        assert message in capsys.readouterr().err

    @pytest.mark.parametrize(
        "input,last_invalid",
        [("0,3,5", "0"), ("1,3,8", "8"), ("0-4", "0")],
    )
    def test_parse_tools_invalid_selection(
        self, input: str, last_invalid: str, capsys: Any
    ) -> None:
        with pytest.raises(SystemExit):
            selected = _parse_tools_input(input)
            _validate_tool_selection(selected)
        message = f"'{last_invalid}' is not a valid selection.\nPlease select from the available tools: 1, 2, 3, 4, 5, 6, 7."
        assert message in capsys.readouterr().err


@pytest.mark.usefixtures("chdir_to_tmp")
class TestNewFromUserPromptsValid:
    """Tests for running `kedro new` interactively."""

    def test_default(self, fake_kedro_cli: Any) -> None:
        """Test new project creation using default New Kedro Project options."""
        result = CliRunner().invoke(
            fake_kedro_cli, ["new"], input=_make_cli_prompt_input()
        )
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    def test_custom_project_name(self, fake_kedro_cli: Any) -> None:
        result = CliRunner().invoke(
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
        self, fake_kedro_cli: Any
    ) -> None:
        result = CliRunner().invoke(
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

    def test_no_prompts(self, fake_kedro_cli: Any) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        (Path("template") / "prompts.yml").unlink()
        result = CliRunner().invoke(fake_kedro_cli, ["new", "--starter", "template"])
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    def test_empty_prompts(self, fake_kedro_cli: Any) -> None:
        shutil.copytree(TEMPLATE_PATH, "template")
        _write_yaml(Path("template") / "prompts.yml", {})
        result = CliRunner().invoke(fake_kedro_cli, ["new", "--starter", "template"])
        _assert_template_ok(result)
        _clean_up_project(Path("./new-kedro-project"))

    @pytest.mark.parametrize(
        "regex, valid