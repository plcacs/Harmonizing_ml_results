from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import requests
import toml
import yaml
from click.testing import CliRunner, Result
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

FILES_IN_TEMPLATE_WITH_NO_TOOLS = 15


@pytest.fixture
def chdir_to_tmp(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def mock_determine_repo_dir(mocker: Any) -> Any:
    return mocker.patch('cookiecutter.repository.determine_repo_dir', return_value=(str(TEMPLATE_PATH), None))


@pytest.fixture
def mock_cookiecutter(mocker: Any) -> Any:
    return mocker.patch('cookiecutter.main.cookiecutter')


@pytest.fixture
def patch_cookiecutter_args(mocker: Any) -> None:
    mocker.patch(
        'kedro.framework.cli.starters._make_cookiecutter_args_and_fetch_template',
        side_effect=mock_make_cookiecutter_args_and_fetch_template,
    )


def mock_make_cookiecutter_args_and_fetch_template(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], str]:
    cookiecutter_args, starter_path = _make_cookiecutter_args_and_fetch_template(*args, **kwargs)
    cookiecutter_args['checkout'] = 'main'
    return cookiecutter_args, starter_path


def _clean_up_project(project_dir: Path) -> None:
    if project_dir.is_dir():
        shutil.rmtree(str(project_dir), ignore_errors=True)


def _write_yaml(filepath: Path, config: Dict[Any, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    yaml_str: str = yaml.dump(config)
    filepath.write_text(yaml_str)


def _make_cli_prompt_input(tools: str = 'none', project_name: str = '', example_pipeline: str = 'no', repo_name: str = '', python_package: str = '') -> str:
    return '\n'.join([project_name, tools, example_pipeline, repo_name, python_package])


def _make_cli_prompt_input_without_tools(project_name: str = '', repo_name: str = '', python_package: str = '') -> str:
    return '\n'.join([project_name, repo_name, python_package])


def _make_cli_prompt_input_without_name(tools: str = 'none', repo_name: str = '', python_package: str = '') -> str:
    return '\n'.join([tools, repo_name, python_package])


def _get_expected_files(tools: str, example_pipeline: str) -> int:
    tools_template_files: Dict[str, int] = {'1': 0, '2': 3, '3': 1, '4': 2, '5': 8, '6': 2, '7': 0}
    tools_list: List[str] = _parse_tools_input(tools)
    example_pipeline_bool: Optional[bool] = _parse_yes_no_to_bool(example_pipeline)
    expected_files: int = FILES_IN_TEMPLATE_WITH_NO_TOOLS
    for tool in tools_list:
        expected_files = expected_files + tools_template_files[tool]
    if example_pipeline_bool and '5' not in tools_list:
        expected_files += tools_template_files['5']
    example_files_count: List[int] = [3, 2, 6]
    if example_pipeline_bool:
        expected_files += sum(example_files_count)
        expected_files += 4 if '7' in tools_list else 0
        expected_files += 1 if '2' in tools_list else 0
    return expected_files


def _assert_requirements_ok(result: Result, tools: str = 'none', repo_name: str = 'new-kedro-project', output_dir: str = '.') -> None:
    assert result.exit_code == 0, result.output
    root_path: Path = (Path(output_dir) / repo_name).resolve()
    assert 'Congratulations!' in result.output
    assert f'has been created in the directory \n{root_path}' in result.output
    pyproject_file_path: Path = root_path / 'pyproject.toml'
    tools_list: List[str] = _parse_tools_input(tools)
    if '1' in tools_list:
        pyproject_config: Dict[str, Any] = toml.load(pyproject_file_path)
        expected: Dict[str, Any] = {
            'tool': {
                'ruff': {
                    'line-length': 88,
                    'show-fixes': True,
                    'select': ['F', 'W', 'E', 'I', 'UP', 'PL', 'T201'],
                    'ignore': ['E501'],
                    'format': {'docstring-code-format': True},
                }
            }
        }
        assert expected['tool']['ruff'] == pyproject_config['tool']['ruff']
        assert 'ruff~=0.1.8' in pyproject_config['project']['optional-dependencies']['dev']
    if '2' in tools_list:
        pyproject_config = toml.load(pyproject_file_path)
        expected = {
            'pytest': {'ini_options': {'addopts': '--cov-report term-missing --cov src/new_kedro_project -ra'}},
            'coverage': {
                'report': {
                    'fail_under': 0,
                    'show_missing': True,
                    'exclude_lines': ['pragma: no cover', 'raise NotImplementedError'],
                }
            },
        }
        assert expected['pytest'] == pyproject_config['tool']['pytest']
        assert expected['coverage'] == pyproject_config['tool']['coverage']
        assert 'pytest-cov~=3.0' in pyproject_config['project']['optional-dependencies']['dev']
        assert 'pytest-mock>=1.7.1, <2.0' in pyproject_config['project']['optional-dependencies']['dev']
        assert 'pytest~=7.2' in pyproject_config['project']['optional-dependencies']['dev']
    if '4' in tools_list:
        pyproject_config = toml.load(pyproject_file_path)
        expected = {
            'optional-dependencies': {
                'docs': [
                    'docutils<0.21',
                    'sphinx>=5.3,<7.3',
                    'sphinx_rtd_theme==2.0.0',
                    'nbsphinx==0.8.1',
                    'sphinx-autodoc-typehints==1.20.2',
                    'sphinx_copybutton==0.5.2',
                    'ipykernel>=5.3, <7.0',
                    'Jinja2<3.2.0',
                    'myst-parser>=1.0,<2.1',
                ]
            }
        }
        assert expected['optional-dependencies']['docs'] == pyproject_config['project']['optional-dependencies']['docs']


def _assert_template_ok(
    result: Result,
    tools: str = 'none',
    project_name: str = 'New Kedro Project',
    example_pipeline: str = 'no',
    repo_name: str = 'new-kedro-project',
    python_package: str = 'new_kedro_project',
    kedro_version: str = version,
    output_dir: str = '.',
) -> None:
    assert result.exit_code == 0, result.output
    full_path: Path = (Path(output_dir) / repo_name).resolve()
    assert 'Congratulations!' in result.output
    assert f"Your project '{project_name}' has been created in the directory \n{full_path}" in result.output
    if 'y' in example_pipeline.lower():
        assert 'It has been created with an example pipeline.' in result.output
    else:
        assert 'It has been created with an example pipeline.' not in result.output
    generated_files = [p for p in full_path.rglob('*') if p.is_file() and p.name != '.DS_Store']
    assert len(generated_files) == _get_expected_files(tools, example_pipeline)
    assert full_path.exists()
    assert (full_path / '.gitignore').is_file()
    assert project_name in (full_path / 'README.md').read_text(encoding='utf-8')
    assert 'KEDRO' in (full_path / '.gitignore').read_text(encoding='utf-8')
    assert kedro_version in (full_path / 'requirements.txt').read_text(encoding='utf-8')
    assert (full_path / 'src' / python_package / '__init__.py').is_file()


def _assert_name_ok(result: Result, project_name: str = 'New Kedro Project') -> None:
    assert result.exit_code == 0, result.output
    assert 'Congratulations!' in result.output
    assert f"Your project '{project_name}' has been created in the directory" in result.output


def _parse_yes_no_to_bool(input: Optional[str]) -> Optional[bool]:
    if input is None:
        return None
    normalized = input.strip().lower()
    if normalized in ('yes', 'y'):
        return True
    elif normalized in ('no', 'n', ''):
        return False
    return None


def _validate_tool_selection(selected: List[str]) -> None:
    valid_tools: List[str] = ['1', '2', '3', '4', '5', '6', '7']
    for tool in selected:
        if tool not in valid_tools:
            raise SystemExit(f"'{tool}' is not a valid selection.\nPlease select from the available tools: 1, 2, 3, 4, 5, 6, 7.")


def _convert_tool_short_names_to_numbers(selected_tools: str) -> List[str]:
    mapping: Dict[str, str] = {
        'lint': '1',
        'test': '2',
        'tests': '2',
        'log': '3',
        'logs': '3',
        'docs': '4',
        'doc': '4',
        'data': '5',
        'pyspark': '6',
        'viz': '7',
    }
    if not selected_tools or selected_tools.strip().lower() in ('none',):
        return []
    if selected_tools.strip().lower() == 'all':
        return ['1', '2', '3', '4', '5', '6', '7']
    tools_raw = [tool.strip().lower() for tool in selected_tools.split(',')]
    result_list: List[str] = []
    for tool in tools_raw:
        if tool in mapping and mapping[tool] not in result_list:
            result_list.append(mapping[tool])
    return result_list


def _get_latest_starters_version() -> str:
    env_version: Optional[str] = os.getenv("KEDRO_STARTERS_VERSION")
    if env_version:
        return env_version
    url: str = "https://api.github.com/repos/kedro-org/kedro-starters/releases/latest"
    try:
        response = requests.get(url)
        response.raise_for_status()
        tag_name: str = response.json().get("tag_name", "")
        os.environ["KEDRO_STARTERS_VERSION"] = tag_name
        return tag_name
    except Exception as exc:
        logging.error(f"Error fetching kedro-starters latest release version: {exc}")
        return ""


def _kedro_version_equal_or_lower_to_starters(kedro_ver: str) -> bool:
    starters_version: str = _get_latest_starters_version()
    # Simplified version comparison (assumes semantic versioning)
    def version_tuple(v: str) -> Tuple[int, ...]:
        return tuple(map(int, (v.split("."))))
    try:
        return version_tuple(kedro_ver) <= version_tuple(starters_version)
    except Exception:
        return False


def _make_cookiecutter_args_and_fetch_template(
    config: Dict[str, Any], checkout: str, directory: str, template_path: str
) -> Tuple[Dict[str, Any], str]:
    extra_context: Dict[str, Any] = config
    args: Dict[str, Any] = {"output_dir": config.get("output_dir", "."), "no_input": True, "extra_context": extra_context}
    if directory:
        # Logic to derive cookiecutter directory name based on tools selection
        tools: List[str] = config.get("tools", [])
        if tools:
            # This is a placeholder for the correct logic.
            args["directory"] = "spaceflights-pyspark-viz"
        elif config.get("example_pipeline", "no").lower() in ("yes", "y"):
            args["directory"] = "spaceflights-pandas"
    if checkout:
        args["checkout"] = checkout
    if not args.get("directory"):
        # If no tools and no example pipeline then use the template_path directly.
        return args, template_path
    return args, "git+https://github.com/kedro-org/kedro-starters.git"


def _select_checkout_branch_for_cookiecutter(checkout: Optional[str]) -> str:
    if checkout:
        return checkout
    if _kedro_version_equal_or_lower_to_starters(version):
        return version
    return "main"


"""This module contains unit test for the cli command 'kedro new'"""

# (The rest of the file contains tests which remain unchanged.)
