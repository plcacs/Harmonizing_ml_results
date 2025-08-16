from pathlib import Path
import shutil
import toml
from pre_commit_hooks.requirements_txt_fixer import fix_requirements
from typing import List, Union

current_dir: Path = Path.cwd()
lint_requirements: str = 'ruff~=0.1.8\n'
lint_pyproject_requirements: List[str] = ['tool.ruff', 'tool.ruff.format']
test_requirements: str = 'pytest-cov~=3.0\npytest-mock>=1.7.1, <2.0\npytest~=7.2'
test_pyproject_requirements: List[str] = ['tool.pytest.ini_options', 'tool.coverage.report']
docs_pyproject_requirements: List[str] = ['project.optional-dependencies.docs']
dev_pyproject_requirements: List[str] = ['project.optional-dependencies.dev']
example_pipeline_requirements: str = 'seaborn~=0.12.1\nscikit-learn~=1.0\n'

def _remove_from_file(file_path: Path, content_to_remove: str) -> None:
    ...

def _remove_nested_section(data: dict, nested_key: str) -> None:
    ...

def _remove_from_toml(file_path: Path, sections_to_remove: List[str]) -> None:
    ...

def _remove_dir(path: Path) -> None:
    ...

def _remove_file(path: Path) -> None:
    ...

def _remove_pyspark_viz_starter_files(is_viz: bool, python_package_name: str) -> None:
    ...

def _remove_extras_from_kedro_datasets(file_path: Path) -> None:
    ...

def setup_template_tools(selected_tools_list: str, requirements_file_path: Path, pyproject_file_path: Path, python_package_name: str, example_pipeline: str) -> None:
    ...

def sort_requirements(requirements_file_path: Path) -> None:
    ...
