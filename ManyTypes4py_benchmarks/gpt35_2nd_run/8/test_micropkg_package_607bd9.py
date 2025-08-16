from pathlib import Path
from typing import Set, List, Dict, Any
import tarfile
import pytest
import toml
from click.testing import CliRunner

PIPELINE_NAME: str = 'my_pipeline'
LETTER_ERROR: str = 'It must contain only letters, digits, and/or underscores.'
FIRST_CHAR_ERROR: str = 'It must start with a letter or underscore.'
TOO_SHORT_ERROR: str = 'It must be at least 2 characters long.'

class TestMicropkgPackageCommand:

    def assert_sdist_contents_correct(self, sdist_location: Path, package_name: str = PIPELINE_NAME, version: str = '0.1') -> None:
        ...

    def test_package_micropkg(self, fake_repo_path: Path, fake_project_cli: Any, options: List[str], package_name: str, success_message: str, fake_metadata: Any) -> None:
        ...

    def test_micropkg_package_same_name_as_package_name(self, fake_metadata: Any, fake_project_cli: Any, fake_repo_path: Path) -> None:
        ...

    def test_micropkg_package_same_name_as_package_name_alias(self, fake_metadata: Any, fake_project_cli: Any, fake_repo_path: Path) -> None:
        ...

    def test_micropkg_package_to_destination(self, fake_project_cli: Any, existing_dir: bool, tmp_path: Path, fake_metadata: Any) -> None:
        ...

    def test_micropkg_package_overwrites_sdist(self, fake_project_cli: Any, tmp_path: Path, fake_metadata: Any) -> None:
        ...

    def test_package_micropkg_bad_alias(self, fake_project_cli: Any, bad_alias: str, error_message: str) -> None:
        ...

    def test_package_micropkg_invalid_module_path(self, fake_project_cli: Any) -> None:
        ...

    def test_package_micropkg_no_config(self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...

    def test_package_non_existing_micropkg_dir(self, fake_package_path: Path, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...

    def test_package_empty_micropkg_dir(self, fake_project_cli: Any, fake_package_path: Path, fake_metadata: Any) -> None:
        ...

    def test_package_modular_pipeline_with_nested_parameters(self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...

    def test_package_pipeline_with_deep_nested_parameters(self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...

    def test_micropkg_package_default(self, fake_repo_path: Path, fake_package_path: Path, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...

    def test_micropkg_package_nested_module(self, fake_project_cli: Any, fake_metadata: Any, fake_repo_path: Path, fake_package_path: Path) -> None:
        ...

class TestMicropkgPackageFromManifest:

    def test_micropkg_package_all(self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any, tmp_path: Path, mocker: Any) -> None:
        ...

    def test_micropkg_package_all_empty_toml(self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any, mocker: Any) -> None:
        ...

    def test_invalid_toml(self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...

    def test_micropkg_package_no_arg_provided(self, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...
