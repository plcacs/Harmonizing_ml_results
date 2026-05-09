import pytest
from click.testing import CliRunner
from click import CommandGroup
from pathlib import Path
from kedro.framework.cli.micropkg import _get_sdist_name, _safe_parse_requirements

PIPELINE_NAME: str = ...
SIMPLE_REQUIREMENTS: str = ...
COMPLEX_REQUIREMENTS: str = ...

class TestMicropkgRequirements:
    def call_pipeline_create(self, cli: CommandGroup, metadata: dict) -> None:
        ...
    
    def call_micropkg_package(self, cli: CommandGroup, metadata: dict) -> None:
        ...
    
    def call_pipeline_delete(self, cli: CommandGroup, metadata: dict) -> None:
        ...
    
    def call_micropkg_pull(self, cli: CommandGroup, metadata: dict, repo_path: Path) -> None:
        ...
    
    def test_existing_complex_project_requirements_txt(
        self, fake_project_cli: CommandGroup, fake_metadata: dict, fake_package_path: Path, fake_repo_path: Path
    ) -> None:
        ...
    
    def test_existing_project_requirements_txt(
        self, fake_project_cli: CommandGroup, fake_metadata: dict, fake_package_path: Path, fake_repo_path: Path
    ) -> None:
        ...
    
    def test_missing_project_requirements_txt(
        self, fake_project_cli: CommandGroup, fake_metadata: dict, fake_package_path: Path, fake_repo_path: Path
    ) -> None:
        ...
    
    def test_no_requirements(self, fake_project_cli: CommandGroup, fake_metadata: dict, fake_repo_path: Path) -> None:
        ...
    
    def test_all_requirements_already_covered(
        self, fake_project_cli: CommandGroup, fake_metadata: dict, fake_repo_path: Path, fake_package_path: Path
    ) -> None:
        ...
    
    def test_no_pipeline_requirements_txt(
        self, fake_project_cli: CommandGroup, fake_metadata: dict, fake_repo_path: Path
    ) -> None:
        ...
    
    def test_empty_pipeline_requirements_txt(
        self, fake_project_cli: CommandGroup, fake_metadata: dict, fake_package_path: Path, fake_repo_path: Path
    ) -> None:
        ...
    
    @pytest.mark.parametrize('requirement', COMPLEX_REQUIREMENTS.splitlines())
    def test_complex_requirements(
        self, requirement: str, fake_project_cli: CommandGroup, fake_metadata: dict, fake_package_path: Path
    ) -> None:
        ...