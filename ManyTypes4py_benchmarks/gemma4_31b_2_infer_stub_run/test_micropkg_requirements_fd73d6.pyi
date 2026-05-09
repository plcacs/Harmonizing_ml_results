import pytest
from pathlib import Path
from typing import Any, Protocol
from click.testing import CliRunner

PIPELINE_NAME: str
SIMPLE_REQUIREMENTS: str
COMPLEX_REQUIREMENTS: str

class CliCommandProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class TestMicropkgRequirements:
    """Many of these tests follow the pattern:
    - create a pipeline with some sort of requirements.txt
    - package the pipeline/micro-package
    - delete the pipeline and pull in the packaged one
    - assert the project's modified requirements.txt is as expected
    """

    def call_pipeline_create(self, cli: CliCommandProtocol, metadata: Any) -> None: ...

    def call_micropkg_package(self, cli: CliCommandProtocol, metadata: Any) -> None: ...

    def call_pipeline_delete(self, cli: CliCommandProtocol, metadata: Any) -> None: ...

    def call_micropkg_pull(self, cli: CliCommandProtocol, metadata: Any, repo_path: Path) -> None: ...

    def test_existing_complex_project_requirements_txt(
        self, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_package_path: Path, 
        fake_repo_path: Path
    ) -> None: ...

    def test_existing_project_requirements_txt(
        self, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_package_path: Path, 
        fake_repo_path: Path
    ) -> None: ...

    def test_missing_project_requirements_txt(
        self, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_package_path: Path, 
        fake_repo_path: Path
    ) -> None: ...

    def test_no_requirements(
        self, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_repo_path: Path
    ) -> None: ...

    def test_all_requirements_already_covered(
        self, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_repo_path: Path, 
        fake_package_path: Path
    ) -> None: ...

    def test_no_pipeline_requirements_txt(
        self, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_repo_path: Path
    ) -> None: ...

    def test_empty_pipeline_requirements_txt(
        self, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_package_path: Path, 
        fake_repo_path: Path
    ) -> None: ...

    @pytest.mark.parametrize('requirement', COMPLEX_REQUIREMENTS.splitlines())
    def test_complex_requirements(
        self, 
        requirement: str, 
        fake_project_cli: CliCommandProtocol, 
        fake_metadata: Any, 
        fake_package_path: Path
    ) -> None: ...