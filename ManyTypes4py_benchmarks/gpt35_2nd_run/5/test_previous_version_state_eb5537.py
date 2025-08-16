from typing import Dict
from dbt.project import Project

class TestPreviousVersionState:
    CURRENT_EXPECTED_MANIFEST_VERSION: int = 12
    CURRENT_EXPECTED_RUN_RESULTS_VERSION: int = 6

    def models(self) -> Dict[str, str]:
        ...

    def seeds(self) -> Dict[str, str]:
        ...

    def snapshots(self) -> Dict[str, str]:
        ...

    def tests(self) -> Dict[str, str]:
        ...

    def macros(self) -> Dict[str, str]:
        ...

    def analyses(self) -> Dict[str, str]:
        ...

    def test_project(self, project: Project) -> None:
        ...

    def generate_latest_manifest(self, project: Project, current_manifest_version: int) -> None:
        ...

    def generate_latest_run_results(self, project: Project, current_run_results_version: int) -> None:
        ...

    def compare_previous_state(self, project: Project, compare_manifest_version: int, expect_pass: bool, num_results: int) -> None:
        ...

    def compare_previous_results(self, project: Project, compare_run_results_version: int, expect_pass: bool, num_results: int) -> None:
        ...

    def test_compare_state_current(self, project: Project) -> None:
        ...

    def test_backwards_compatible_versions(self, project: Project) -> None:
        ...

    def test_nonbackwards_compatible_versions(self, project: Project) -> None:
        ...

    def test_get_manifest_schema_version(self, project: Project) -> None:
        ...

    def test_compare_results_current(self, project: Project) -> None:
        ...

    def test_backwards_compatible_run_results_versions(self, project: Project) -> None:
        ...
