import shutil
from pathlib import Path
import pytest
import yaml
from click.testing import CliRunner
from kedro_datasets.pandas import CSVDataset
from pandas import DataFrame
from kedro.framework.cli.pipeline import _sync_dirs
from kedro.framework.project import settings
from kedro.framework.session import KedroSession

PACKAGE_NAME: str = 'dummy_package'
PIPELINE_NAME: str = 'my_pipeline'

@pytest.fixture(params=['base'])
def make_pipelines(request, fake_repo_path, fake_package_path, mocker) -> None:
    # ...

@pytest.mark.usefixtures('chdir_to_dummy_project', 'make_pipelines')
class TestPipelineCreateCommand:
    # ...

    @pytest.mark.parametrize('env', [None, 'local'])
    def test_create_pipeline(self, fake_repo_path: Path, fake_project_cli: CliRunner, fake_metadata: object, env: str | None) -> None:
        # ...

    @pytest.mark.parametrize('bad_name', [('bad name', LETTER_ERROR), ('bad%name', LETTER_ERROR), ('1bad', FIRST_CHAR_ERROR), ('a', TOO_SHORT_ERROR)])
    def test_bad_pipeline_name(self, fake_project_cli: CliRunner, fake_metadata: object, bad_name: tuple[str, str]) -> None:
        # ...

    @pytest.mark.parametrize('input_', ['n', 'N', 'random'])
    def test_pipeline_delete_confirmation(self, fake_repo_path: Path, fake_project_cli: CliRunner, fake_metadata: object, fake_package_path: Path, input_: str) -> None:
        # ...

    @pytest.mark.parametrize('input_', ['n', 'N', 'random'])
    def test_pipeline_delete_confirmation_skip(self, fake_repo_path: Path, fake_project_cli: CliRunner, fake_metadata: object, fake_package_path: Path, input_: str) -> None:
        # ...

class TestSyncDirs:
    @pytest.fixture(autouse=True)
    def mock_click(self, mocker: pytest.Mocker) -> None:
        # ...

    @pytest.fixture
    def source(self, tmp_path: Path) -> Path:
        # ...

    def test_sync_target_exists(self, source: Path, tmp_path: Path) -> None:
        # ...

    def test_sync_no_target(self, source: Path, tmp_path: Path) -> None:
        # ...
