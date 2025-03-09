import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pytest
import yaml
from click.testing import CliRunner
from kedro_datasets.pandas import CSVDataset
from pandas import DataFrame

from kedro.framework.cli.pipeline import _sync_dirs
from kedro.framework.project import settings
from kedro.framework.session import KedroSession

PACKAGE_NAME: str = "dummy_package"
PIPELINE_NAME: str = "my_pipeline"


@pytest.fixture(params=["base"])
def make_pipelines(request, fake_repo_path: Path, fake_package_path: Path, mocker) -> None:
    source_path: Path = fake_package_path / "pipelines" / PIPELINE_NAME
    tests_path: Path = fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME
    conf_path: Path = fake_repo_path / settings.CONF_SOURCE / request.param
    old_conf_path: Path = conf_path / "parameters"

    for path in (source_path, tests_path, conf_path, old_conf_path):
        path.mkdir(parents=True, exist_ok=True)

    (tests_path / "test_pipe.py").touch()
    (source_path / "pipe.py").touch()
    (conf_path / f"parameters_{PIPELINE_NAME}.yml").touch()
    (old_conf_path / f"{PIPELINE_NAME}.yml").touch()

    yield
    mocker.stopall()
    shutil.rmtree(str(source_path), ignore_errors=True)
    shutil.rmtree(str(tests_path), ignore_errors=True)
    shutil.rmtree(str(conf_path), ignore_errors=True)


LETTER_ERROR: str = "It must contain only letters, digits, and/or underscores."
FIRST_CHAR_ERROR: str = "It must start with a letter or underscore."
TOO_SHORT_ERROR: str = "It must be at least 2 characters long."


@pytest.mark.usefixtures("chdir_to_dummy_project")
class TestPipelineCreateCommand:
    @pytest.mark.parametrize("env", [None, "local"])
    def test_create_pipeline(
        self, fake_repo_path: Path, fake_project_cli, fake_metadata, env: Optional[str], fake_package_path: Path
    ) -> None:
        pipelines_dir: Path = fake_package_path / "pipelines"
        assert pipelines_dir.is_dir()

        assert not (pipelines_dir / PIPELINE_NAME).exists()

        cmd: List[str] = ["pipeline", "create", PIPELINE_NAME]
        cmd += ["-e", env] if env else []
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)

        assert result.exit_code == 0

        assert f"Creating the pipeline '{PIPELINE_NAME}': OK" in result.output
        assert f"Location: '{pipelines_dir / PIPELINE_NAME}'" in result.output
        assert f"Pipeline '{PIPELINE_NAME}' was successfully created." in result.output

        conf_env: str = env or "base"
        conf_dir: Path = (fake_repo_path / settings.CONF_SOURCE / conf_env).resolve()
        actual_configs: List[Path] = list(conf_dir.glob(f"**/*{PIPELINE_NAME}.yml"))
        expected_configs: List[Path] = [conf_dir / f"parameters_{PIPELINE_NAME}.yml"]
        assert actual_configs == expected_configs

        test_dir: Path = fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME
        expected_files: set = {"__init__.py", "test_pipeline.py"}
        actual_files: set = {f.name for f in test_dir.iterdir()}
        assert actual_files == expected_files

    @pytest.mark.parametrize("env", [None, "local"])
    def test_create_pipeline_template(
        self,
        fake_repo_path: Path,
        fake_project_cli,
        fake_metadata,
        env: Optional[str],
        fake_package_path: Path,
        fake_local_template_dir: Path,
    ) -> None:
        pipelines_dir: Path = fake_package_path / "pipelines"
        assert pipelines_dir.is_dir()

        assert not (pipelines_dir / PIPELINE_NAME).exists()

        cmd: List[str] = ["pipeline", "create", PIPELINE_NAME]
        cmd += ["-e", env] if env else []
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)

        assert (
            f"Using pipeline template at: '{fake_repo_path / 'templates'}"
            in result.output
        )
        assert f"Creating the pipeline '{PIPELINE_NAME}': OK" in result.output
        assert f"Location: '{pipelines_dir / PIPELINE_NAME}'" in result.output
        assert f"Pipeline '{PIPELINE_NAME}' was successfully created." in result.output

        assert (pipelines_dir / PIPELINE_NAME / f"pipeline_{PIPELINE_NAME}.py").exists()

        assert result.exit_code == 0

    @pytest.mark.parametrize("env", [None, "local"])
    def test_create_pipeline_template_command_line_override(
        self,
        fake_repo_path: Path,
        fake_project_cli,
        fake_metadata,
        env: Optional[str],
        fake_package_path: Path,
        fake_local_template_dir: Path,
    ) -> None:
        pipelines_dir: Path = fake_package_path / "pipelines"
        assert pipelines_dir.is_dir()

        assert not (pipelines_dir / PIPELINE_NAME).exists()

        shutil.copytree(
            fake_local_template_dir,
            fake_repo_path / "local_templates",
            dirs_exist_ok=True,
        )

        cmd: List[str] = ["pipeline", "create", PIPELINE_NAME]
        cmd += ["-t", str(fake_repo_path / "local_templates/pipeline")]
        cmd += ["-e", env] if env else []
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)

        assert (
            f"Using pipeline template at: '{fake_repo_path / 'local_templates'}"
            in result.output
        )
        assert f"Creating the pipeline '{PIPELINE_NAME}': OK" in result.output
        assert f"Location: '{pipelines_dir / PIPELINE_NAME}'" in result.output
        assert f"Pipeline '{PIPELINE_NAME}' was successfully created." in result.output

        assert (pipelines_dir / PIPELINE_NAME / f"pipeline_{PIPELINE_NAME}.py").exists()

        assert result.exit_code == 0

    @pytest.mark.parametrize("env", [None, "local"])
    def test_create_pipeline_skip_config(
        self, fake_repo_path: Path, fake_project_cli, fake_metadata, env: Optional[str]
    ) -> None:
        cmd: List[str] = ["pipeline", "create", "--skip-config", PIPELINE_NAME]
        cmd += ["-e", env] if env else []

        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)
        assert result.exit_code == 0
        assert f"Creating the pipeline '{PIPELINE_NAME}': OK" in result.output
        assert f"Pipeline '{PIPELINE_NAME}' was successfully created." in result.output

        conf_dirs: List[Path] = list((fake_repo_path / settings.CONF_SOURCE).rglob(PIPELINE_NAME))
        assert not conf_dirs

        test_dir: Path = fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME
        assert test_dir.is_dir()

    def test_catalog_and_params(
        self, fake_repo_path: Path, fake_project_cli, fake_metadata, fake_package_path: Path
    ) -> None:
        pipelines_dir: Path = fake_package_path / "pipelines"
        assert pipelines_dir.is_dir()

        cmd: List[str] = ["pipeline", "create", PIPELINE_NAME]
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)
        assert result.exit_code == 0

        conf_dir: Path = fake_repo_path / settings.CONF_SOURCE / "base"
        catalog_dict: Dict[str, Dict[str, str]] = {
            "ds_from_pipeline": {
                "type": "pandas.CSVDataset",
                "filepath": "data/01_raw/iris.csv",
            }
        }
        catalog_file: Path = conf_dir / f"catalog_{PIPELINE_NAME}.yml"
        with catalog_file.open("w") as f:
            yaml.dump(catalog_dict, f)

        params_file: Path = conf_dir / f"parameters_{PIPELINE_NAME}.yml"
        assert params_file.is_file()
        params_dict: Dict[str, Dict[str, Union[List[int], None]]] = {"params_from_pipeline": {"p1": [1, 2, 3], "p2": None}}
        with params_file.open("w") as f:
            yaml.dump(params_dict, f)

        with KedroSession.create() as session:
            ctx = session.load_context()
        assert isinstance(ctx.catalog._get_dataset("ds_from_pipeline"), CSVDataset)
        assert isinstance(ctx.catalog.load("ds_from_pipeline"), DataFrame)
        assert ctx.params["params_from_pipeline"] == params_dict["params_from_pipeline"]

    def test_skip_copy(self, fake_repo_path: Path, fake_project_cli, fake_metadata) -> None:
        for dirname in ("catalog", "parameters"):
            path: Path = (
                fake_repo_path
                / settings.CONF_SOURCE
                / "base"
                / f"{dirname}_{PIPELINE_NAME}.yml"
            )
            path.parent.mkdir(exist_ok=True)
            path.touch()

        tests_init: Path = (
            fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME / "__init__.py"
        )
        tests_init.parent.mkdir(parents=True, exist_ok=True)
        tests_init.touch()

        cmd: List[str] = ["pipeline", "create", PIPELINE_NAME]
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)

        assert result.exit_code == 0
        assert "__init__.py': SKIPPED" in result.output
        assert f"parameters_{PIPELINE_NAME}.yml': SKIPPED" in result.output
        assert result.output.count("SKIPPED") == 2

    def test_failed_copy(
        self, fake_project_cli, fake_metadata, fake_package_path: Path, mocker
    ) -> None:
        error: Exception = Exception("Mock exception")
        mocked_copy = mocker.patch("shutil.copyfile", side_effect=error)

        cmd: List[str] = ["pipeline", "create", PIPELINE_NAME]
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)
        mocked_copy.assert_called_once()
        assert result.exit_code
        assert result.output.count("FAILED") == 1
        assert result.exception is error

        pipelines_dir: Path = fake_package_path / "pipelines"
        assert (pipelines_dir / PIPELINE_NAME / "pipeline.py").is_file()

    def test_no_pipeline_arg_error(
        self, fake_project_cli, fake_metadata, fake_package_path: Path
    ) -> None:
        pipelines_dir: Path = fake_package_path / "pipelines"
        assert pipelines_dir.is_dir()

        result = CliRunner().invoke(
            fake_project_cli, ["pipeline", "create"], obj=fake_metadata
        )
        assert result.exit_code
        assert "Missing argument 'NAME'" in result.output

    @pytest.mark.parametrize(
        "bad_name,error_message",
        [
            ("bad name", LETTER_ERROR),
            ("bad%name", LETTER_ERROR),
            ("1bad", FIRST_CHAR_ERROR),
            ("a", TOO_SHORT_ERROR),
        ],
    )
    def test_bad_pipeline_name(
        self, fake_project_cli, fake_metadata, bad_name: str, error_message: str
    ) -> None:
        result = CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", bad_name], obj=fake_metadata
        )
        assert result.exit_code
        assert error_message in result.output

    def test_duplicate_pipeline_name(
        self, fake_project_cli, fake_metadata, fake_package_path: Path
    ) -> None:
        pipelines_dir: Path = fake_package_path / "pipelines"
        assert pipelines_dir.is_dir()

        cmd: List[str] = ["pipeline", "create", PIPELINE_NAME]
        first = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)
        assert first.exit_code == 0

        second = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)
        assert second.exit_code
        assert f"Creating the pipeline '{PIPELINE_NAME}': FAILED" in second.output
        assert "directory already exists" in second.output

    def test_bad_env(self, fake_project_cli, fake_metadata) -> None:
        env: str = "no_such_env"
        cmd: List[str] = ["pipeline", "create", "-e", env, PIPELINE_NAME]
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)
        assert result.exit_code
        assert f"Unable to locate environment '{env}'" in result.output


@pytest.mark.usefixtures("chdir_to_dummy_project", "make_pipelines")
class TestPipelineDeleteCommand:
    @pytest.mark.parametrize(
        "make_pipelines,env,expected_conf",
        [("base", None, "base"), ("local", "local", "local")],
        indirect=["make_pipelines"],
    )
    def test_delete_pipeline(
        self,
        env: Optional[str],
        expected_conf: str,
        fake_repo_path: Path,
        fake_project_cli,
        fake_metadata,
        fake_package_path: Path,
    ) -> None:
        options: List[str] = ["--env", env] if env else []
        result = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "delete", "-y", PIPELINE_NAME, *options],
            obj=fake_metadata,
        )

        source_path: Path = fake_package_path / "pipelines" / PIPELINE_NAME
        tests_path: Path = fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME
        conf_path: Path = fake_repo_path / settings.CONF_SOURCE / expected_conf
        params_path: Path = conf_path / f"parameters_{PIPELINE_NAME}.yml"
        old_params_path: Path = conf_path / "parameters" / f"{PIPELINE_NAME}.yml"

        assert f"Deleting '{source_path}': OK" in result.output
        assert f"Deleting '{tests_path}': OK" in result.output
        assert f"Deleting '{params_path}': OK" in result.output
        assert f"Deleting '{old_params_path}': OK" in result.output

        assert f"Pipeline '{PIPELINE_NAME}' was successfully deleted." in result.output
        assert (
            f"If you added the pipeline '{PIPELINE_NAME}' to 'register_pipelines()' in "
            f"""'{fake_package_path / "pipeline_registry.py"}', you will need to remove it."""
        ) in result.output

        assert not source_path.exists()
        assert not tests_path.exists()
        assert not params_path.exists()
        assert not params_path.exists()

    def test_delete_pipeline_skip(
        self, fake_repo_path: Path, fake_project_cli, fake_metadata, fake_package_path: Path
    ) -> None:
        source_path: Path = fake_package_path / "pipelines" / PIPELINE_NAME

        shutil.rmtree(str(source_path))

        result = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "delete", "-y", PIPELINE_NAME],
            obj=fake_metadata,
        )
        tests_path: Path = fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME
        params_path: Path = (
            fake_repo_path
            / settings.CONF_SOURCE
            / "base"
            / f"parameters_{PIPELINE_NAME}.yml"
        )

        assert f"Deleting '{source_path}'" not in result.output
        assert f"Deleting '{tests_path}': OK" in result.output
        assert f"Deleting '{params_path}': OK" in result.output

        assert f"Pipeline '{PIPELINE_NAME}' was successfully deleted." in result.output
        assert (
            f"If you added the pipeline '{PIPELINE_NAME}' to 'register_pipelines()' in "
            f"""'{fake_package_path / "pipeline_registry.py"}', you will need to remove it."""
        ) in result.output

        assert not source_path.exists()
        assert not tests_path.exists()
        assert not params_path.exists()

    def test_delete_pipeline_fail(
        self, fake_project_cli, fake_metadata, fake_package_path: Path, mocker
    ) -> None:
        source_path: Path = fake_package_path / "pipelines" / PIPELINE_NAME

        mocker.patch(
            "kedro.framework.cli.pipeline.shutil.rmtree",
            side_effect=PermissionError("permission"),
        )
        result = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "delete", "-y", PIPELINE_NAME],
            obj=fake_metadata,
        )

        assert result.exit_code, result.output
        assert f"Deleting '{source_path}': FAILED" in result.output

    @pytest.mark.parametrize(
        "bad_name,error_message",
        [
            ("bad name", LETTER_ERROR),
            ("bad%name", LETTER_ERROR),
            ("1bad", FIRST_CHAR_ERROR),
            ("a", TOO_SHORT_ERROR),
        ],
    )
    def test_bad_pipeline_name(
        self, fake_project_cli, fake_metadata, bad_name: str, error_message: str
    ) -> None:
        result = CliRunner().invoke(
            fake_project_cli, ["pipeline", "delete", "-y", bad_name], obj=fake_metadata
        )
        assert result.exit_code
        assert error_message in result.output

    def test_pipeline_not_found(self, fake_project_cli, fake_metadata) -> None:
        result = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "delete", "-y", "non_existent"],
            obj=fake_metadata,
        )
        assert result.exit_code
        assert "Pipeline 'non_existent' not found." in result.output

    def test_bad_env(self, fake_project_cli, fake_metadata) -> None:
        result = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "delete", "-y", "-e", "invalid_env", PIPELINE_NAME],
            obj=fake_metadata,
        )
        assert result.exit_code
        assert "Unable to locate environment 'invalid_env'" in result.output

    @pytest.mark.parametrize("input_", ["n", "N", "random"])
    def test_pipeline_delete_confirmation(
        self, fake_repo_path: Path, fake_project_cli, fake_metadata, fake_package_path: Path, input_: str
    ) -> None:
        result = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "delete", PIPELINE_NAME],
            input=input_,
            obj=fake_metadata,
        )

        source_path: Path = fake_package_path / "pipelines" / PIPELINE_NAME
        tests_path: Path = fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME
        params_path: Path = (
            fake_repo_path
            / settings.CONF_SOURCE
            / "base"
            / f"parameters_{PIPELINE_NAME}.yml"
        )

        assert "The following paths will be removed:" in result.output
        assert str(source_path) in result.output
        assert str(tests_path) in result.output
        assert str(params_path) in result.output

        assert (
            f"Are you sure you want to delete pipeline '{PIPELINE_NAME}'"
            in result.output
        )
        assert "Deletion aborted!" in result.output

        assert source_path.is_dir()
        assert tests_path.is_dir()
        assert params_path.is_file()

    @pytest.mark.parametrize("input_", ["n", "N", "random"])
    def test_pipeline_delete_confirmation_skip(
        self, fake_repo_path: Path, fake_project_cli, fake_metadata, fake_package_path: Path, input_: str
    ) -> None:
        source_path: Path = fake_package_path / "pipelines" / PIPELINE_NAME
        shutil.rmtree(str(source_path))
        result = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "delete", PIPELINE_NAME],
            input=input_,
            obj=fake_metadata,
        )

        tests_path: Path = fake_repo_path / "tests" / "pipelines" / PIPELINE_NAME
        params_path: Path = (
            fake_repo_path
            / settings.CONF_SOURCE
            / "base"
            / f"parameters_{PIPELINE_NAME}.yml"
        )

        assert "The following paths will be removed:" in result.output
        assert str(source_path) not in result.output
        assert str(tests_path) in result.output
        assert str(params_path) in result.output

        assert (
            f"Are you sure you want to delete pipeline '{PIPELINE_NAME}'"
            in result.output
        )
        assert "Deletion aborted!" in result.output

        assert tests_path.is_dir()
        assert params_path.is_file()


class TestSyncDirs:
    @pytest.fixture(autouse=True)
    def mock_click(self, mocker) -> None:
        mocker.patch("click.secho")

    @pytest.fixture
    def source(self, tmp_path: Path) -> Path:
        source_dir: Path = Path(tmp_path) / "source"
        source_dir.mkdir()
        (source_dir / "existing").mkdir()
        (source_dir / "existing" / "source_file").touch()
        (source_dir / "existing" / "common").write_text("source", encoding="utf-8")
        (source_dir / "new").mkdir()
        (source_dir / "new" / "source_file").touch()
        return source_dir

    def test_sync_target_exists(self, source: Path, tmp_path: Path) -> None:
        target: Path = Path(tmp_path) / "target"
        target.mkdir()
        (target / "existing").mkdir()
        (target / "existing" / "target_file").touch()
        (target / "existing" / "common").write_text("target", encoding="utf-8")

        _sync_dirs(source, target)

        assert (source / "existing" / "source_file").is_file()
        assert (source / "existing" / "common").read_text() == "source"
        assert not (source / "existing" / "target_file").exists()
        assert (source / "new" / "source_file").is_file()

        assert (target / "existing" / "source_file").is_file()
        assert (target / "existing" / "common").read_text(encoding="utf-8") == "target"
        assert (target / "existing" / "target_file").exists()
        assert (target / "new" / "source_file").is_file()

    def test_sync_no_target(self, source: Path, tmp_path: Path) -> None:
        target: Path = Path(tmp_path) / "target"

        _sync_dirs(source, target)

        assert (source / "existing" / "source_file").is_file()
        assert (source / "existing" / "common").read_text() == "source"
        assert not (source / "existing" / "target_file").exists()
        assert (source / "new" / "source_file").is_file()

        assert (target / "existing" / "source_file").is_file()
        assert (target / "existing" / "common").read_text(encoding="utf-8") == "source"
        assert not (target / "existing" / "target_file").exists()
        assert (target / "new" / "source_file").is_file()
