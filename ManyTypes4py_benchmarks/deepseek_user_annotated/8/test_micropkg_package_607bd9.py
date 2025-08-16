import tarfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest
import toml
from click.testing import CliRunner
from kedro.framework.cli.micropkg import _get_sdist_name

PIPELINE_NAME: str = "my_pipeline"

LETTER_ERROR: str = "It must contain only letters, digits, and/or underscores."
FIRST_CHAR_ERROR: str = "It must start with a letter or underscore."
TOO_SHORT_ERROR: str = "It must be at least 2 characters long."


@pytest.mark.usefixtures("chdir_to_dummy_project", "cleanup_dist")
class TestMicropkgPackageCommand:
    def assert_sdist_contents_correct(
        self,
        sdist_location: Path,
        package_name: str = PIPELINE_NAME,
        version: str = "0.1",
    ) -> None:
        sdist_name: str = _get_sdist_name(name=package_name, version=version)
        sdist_file: Path = sdist_location / sdist_name
        assert sdist_file.is_file()
        assert len(list(sdist_location.iterdir())) == 1

        with tarfile.open(sdist_file, "r") as tar:
            sdist_contents: Set[str] = set(tar.getnames())

        expected_files: Set[str] = {
            f"{package_name}-{version}/{package_name}/__init__.py",
            f"{package_name}-{version}/{package_name}/nodes.py",
            f"{package_name}-{version}/{package_name}/pipeline.py",
            f"{package_name}-{version}/{package_name}/config/parameters_{package_name}.yml",
            f"{package_name}-{version}/tests/__init__.py",
            f"{package_name}-{version}/tests/test_pipeline.py",
        }
        assert expected_files <= sdist_contents

    @pytest.mark.parametrize(
        "options,package_name,success_message",
        [
            ([], PIPELINE_NAME, f"'dummy_package.pipelines.{PIPELINE_NAME}' packaged!"),
            (
                ["--alias", "alternative"],
                "alternative",
                f"'dummy_package.pipelines.{PIPELINE_NAME}' packaged as 'alternative'!",
            ),
        ],
    )
    def test_package_micropkg(
        self,
        fake_repo_path: Path,
        fake_project_cli: Any,
        options: List[str],
        package_name: str,
        success_message: str,
        fake_metadata: Any,
    ) -> None:
        result: Any = CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", PIPELINE_NAME], obj=fake_metadata
        )
        assert result.exit_code == 0
        result = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", f"pipelines.{PIPELINE_NAME}", *options],
            obj=fake_metadata,
        )

        assert result.exit_code == 0
        assert success_message in result.output

        sdist_location: Path = fake_repo_path / "dist"
        assert f"Location: {sdist_location}" in result.output

        self.assert_sdist_contents_correct(
            sdist_location=sdist_location, package_name=package_name, version="0.1"
        )

    def test_micropkg_package_same_name_as_package_name(
        self, fake_metadata: Any, fake_project_cli: Any, fake_repo_path: Path
    ) -> None:
        """Create modular pipeline with the same name as the
        package name, then package as is. The command should run
        and the resulting sdist should have all expected contents.
        """
        pipeline_name: str = fake_metadata.package_name
        result: Any = CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", pipeline_name], obj=fake_metadata
        )
        assert result.exit_code == 0

        result = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", f"pipelines.{pipeline_name}"],
            obj=fake_metadata,
        )
        sdist_location: Path = fake_repo_path / "dist"

        assert result.exit_code == 0
        assert f"Location: {sdist_location}" in result.output
        self.assert_sdist_contents_correct(
            sdist_location=sdist_location, package_name=pipeline_name
        )

    def test_micropkg_package_same_name_as_package_name_alias(
        self, fake_metadata: Any, fake_project_cli: Any, fake_repo_path: Path
    ) -> None:
        """Create modular pipeline, then package under alias
        the same name as the package name. The command should run
        and the resulting sdist should have all expected contents.
        """
        alias: str = fake_metadata.package_name
        result: Any = CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", PIPELINE_NAME], obj=fake_metadata
        )
        assert result.exit_code == 0

        result = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", f"pipelines.{PIPELINE_NAME}", "--alias", alias],
            obj=fake_metadata,
        )
        sdist_location: Path = fake_repo_path / "dist"

        assert result.exit_code == 0
        assert f"Location: {sdist_location}" in result.output
        self.assert_sdist_contents_correct(
            sdist_location=sdist_location, package_name=alias
        )

    @pytest.mark.parametrize("existing_dir", [True, False])
    def test_micropkg_package_to_destination(
        self,
        fake_project_cli: Any,
        existing_dir: bool,
        tmp_path: Path,
        fake_metadata: Any,
    ) -> None:
        destination: Path = (tmp_path / "in" / "here").resolve()
        if existing_dir:
            destination.mkdir(parents=True)

        result: Any = CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", PIPELINE_NAME], obj=fake_metadata
        )
        assert result.exit_code == 0
        result = CliRunner().invoke(
            fake_project_cli,
            [
                "micropkg",
                "package",
                f"pipelines.{PIPELINE_NAME}",
                "--destination",
                str(destination),
            ],
            obj=fake_metadata,
        )

        assert result.exit_code == 0
        success_message: str = (
            f"'dummy_package.pipelines.{PIPELINE_NAME}' packaged! "
            f"Location: {destination}"
        )
        assert success_message in result.output

        self.assert_sdist_contents_correct(sdist_location=destination)

    def test_micropkg_package_overwrites_sdist(
        self, fake_project_cli: Any, tmp_path: Path, fake_metadata: Any
    ) -> None:
        destination: Path = (tmp_path / "in" / "here").resolve()
        destination.mkdir(parents=True)
        sdist_file: Path = destination / _get_sdist_name(
            name=PIPELINE_NAME, version="0.1"
        )
        sdist_file.touch()

        result: Any = CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", PIPELINE_NAME], obj=fake_metadata
        )
        assert result.exit_code == 0
        result = CliRunner().invoke(
            fake_project_cli,
            [
                "micropkg",
                "package",
                f"pipelines.{PIPELINE_NAME}",
                "--destination",
                str(destination),
            ],
            obj=fake_metadata,
        )
        assert result.exit_code == 0

        warning_message: str = f"Package file {sdist_file} will be overwritten!"
        success_message: str = (
            f"'dummy_package.pipelines.{PIPELINE_NAME}' packaged! "
            f"Location: {destination}"
        )
        assert warning_message in result.output
        assert success_message in result.output

        self.assert_sdist_contents_correct(sdist_location=destination)

    @pytest.mark.parametrize(
        "bad_alias,error_message",
        [
            ("bad name", LETTER_ERROR),
            ("bad%name", LETTER_ERROR),
            ("1bad", FIRST_CHAR_ERROR),
            ("a", TOO_SHORT_ERROR),
        ],
    )
    def test_package_micropkg_bad_alias(
        self, fake_project_cli: Any, bad_alias: str, error_message: str
    ) -> None:
        result: Any = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", f"pipelines.{PIPELINE_NAME}", "--alias", bad_alias],
        )
        assert result.exit_code
        assert error_message in result.output

    def test_package_micropkg_invalid_module_path(self, fake_project_cli: Any) -> None:
        result: Any = CliRunner().invoke(
            fake_project_cli, ["micropkg", "package", f"pipelines/{PIPELINE_NAME}"]
        )
        error_message: str = (
            "The micro-package location you provided is not a valid Python module path"
        )

        assert result.exit_code
        assert error_message in result.output

    def test_package_micropkg_no_config(
        self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any
    ) -> None:
        version: str = "0.1"
        result: Any = CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "create", PIPELINE_NAME, "--skip-config"],
            obj=fake_metadata,
        )
        assert result.exit_code == 0
        result = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", f"pipelines.{PIPELINE_NAME}"],
            obj=fake_metadata,
        )

        assert result.exit_code == 0
        assert f"'dummy_package.pipelines.{PIPELINE_NAME}' packaged!" in result.output

        sdist_location: Path = fake_repo_path / "dist"
        assert f"Location: {sdist_location}" in result.output

        # the sdist contents are slightly different (config shouldn't be included),
        # which is why we can't call self.assert_sdist_contents_correct here
        sdist_file: Path = sdist_location / _get_sdist_name(
            name=PIPELINE_NAME, version=version
        )
        assert sdist_file.is_file()
        assert len(list((fake_repo_path / "dist").iterdir())) == 1

        with tarfile.open(sdist_file, "r") as tar:
            sdist_contents: Set[str] = set(tar.getnames())

        expected_files: Set[str] = {
            f"{PIPELINE_NAME}-{version}/{PIPELINE_NAME}/__init__.py",
            f"{PIPELINE_NAME}-{version}/{PIPELINE_NAME}/nodes.py",
            f"{PIPELINE_NAME}-{version}/{PIPELINE_NAME}/pipeline.py",
            f"{PIPELINE_NAME}-{version}/tests/__init__.py",
            f"{PIPELINE_NAME}-{version}/tests/test_pipeline.py",
        }
        assert expected_files <= sdist_contents
        assert f"{PIPELINE_NAME}/config/parameters.yml" not in sdist_contents

    def test_package_non_existing_micropkg_dir(
        self, fake_package_path: Path, fake_project_cli: Any, fake_metadata: Any
    ) -> None:
        result: Any = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", "pipelines.non_existing"],
            obj=fake_metadata,
        )
        assert result.exit_code == 1
        pipeline_dir: Path = fake_package_path / "pipelines" / "non_existing"
        error_message: str = f"Error: Directory '{pipeline_dir}' doesn't exist."
        assert error_message in result.output

    def test_package_empty_micropkg_dir(
        self, fake_project_cli: Any, fake_package_path: Path, fake_metadata: Any
    ) -> None:
        pipeline_dir: Path = fake_package_path / "pipelines" / "empty_dir"
        pipeline_dir.mkdir()

        result: Any = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", "pipelines.empty_dir"],
            obj=fake_metadata,
        )
        assert result.exit_code == 1
        error_message: str = f"Error: '{pipeline_dir}' is an empty directory."
        assert error_message in result.output

    def test_package_modular_pipeline_with_nested_parameters(
        self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any
    ) -> None:
        """
        The setup for the test is as follows:

        Create two modular pipelines, to verify that only the parameter file with matching pipeline
        name will be packaged.

        Add a directory with a parameter file to verify that if a project has parameters structured
        like below, that the ones inside a directory with the pipeline name are packaged as well
        when calling `kedro micropkg package` for a specific pipeline.

        parameters
            └── retail
                └── params1.ym
        """
        CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", "retail"], obj=fake_metadata
        )
        CliRunner().invoke(
            fake_project_cli,
            ["pipeline", "create", "retail_banking"],
            obj=fake_metadata,
        )
        nested_param_path: Path = Path(
            fake_repo_path / "conf" / "base" / "parameters" / "retail"
        )
        nested_param_path.mkdir(parents=True, exist_ok=True)
        (nested_param_path / "params1.yml").touch()

        result: Any = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", "pipelines.retail"],
            obj=fake_metadata,
        )

        assert result.exit_code == 0
        assert "'dummy_package.pipelines.retail' packaged!" in result.output

        sdist_location: Path = fake_repo_path / "dist"
        assert f"Location: {sdist_location}" in result.output

        sdist_name: str = _get_sdist_name(name="retail", version="0.1")
        sdist_file: Path = sdist_location / sdist_name
        assert sdist_file.is_file()
        assert len(list(sdist_location.iterdir())) == 1

        with tarfile.open(sdist_file, "r") as tar:
            sdist_contents: Set[str] = set(tar.getnames())
        assert (
            "retail-0.1/retail/config/parameters/retail/params1.yml" in sdist_contents
        )
        assert "retail-0.1/retail/config/parameters_retail.yml" in sdist_contents
        assert (
            "retail-0.1/retail/config/parameters_retail_banking.yml"
            not in sdist_contents
        )

    def test_package_pipeline_with_deep_nested_parameters(
        self, fake_repo_path: Path, fake_project_cli: Any, fake_metadata: Any
    ) -> None:
        CliRunner().invoke(
            fake_project_cli, ["pipeline", "create", "retail"], obj=fake_metadata
        )
        deep_nested_param_path: Path = Path(
            fake_repo_path / "conf" / "base" / "parameters" / "deep" / "retail"
        )
        deep_nested_param_path.mkdir(parents=True, exist_ok=True)
        (deep_nested_param_path / "params1.yml").touch()

        deep_nested_param_path2: Path = Path(
            fake_repo_path / "conf" / "base" / "parameters" / "retail" / "deep"
        )
        deep_nested_param_path2.mkdir(parents=True, exist_ok=True)
        (deep_nested_param_path2 / "params1.yml").touch()

        deep_nested_param_path3: Path = Path(
            fake_repo_path / "conf" / "base" / "parameters" / "deep"
        )
        deep_nested_param_path3.mkdir(parents=True, exist_ok=True)
        (deep_nested_param_path3 / "retail.yml").touch()

        super_deep_nested_param_path: Path = Path(
            fake_repo_path
            / "conf"
            / "base"
            / "parameters"
            / "a"
            / "b"
            / "c"
            / "d"
            / "retail"
        )
        super_deep_nested_param_path.mkdir(parents=True, exist_ok=True)
        (super_deep_nested_param_path / "params3.yml").touch()
        result: Any = CliRunner().invoke(
            fake_project_cli,
            ["micropkg", "package", "pipelines.retail"],
            obj=fake_metadata,
        )

        assert result.exit_code == 0
        assert "'dummy_package.pipelines.retail' packaged!" in result.output

        sdist_location: Path = fake_repo_path / "dist"
        assert f"Location: {sdist_location}" in result.output

        sdist_name: str = _get_sdist_name(name="retail", version="0.1")
        sdist_file: Path = sdist_location / sdist_name
        assert sdist_file.is_file()
        assert len(list(sdist_location.iterdir())) == 1

        with tarfile.open(sdist_file, "r") as tar:
            sdist_contents: Set[str] = set(tar.getnames())
        assert (
            "retail-0.1/retail/config/parameters/deep/retail/params1.yml"
            in sdist_contents
        )
        assert (
            "retail-0.1/retail/config/parameters/retail/deep/params1.yml"
            in sdist_contents
        )
        assert "