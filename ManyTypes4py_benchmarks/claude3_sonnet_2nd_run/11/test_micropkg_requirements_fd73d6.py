import pytest
from click.testing import CliRunner
from kedro.framework.cli.micropkg import _get_sdist_name, _safe_parse_requirements
from typing import Set, Any

PIPELINE_NAME: str = 'my_pipeline'
SIMPLE_REQUIREMENTS: str = 'A\nA.B-C_D\naa\nname\nname<=1\nname>=3\nname>=3,<2\nname==1.2.3\nname!=1.2.3 # inline comment\n# whole line comment\nname@http://foo.com\nname [fred,bar] @ http://foo.com ; python_version==\'2.7\'\nname[quux, strange];python_version<\'2.7\' and platform_version==\'2\'\nname; os_name==\'a\' or os_name==\'b\'\nrequests [security,tests] >= 2.8.1, == 2.8.* ; python_version < "2.7"\npip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ees\n'
COMPLEX_REQUIREMENTS: str = '--extra-index-url https://this.wont.work\n-r other_requirements.txt\n./path/to/package.whl\nhttp://some.website.com/package.whl\n'

@pytest.mark.usefixtures('chdir_to_dummy_project', 'cleanup_dist')
class TestMicropkgRequirements:
    """Many of these tests follow the pattern:
    - create a pipeline with some sort of requirements.txt
    - package the pipeline/micro-package
    - delete the pipeline and pull in the packaged one
    - assert the project's modified requirements.txt is as expected
    """

    def call_pipeline_create(self, cli: Any, metadata: Any) -> None:
        result = CliRunner().invoke(cli, ['pipeline', 'create', PIPELINE_NAME], obj=metadata)
        assert result.exit_code == 0

    def call_micropkg_package(self, cli: Any, metadata: Any) -> None:
        result = CliRunner().invoke(cli, ['micropkg', 'package', f'pipelines.{PIPELINE_NAME}'], obj=metadata)
        assert result.exit_code == 0

    def call_pipeline_delete(self, cli: Any, metadata: Any) -> None:
        result = CliRunner().invoke(cli, ['pipeline', 'delete', '-y', PIPELINE_NAME], obj=metadata)
        assert result.exit_code == 0

    def call_micropkg_pull(self, cli: Any, metadata: Any, repo_path: Any) -> None:
        sdist_file = repo_path / 'dist' / _get_sdist_name(name=PIPELINE_NAME, version='0.1')
        assert sdist_file.is_file()
        result = CliRunner().invoke(cli, ['micropkg', 'pull', str(sdist_file)], obj=metadata)
        assert result.exit_code == 0

    def test_existing_complex_project_requirements_txt(self, fake_project_cli: Any, fake_metadata: Any, fake_package_path: Any, fake_repo_path: Any) -> None:
        """Pipeline requirements.txt and project requirements.txt."""
        project_requirements_txt = fake_repo_path / 'requirements.txt'
        with open(project_requirements_txt, 'a', encoding='utf-8') as file:
            file.write(COMPLEX_REQUIREMENTS)
        existing_requirements: Set[str] = _safe_parse_requirements(project_requirements_txt.read_text())
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_requirements_txt = fake_package_path / 'pipelines' / PIPELINE_NAME / 'requirements.txt'
        pipeline_requirements_txt.write_text(SIMPLE_REQUIREMENTS)
        self.call_micropkg_package(fake_project_cli, fake_metadata)
        self.call_pipeline_delete(fake_project_cli, fake_metadata)
        self.call_micropkg_pull(fake_project_cli, fake_metadata, fake_repo_path)
        packaged_requirements: Set[str] = _safe_parse_requirements(SIMPLE_REQUIREMENTS)
        pulled_requirements: Set[str] = _safe_parse_requirements(project_requirements_txt.read_text())
        assert pulled_requirements == existing_requirements | packaged_requirements
        assert COMPLEX_REQUIREMENTS in project_requirements_txt.read_text()

    def test_existing_project_requirements_txt(self, fake_project_cli: Any, fake_metadata: Any, fake_package_path: Any, fake_repo_path: Any) -> None:
        """Pipeline requirements.txt and project requirements.txt."""
        project_requirements_txt = fake_repo_path / 'requirements.txt'
        existing_requirements: Set[str] = _safe_parse_requirements(project_requirements_txt.read_text())
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_requirements_txt = fake_package_path / 'pipelines' / PIPELINE_NAME / 'requirements.txt'
        pipeline_requirements_txt.write_text(SIMPLE_REQUIREMENTS)
        self.call_micropkg_package(fake_project_cli, fake_metadata)
        self.call_pipeline_delete(fake_project_cli, fake_metadata)
        self.call_micropkg_pull(fake_project_cli, fake_metadata, fake_repo_path)
        packaged_requirements: Set[str] = _safe_parse_requirements(SIMPLE_REQUIREMENTS)
        pulled_requirements: Set[str] = _safe_parse_requirements(project_requirements_txt.read_text())
        assert pulled_requirements == existing_requirements | packaged_requirements

    def test_missing_project_requirements_txt(self, fake_project_cli: Any, fake_metadata: Any, fake_package_path: Any, fake_repo_path: Any) -> None:
        """Pipeline requirements.txt without requirements.txt at
        project level."""
        project_requirements_txt = fake_repo_path / 'requirements.txt'
        project_requirements_txt.unlink()
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_requirements_txt = fake_package_path / 'pipelines' / PIPELINE_NAME / 'requirements.txt'
        pipeline_requirements_txt.write_text(SIMPLE_REQUIREMENTS)
        packaged_requirements: Set[str] = _safe_parse_requirements(SIMPLE_REQUIREMENTS)
        self.call_micropkg_package(fake_project_cli, fake_metadata)
        self.call_pipeline_delete(fake_project_cli, fake_metadata)
        self.call_micropkg_pull(fake_project_cli, fake_metadata, fake_repo_path)
        assert project_requirements_txt.exists()
        pulled_requirements: Set[str] = _safe_parse_requirements(project_requirements_txt.read_text())
        assert packaged_requirements == pulled_requirements

    def test_no_requirements(self, fake_project_cli: Any, fake_metadata: Any, fake_repo_path: Any) -> None:
        """No pipeline requirements.txt, and also no requirements.txt
        at project level."""
        project_requirements_txt = fake_repo_path / 'requirements.txt'
        project_requirements_txt.unlink()
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        self.call_micropkg_package(fake_project_cli, fake_metadata)
        self.call_pipeline_delete(fake_project_cli, fake_metadata)
        self.call_micropkg_pull(fake_project_cli, fake_metadata, fake_repo_path)
        assert not project_requirements_txt.exists()

    def test_all_requirements_already_covered(self, fake_project_cli: Any, fake_metadata: Any, fake_repo_path: Any, fake_package_path: Any) -> None:
        """All requirements from pipeline requirements.txt already exist at project
        level requirements.txt."""
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_requirements_txt = fake_package_path / 'pipelines' / PIPELINE_NAME / 'requirements.txt'
        project_requirements_txt = fake_repo_path / 'requirements.txt'
        pipeline_requirements_txt.write_text(SIMPLE_REQUIREMENTS)
        project_requirements_txt.write_text(SIMPLE_REQUIREMENTS)
        self.call_micropkg_package(fake_project_cli, fake_metadata)
        self.call_pipeline_delete(fake_project_cli, fake_metadata)
        self.call_micropkg_pull(fake_project_cli, fake_metadata, fake_repo_path)
        assert project_requirements_txt.read_text() == SIMPLE_REQUIREMENTS

    def test_no_pipeline_requirements_txt(self, fake_project_cli: Any, fake_metadata: Any, fake_repo_path: Any) -> None:
        """No pipeline requirements.txt and no project requirements.txt does not
        create project requirements.txt."""
        project_requirements_txt = fake_repo_path / 'requirements.txt'
        project_requirements_txt.unlink()
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        self.call_micropkg_package(fake_project_cli, fake_metadata)
        self.call_pipeline_delete(fake_project_cli, fake_metadata)
        self.call_micropkg_pull(fake_project_cli, fake_metadata, fake_repo_path)
        assert not project_requirements_txt.exists()

    def test_empty_pipeline_requirements_txt(self, fake_project_cli: Any, fake_metadata: Any, fake_package_path: Any, fake_repo_path: Any) -> None:
        """Empty pipeline requirements.txt and no project requirements.txt does not
        create project requirements.txt."""
        project_requirements_txt = fake_repo_path / 'requirements.txt'
        project_requirements_txt.unlink()
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_requirements_txt = fake_package_path / 'pipelines' / PIPELINE_NAME / 'requirements.txt'
        pipeline_requirements_txt.touch()
        self.call_micropkg_package(fake_project_cli, fake_metadata)
        self.call_pipeline_delete(fake_project_cli, fake_metadata)
        self.call_micropkg_pull(fake_project_cli, fake_metadata, fake_repo_path)
        assert not project_requirements_txt.exists()

    @pytest.mark.parametrize('requirement', COMPLEX_REQUIREMENTS.splitlines())
    def test_complex_requirements(self, requirement: str, fake_project_cli: Any, fake_metadata: Any, fake_package_path: Any) -> None:
        """Options that are valid in requirements.txt but cannot be packaged in
        pyproject.toml."""
        self.call_pipeline_create(fake_project_cli, fake_metadata)
        pipeline_requirements_txt = fake_package_path / 'pipelines' / PIPELINE_NAME / 'requirements.txt'
        pipeline_requirements_txt.write_text(requirement)
        result = CliRunner().invoke(fake_project_cli, ['micropkg', 'package', f'pipelines.{PIPELINE_NAME}'], obj=fake_metadata)
        assert result.exit_code == 1
        assert 'InvalidRequirement: Expected package name at the start of dependency specifier' in result.output or 'InvalidRequirement: Expected end or semicolon' in result.output or 'InvalidRequirement: Parse error' in result.output
