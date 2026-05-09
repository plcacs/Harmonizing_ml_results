import pytest
from chalice import pipeline
from chalice import __version__ as chalice_version
from chalice.pipeline import InvalidCodeBuildPythonVersion, PipelineParameters

@pytest.fixture
def pipeline_gen() -> pipeline.CreatePipelineTemplateLegacy:
    return pipeline.CreatePipelineTemplateLegacy()

@pytest.fixture
def pipeline_params() -> PipelineParameters:
    return pipeline.PipelineParameters('appname', 'python2.7')

class TestPipelineGenLegacy:
    def setup_method(self) -> None:
        self.pipeline_gen = pipeline.CreatePipelineTemplateLegacy()

    # ...

class TestPipelineGenV2:
    def setup_method(self) -> None:
        self.pipeline_gen = pipeline.CreatePipelineTemplateV2()

    # ...

def test_source_repo_resource(pipeline_params: PipelineParameters) -> None:
    # ...

def test_codebuild_resource(pipeline_params: PipelineParameters) -> None:
    # ...

def test_codepipeline_resource(pipeline_params: PipelineParameters) -> None:
    # ...

def test_install_requirements_in_buildspec(pipeline_params: PipelineParameters) -> None:
    # ...

def test_default_version_range_locks_minor_version() -> None:
    # ...

def test_can_validate_python_version() -> None:
    # ...

def test_can_extract_python_version() -> None:
    # ...

def test_can_generate_github_source(pipeline_params: PipelineParameters) -> None:
    # ...

def test_can_create_buildspec_v2(pipeline_params: PipelineParameters) -> str:
    # ...

def test_build_extractor(template: dict) -> str:
    # ...
