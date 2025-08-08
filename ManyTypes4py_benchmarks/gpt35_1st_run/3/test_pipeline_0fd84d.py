from chalice.pipeline import PipelineParameters, InvalidCodeBuildPythonVersion
from typing import Dict, Any

def test_source_repo_resource(pipeline_params: PipelineParameters) -> None:
def test_codebuild_resource(pipeline_params: PipelineParameters) -> None:
def test_codepipeline_resource(pipeline_params: PipelineParameters) -> None:
def test_install_requirements_in_buildspec(pipeline_params: PipelineParameters) -> None:
def test_default_version_range_locks_minor_version() -> None:
def test_can_validate_python_version() -> None:
def test_can_extract_python_version() -> None:
def test_can_generate_github_source(pipeline_params: PipelineParameters) -> None:
def test_can_create_buildspec_v2() -> None:
