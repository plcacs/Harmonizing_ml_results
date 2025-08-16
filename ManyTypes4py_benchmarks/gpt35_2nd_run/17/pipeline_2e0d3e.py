from typing import List, Dict, Any, Optional, Callable

def create_buildspec_v2(pipeline_params: PipelineParameters) -> Dict[str, Any]:
    ...

def create_buildspec_legacy(pipeline_params: PipelineParameters) -> Dict[str, Any]:
    ...

class InvalidCodeBuildPythonVersion(Exception):

    def __init__(self, version: str, msg: Optional[str] = None) -> None:
        ...

class PipelineParameters(object):

    def __init__(self, app_name: str, lambda_python_version: str, codebuild_image: Optional[str] = None, code_source: str = 'codecommit', chalice_version_range: Optional[str] = None, pipeline_version: str = 'v1') -> None:
        ...

class BasePipelineTemplate(object):

    def create_template(self, pipeline_params: PipelineParameters) -> Dict[str, Any]:
        ...

class CreatePipelineTemplateV2(BasePipelineTemplate):

    def create_template(self, pipeline_params: PipelineParameters) -> Dict[str, Any]:
        ...

class CreatePipelineTemplateLegacy(BasePipelineTemplate):

    def create_template(self, pipeline_params: PipelineParameters) -> Dict[str, Any]:
        ...

class BaseResource(object):

    def add_to_template(self, template: Dict[str, Any], pipeline_params: PipelineParameters) -> None:
        ...

class CodeCommitSourceRepository(BaseResource):

    def add_to_template(self, template: Dict[str, Any], pipeline_params: PipelineParameters) -> None:
        ...

class GithubSource(BaseResource):

    def add_to_template(self, template: Dict[str, Any], pipeline_params: PipelineParameters) -> None:
        ...

class CodeBuild(BaseResource):

    def __init__(self, buildspec_generator: Callable[[PipelineParameters], Dict[str, Any]] = create_buildspec_legacy) -> None:
        ...

    def add_to_template(self, template: Dict[str, Any], pipeline_params: PipelineParameters) -> None:
        ...

class CodePipeline(BaseResource):

    def add_to_template(self, template: Dict[str, Any], pipeline_params: PipelineParameters) -> None:
        ...

class BuildSpecExtractor(object):

    def extract_buildspec(self, template: Dict[str, Any]) -> str:
        ...
