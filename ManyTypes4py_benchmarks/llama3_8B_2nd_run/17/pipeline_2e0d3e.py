import copy
import re
from typing import List, Dict, Any, Optional, Callable
import yaml
from chalice.config import Config
from chalice import constants
from chalice import __version__ as chalice_version

def create_buildspec_v2(pipeline_params: PipelineParameters) -> Dict[str, Any]:
    ...

def create_buildspec_legacy(pipeline_params: PipelineParameters) -> Dict[str, Any]:
    ...

class InvalidCodeBuildPythonVersion(Exception):
    ...

class PipelineParameters:
    ...

class BasePipelineTemplate:
    ...

class CreatePipelineTemplateV2(BasePipelineTemplate):
    ...

class CreatePipelineTemplateLegacy(BasePipelineTemplate):
    ...

class BaseResource:
    ...

class CodeCommitSourceRepository(BaseResource):
    ...

class GithubSource(BaseResource):
    ...

class CodeBuild(BaseResource):
    ...

class CodePipeline(BaseResource):
    ...

class BuildSpecExtractor:
    ...
