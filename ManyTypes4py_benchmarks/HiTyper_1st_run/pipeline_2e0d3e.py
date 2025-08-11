import copy
import re
from typing import List, Dict, Any, Optional, Callable
import yaml
from chalice.config import Config
from chalice import constants
from chalice import __version__ as chalice_version

def create_buildspec_v2(pipeline_params: Union[dict[str, typing.Any], list[dict]]) -> dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, dict[typing.Text, typing.Union[list[typing.Text],dict[typing.Text, ]]]],dict[typing.Text, typing.Union[typing.Text,list[typing.Text]]]]]:
    install_commands = ["pip install 'chalice%s'" % pipeline_params.chalice_version_range, 'pip install -r requirements.txt']
    build_commands = ['chalice package /tmp/packaged', 'aws cloudformation package --template-file /tmp/packaged/sam.json --s3-bucket ${APP_S3_BUCKET} --output-template-file transformed.yaml']
    buildspec = {'version': '0.2', 'phases': {'install': {'commands': install_commands, 'runtime-versions': {'python': pipeline_params.py_major_minor}}, 'build': {'commands': build_commands}}, 'artifacts': {'type': 'zip', 'files': ['transformed.yaml']}}
    return buildspec

def create_buildspec_legacy(pipeline_params: Union[list[dict], dict[str, typing.Any], dict[str, float], None]) -> dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, dict[typing.Text, list[typing.Text]]],dict[typing.Text, typing.Union[typing.Text,list[typing.Text]]]]]:
    install_commands = ['sudo pip install --upgrade awscli', 'aws --version', "sudo pip install 'chalice%s'" % pipeline_params.chalice_version_range, 'sudo pip install -r requirements.txt', 'chalice package /tmp/packaged', 'aws cloudformation package --template-file /tmp/packaged/sam.json --s3-bucket ${APP_S3_BUCKET} --output-template-file transformed.yaml']
    buildspec = {'version': '0.1', 'phases': {'install': {'commands': install_commands}}, 'artifacts': {'type': 'zip', 'files': ['transformed.yaml']}}
    return buildspec

class InvalidCodeBuildPythonVersion(Exception):

    def __init__(self, version, msg=None) -> None:
        if msg is None:
            msg = 'CodeBuild does not yet support python version %s.' % version
        super(InvalidCodeBuildPythonVersion, self).__init__(msg)

class PipelineParameters(object):
    _PYTHON_VERSION = re.compile('python(.+)')

    def __init__(self, app_name: Union[str, bool, None], lambda_python_version: Union[int, None], codebuild_image: Union[None, str, list["TextLineItemDraft"], list[str]]=None, code_source: typing.Text='codecommit', chalice_version_range: Union[None, str]=None, pipeline_version: typing.Text='v1') -> None:
        self.app_name = app_name
        self.lambda_python_version = lambda_python_version
        self.py_major_minor = self._extract_version(lambda_python_version)
        self.codebuild_image = codebuild_image
        self.code_source = code_source
        if chalice_version_range is None:
            chalice_version_range = self._lock_to_minor_version()
        self.chalice_version_range = chalice_version_range
        self.pipeline_version = pipeline_version

    def _extract_version(self, lambda_python_version: Union[str, int, list[int]]) -> str:
        matched = self._PYTHON_VERSION.match(lambda_python_version)
        if matched is None:
            raise InvalidCodeBuildPythonVersion(lambda_python_version)
        return matched.group(1)

    def _lock_to_minor_version(self) -> typing.Text:
        parts = [int(p) for p in chalice_version.split('.')]
        min_version = '%s.%s.%s' % (parts[0], parts[1], 0)
        max_version = '%s.%s.%s' % (parts[0], parts[1] + 1, 0)
        return '>=%s,<%s' % (min_version, max_version)

class BasePipelineTemplate(object):

    def create_template(self, pipeline_params: Union[list[dict], dict, config.CreateOptions]):
        raise NotImplementedError('create_template')

class CreatePipelineTemplateV2(BasePipelineTemplate):
    _BASE_TEMPLATE = {'AWSTemplateFormatVersion': '2010-09-09', 'Parameters': {'ApplicationName': {'Default': 'ChaliceApp', 'Type': 'String', 'Description': 'Enter the name of your application'}, 'CodeBuildImage': {'Default': 'aws/codebuild/amazonlinux2-x86_64-standard:3.0', 'Type': 'String', 'Description': 'Name of codebuild image to use.'}}, 'Resources': {}, 'Outputs': {}}

    def create_template(self, pipeline_params: Union[list[dict], dict, config.CreateOptions]):
        self._validate_python_version(pipeline_params.py_major_minor)
        t = copy.deepcopy(self._BASE_TEMPLATE)
        params = t['Parameters']
        params['ApplicationName']['Default'] = pipeline_params.app_name
        resources = []
        if pipeline_params.code_source == 'github':
            resources.append(GithubSource())
        else:
            resources.append(CodeCommitSourceRepository())
        resources.extend([CodeBuild(create_buildspec_v2), CodePipeline()])
        for resource in resources:
            resource.add_to_template(t, pipeline_params)
        return t

    def _validate_python_version(self, python_version: str) -> None:
        major, minor = [int(v) for v in python_version.split('.')]
        if (major, minor) < (3, 7):
            raise InvalidCodeBuildPythonVersion(python_version, 'This CodeBuild image does not support python version: %s' % python_version)

class CreatePipelineTemplateLegacy(BasePipelineTemplate):
    _CODEBUILD_IMAGE = {'python2.7': 'python:2.7.12', 'python3.6': 'python:3.6.5', 'python3.7': 'python:3.7.1'}
    _BASE_TEMPLATE = {'AWSTemplateFormatVersion': '2010-09-09', 'Parameters': {'ApplicationName': {'Default': 'ChaliceApp', 'Type': 'String', 'Description': 'Enter the name of your application'}, 'CodeBuildImage': {'Default': 'aws/codebuild/python:2.7.12', 'Type': 'String', 'Description': 'Name of codebuild image to use.'}}, 'Resources': {}, 'Outputs': {}}

    def create_template(self, pipeline_params: Union[list[dict], dict, config.CreateOptions]):
        t = copy.deepcopy(self._BASE_TEMPLATE)
        params = t['Parameters']
        params['ApplicationName']['Default'] = pipeline_params.app_name
        params['CodeBuildImage']['Default'] = self._get_codebuild_image(pipeline_params)
        resources = []
        if pipeline_params.code_source == 'github':
            resources.append(GithubSource())
        else:
            resources.append(CodeCommitSourceRepository())
        resources.extend([CodeBuild(create_buildspec_legacy), CodePipeline()])
        for resource in resources:
            resource.add_to_template(t, pipeline_params)
        return t

    def _get_codebuild_image(self, params: Union[dict, None, dict[str, str]]) -> typing.Text:
        if params.codebuild_image is not None:
            return params.codebuild_image
        try:
            image_suffix = self._CODEBUILD_IMAGE[params.lambda_python_version]
            return 'aws/codebuild/%s' % image_suffix
        except KeyError as e:
            raise InvalidCodeBuildPythonVersion(str(e))

class BaseResource(object):

    def add_to_template(self, template: Union[dict[str, typing.Any], core.common.Config], pipeline_params: Union[config.Config, dict, dict[str, typing.Any]]) -> None:
        raise NotImplementedError('add_to_template')

class CodeCommitSourceRepository(BaseResource):

    def add_to_template(self, template: Union[dict[str, typing.Any], core.common.Config], pipeline_params: Union[config.Config, dict, dict[str, typing.Any]]) -> None:
        resources = template.setdefault('Resources', {})
        resources['SourceRepository'] = {'Type': 'AWS::CodeCommit::Repository', 'Properties': {'RepositoryName': {'Ref': 'ApplicationName'}, 'RepositoryDescription': {'Fn::Sub': 'Source code for ${ApplicationName}'}}}
        template.setdefault('Outputs', {})['SourceRepoURL'] = {'Value': {'Fn::GetAtt': 'SourceRepository.CloneUrlHttp'}}

class GithubSource(BaseResource):

    def add_to_template(self, template: Union[dict[str, typing.Any], core.common.Config], pipeline_params: Union[config.Config, dict, dict[str, typing.Any]]) -> None:
        p = template.setdefault('Parameters', {})
        p['GithubOwner'] = {'Type': 'String', 'Description': 'The github owner or org name of the repository.'}
        p['GithubRepoName'] = {'Type': 'String', 'Description': 'The name of the github repository.'}
        if pipeline_params.pipeline_version == 'v1':
            p['GithubPersonalToken'] = {'Type': 'String', 'Description': 'Personal access token for the github repo.', 'NoEcho': True}
        else:
            p['GithubRepoSecretId'] = {'Type': 'String', 'Default': 'GithubRepoAccess', 'Description': 'The name/ID of the SecretsManager secret that contains the personal access token for the github repo.'}
            p['GithubRepoSecretJSONKey'] = {'Type': 'String', 'Default': 'OAuthToken', 'Description': 'The name of the JSON key in the SecretsManager secret that contains the personal access token for the github repo.'}

class CodeBuild(BaseResource):

    def __init__(self, buildspec_generator=create_buildspec_legacy) -> None:
        self._buildspec_generator = buildspec_generator

    def add_to_template(self, template: Union[dict[str, typing.Any], core.common.Config], pipeline_params: Union[config.Config, dict, dict[str, typing.Any]]) -> None:
        resources = template.setdefault('Resources', {})
        outputs = template.setdefault('Outputs', {})
        self._add_s3_bucket(resources, outputs)
        self._add_codebuild_role(resources, outputs)
        self._add_codebuild_policy(resources)
        self._add_package_build(resources, pipeline_params)

    def _add_package_build(self, resources: Any, pipeline_params: Union[dict[str, typing.Any], dict, None]) -> None:
        resources['AppPackageBuild'] = {'Type': 'AWS::CodeBuild::Project', 'Properties': {'Artifacts': {'Type': 'CODEPIPELINE'}, 'Environment': {'ComputeType': 'BUILD_GENERAL1_SMALL', 'Image': {'Ref': 'CodeBuildImage'}, 'Type': 'LINUX_CONTAINER', 'EnvironmentVariables': [{'Name': 'APP_S3_BUCKET', 'Value': {'Ref': 'ApplicationBucket'}}]}, 'Name': {'Fn::Sub': '${ApplicationName}Build'}, 'ServiceRole': {'Fn::GetAtt': 'CodeBuildRole.Arn'}, 'Source': {'Type': 'CODEPIPELINE', 'BuildSpec': yaml.dump(self._buildspec_generator(pipeline_params))}}}

    def _add_s3_bucket(self, resources: Any, outputs: Any) -> None:
        resources['ApplicationBucket'] = {'Type': 'AWS::S3::Bucket'}
        outputs['S3ApplicationBucket'] = {'Value': {'Ref': 'ApplicationBucket'}}

    def _add_codebuild_role(self, resources: Any, outputs: Any) -> None:
        resources['CodeBuildRole'] = {'Type': 'AWS::IAM::Role', 'Properties': {'AssumeRolePolicyDocument': {'Version': '2012-10-17', 'Statement': [{'Action': ['sts:AssumeRole'], 'Effect': 'Allow', 'Principal': {'Service': [{'Fn::Sub': 'codebuild.${AWS::URLSuffix}'}]}}]}}}
        outputs['CodeBuildRoleArn'] = {'Value': {'Fn::GetAtt': 'CodeBuildRole.Arn'}}

    def _add_codebuild_policy(self, resources: Any) -> None:
        resources['CodeBuildPolicy'] = {'Type': 'AWS::IAM::Policy', 'Properties': {'PolicyName': 'CodeBuildPolicy', 'PolicyDocument': constants.CODEBUILD_POLICY, 'Roles': [{'Ref': 'CodeBuildRole'}]}}

class CodePipeline(BaseResource):

    def add_to_template(self, template: Union[dict[str, typing.Any], core.common.Config], pipeline_params: Union[config.Config, dict, dict[str, typing.Any]]) -> None:
        resources = template.setdefault('Resources', {})
        outputs = template.setdefault('Outputs', {})
        self._add_pipeline(resources, pipeline_params)
        self._add_bucket_store(resources, outputs)
        self._add_codepipeline_role(resources, outputs)
        self._add_cfn_deploy_role(resources, outputs)

    def _add_cfn_deploy_role(self, resources: Any, outputs: Any) -> None:
        outputs['CFNDeployRoleArn'] = {'Value': {'Fn::GetAtt': 'CFNDeployRole.Arn'}}
        resources['CFNDeployRole'] = {'Type': 'AWS::IAM::Role', 'Properties': {'Policies': [{'PolicyName': 'DeployAccess', 'PolicyDocument': {'Version': '2012-10-17', 'Statement': [{'Action': '*', 'Resource': '*', 'Effect': 'Allow'}]}}], 'AssumeRolePolicyDocument': {'Version': '2012-10-17', 'Statement': [{'Action': ['sts:AssumeRole'], 'Effect': 'Allow', 'Principal': {'Service': [{'Fn::Sub': 'cloudformation.${AWS::URLSuffix}'}]}}]}}}

    def _add_pipeline(self, resources: Any, pipeline_params: Union[dict[str, typing.Any], dict]) -> None:
        properties = {'Name': {'Fn::Sub': '${ApplicationName}Pipeline'}, 'ArtifactStore': {'Type': 'S3', 'Location': {'Ref': 'ArtifactBucketStore'}}, 'RoleArn': {'Fn::GetAtt': 'CodePipelineRole.Arn'}, 'Stages': self._create_pipeline_stages(pipeline_params)}
        resources['AppPipeline'] = {'Type': 'AWS::CodePipeline::Pipeline', 'Properties': properties}

    def _create_pipeline_stages(self, pipeline_params: Union[dict, None, list[dict], dict[str, typing.Any]]) -> list[typing.Union[str,dict[str, str]]]:
        stages = []
        source = self._create_source_stage(pipeline_params)
        if source:
            stages.append(source)
        stages.extend([self._create_build_stage(), self._create_beta_stage()])
        return stages

    def _code_commit_source(self) -> dict[typing.Text, typing.Union[typing.Text,list[dict[typing.Text, typing.Union[dict[typing.Text, typing.Union[typing.Text,int]],dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, typing.Text]]],list[dict[typing.Text, typing.Text]],int,typing.Text]]]]]:
        return {'Name': 'Source', 'Actions': [{'ActionTypeId': {'Category': 'Source', 'Owner': 'AWS', 'Version': 1, 'Provider': 'CodeCommit'}, 'Configuration': {'BranchName': 'master', 'RepositoryName': {'Fn::GetAtt': 'SourceRepository.Name'}}, 'OutputArtifacts': [{'Name': 'SourceRepo'}], 'RunOrder': 1, 'Name': 'Source'}]}

    def _create_source_stage(self, pipeline_params: Union[str, dict[str, typing.Any], entity_embeddings.Config]) -> Union[dict[str, str], str, dict]:
        if pipeline_params.code_source == 'codecommit':
            return self._code_commit_source()
        return self._github_source(pipeline_params.pipeline_version)

    def _github_source(self, pipeline_version: Union[str, int]) -> dict[typing.Text, typing.Union[typing.Text,list[dict[typing.Text, typing.Union[typing.Text,int,list[dict[typing.Text, typing.Text]],dict[typing.Text, typing.Union[dict[typing.Text, typing.Text],dict[typing.Text, list[typing.Union[typing.Text,list[typing.Union[typing.Text,dict[typing.Text, typing.Text]]]]]],typing.Text,bool]]]]]]]:
        oauth_token = {'Ref': 'GithubPersonalToken'}
        if pipeline_version == 'v2':
            oauth_token = {'Fn::Join': ['', ['{{resolve:secretsmanager:', {'Ref': 'GithubRepoSecretId'}, ':SecretString:', {'Ref': 'GithubRepoSecretJSONKey'}, '}}']]}
        return {'Name': 'Source', 'Actions': [{'Name': 'Source', 'ActionTypeId': {'Category': 'Source', 'Owner': 'ThirdParty', 'Version': '1', 'Provider': 'GitHub'}, 'RunOrder': 1, 'OutputArtifacts': [{'Name': 'SourceRepo'}], 'Configuration': {'Owner': {'Ref': 'GithubOwner'}, 'Repo': {'Ref': 'GithubRepoName'}, 'OAuthToken': oauth_token, 'Branch': 'master', 'PollForSourceChanges': True}}]}

    def _create_build_stage(self) -> dict[typing.Text, typing.Union[typing.Text,list[dict[typing.Text, typing.Union[list[dict[typing.Text, typing.Text]],typing.Text,dict[typing.Text, typing.Text],dict[typing.Text, dict[typing.Text, typing.Text]],int]]]]]:
        return {'Name': 'Build', 'Actions': [{'InputArtifacts': [{'Name': 'SourceRepo'}], 'Name': 'CodeBuild', 'ActionTypeId': {'Category': 'Build', 'Owner': 'AWS', 'Version': '1', 'Provider': 'CodeBuild'}, 'OutputArtifacts': [{'Name': 'CompiledCFNTemplate'}], 'Configuration': {'ProjectName': {'Ref': 'AppPackageBuild'}}, 'RunOrder': 1}]}

    def _create_beta_stage(self) -> dict[typing.Text, typing.Union[typing.Text,list[dict[typing.Text, typing.Union[list[dict[typing.Text, typing.Text]],typing.Text,dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, typing.Text]]],int]]]]]:
        return {'Name': 'Beta', 'Actions': [{'ActionTypeId': {'Category': 'Deploy', 'Owner': 'AWS', 'Version': '1', 'Provider': 'CloudFormation'}, 'InputArtifacts': [{'Name': 'CompiledCFNTemplate'}], 'Name': 'CreateBetaChangeSet', 'Configuration': {'ActionMode': 'CHANGE_SET_REPLACE', 'ChangeSetName': {'Fn::Sub': '${ApplicationName}ChangeSet'}, 'RoleArn': {'Fn::GetAtt': 'CFNDeployRole.Arn'}, 'Capabilities': 'CAPABILITY_IAM', 'StackName': {'Fn::Sub': '${ApplicationName}BetaStack'}, 'TemplatePath': 'CompiledCFNTemplate::transformed.yaml'}, 'RunOrder': 1}, {'RunOrder': 2, 'ActionTypeId': {'Category': 'Deploy', 'Owner': 'AWS', 'Version': '1', 'Provider': 'CloudFormation'}, 'Configuration': {'StackName': {'Fn::Sub': '${ApplicationName}BetaStack'}, 'ActionMode': 'CHANGE_SET_EXECUTE', 'ChangeSetName': {'Fn::Sub': '${ApplicationName}ChangeSet'}, 'OutputFileName': 'StackOutputs.json'}, 'Name': 'ExecuteChangeSet', 'OutputArtifacts': [{'Name': 'AppDeploymentValues'}]}]}

    def _add_bucket_store(self, resources: Any, outputs: Any) -> None:
        resources['ArtifactBucketStore'] = {'Type': 'AWS::S3::Bucket', 'Properties': {'VersioningConfiguration': {'Status': 'Enabled'}}}
        outputs['S3PipelineBucket'] = {'Value': {'Ref': 'ArtifactBucketStore'}}

    def _add_codepipeline_role(self, resources: Any, outputs: Any) -> None:
        outputs['CodePipelineRoleArn'] = {'Value': {'Fn::GetAtt': 'CodePipelineRole.Arn'}}
        resources['CodePipelineRole'] = {'Type': 'AWS::IAM::Role', 'Properties': {'Policies': [{'PolicyName': 'DefaultPolicy', 'PolicyDocument': constants.CODEPIPELINE_POLICY}], 'AssumeRolePolicyDocument': {'Version': '2012-10-17', 'Statement': [{'Action': ['sts:AssumeRole'], 'Effect': 'Allow', 'Principal': {'Service': [{'Fn::Sub': 'codepipeline.${AWS::URLSuffix}'}]}}]}}}

class BuildSpecExtractor(object):

    def extract_buildspec(self, template: str) -> list[str]:
        source = template['Resources']['AppPackageBuild']['Properties']['Source']
        buildspec = source.pop('BuildSpec')
        return buildspec