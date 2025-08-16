from chalice.pipeline import PipelineParameters, InvalidCodeBuildPythonVersion
from typing import Dict, Any

def test_source_repo_resource(pipeline_params: PipelineParameters) -> None:
    template: Dict[str, Any] = {}
    pipeline.CodeCommitSourceRepository().add_to_template(template, pipeline_params)
    assert template == {'Resources': {'SourceRepository': {'Type': 'AWS::CodeCommit::Repository', 'Properties': {'RepositoryName': {'Ref': 'ApplicationName'}, 'RepositoryDescription': {'Fn::Sub': 'Source code for ${ApplicationName}'}}}}, 'Outputs': {'SourceRepoURL': {'Value': {'Fn::GetAtt': 'SourceRepository.CloneUrlHttp'}}}}

def test_codebuild_resource(pipeline_params: PipelineParameters) -> None:
    template: Dict[str, Any] = {}
    pipeline.CodeBuild().add_to_template(template, pipeline_params)
    resources = template['Resources']
    assert 'ApplicationBucket' in resources
    assert 'CodeBuildRole' in resources
    assert 'CodeBuildPolicy' in resources
    assert 'AppPackageBuild' in resources
    assert resources['ApplicationBucket'] == {'Type': 'AWS::S3::Bucket'}
    assert template['Outputs']['CodeBuildRoleArn'] == {'Value': {'Fn::GetAtt': 'CodeBuildRole.Arn'}}

def test_codepipeline_resource(pipeline_params: PipelineParameters) -> None:
    template: Dict[str, Any] = {}
    pipeline.CodePipeline().add_to_template(template, pipeline_params)
    resources = template['Resources']
    assert 'AppPipeline' in resources
    assert 'ArtifactBucketStore' in resources
    assert 'CodePipelineRole' in resources
    assert 'CFNDeployRole' in resources
    assert resources['AppPipeline']['Type'] == 'AWS::CodePipeline::Pipeline'
    assert resources['ArtifactBucketStore']['Type'] == 'AWS::S3::Bucket'
    assert resources['CodePipelineRole']['Type'] == 'AWS::IAM::Role'
    assert resources['CFNDeployRole']['Type'] == 'AWS::IAM::Role'
    properties = resources['AppPipeline']['Properties']
    stages = properties['Stages']
    beta_stage = stages[2]
    beta_config = beta_stage['Actions'][0]['Configuration']
    assert beta_config == {'ActionMode': 'CHANGE_SET_REPLACE', 'Capabilities': 'CAPABILITY_IAM', 'ChangeSetName': {'Fn::Sub': '${ApplicationName}ChangeSet'}, 'RoleArn': {'Fn::GetAtt': 'CFNDeployRole.Arn'}, 'StackName': {'Fn::Sub': '${ApplicationName}BetaStack'}, 'TemplatePath': 'CompiledCFNTemplate::transformed.yaml'}

def test_install_requirements_in_buildspec(pipeline_params: PipelineParameters) -> None:
    template: Dict[str, Any] = {}
    pipeline_params.chalice_version_range = '>=1.0.0,<2.0.0'
    pipeline.CodeBuild().add_to_template(template, pipeline_params)
    build = template['Resources']['AppPackageBuild']
    build_spec = build['Properties']['Source']['BuildSpec']
    assert 'pip install -r requirements.txt' in build_spec
    assert "pip install 'chalice>=1.0.0,<2.0.0'" in build_spec

def test_default_version_range_locks_minor_version() -> None:
    parts = [int(p) for p in chalice_version.split('.')]
    min_version = '%s.%s.%s' % (parts[0], parts[1], 0)
    max_version = '%s.%s.%s' % (parts[0], parts[1] + 1, 0)
    params = pipeline.PipelineParameters('appname', 'python2.7')
    assert params.chalice_version_range == '>=%s,<%s' % (min_version, max_version)

def test_can_validate_python_version() -> None:
    with pytest.raises(InvalidCodeBuildPythonVersion):
        pipeline.PipelineParameters('myapp', lambda_python_version='bad-python-value')

def test_can_extract_python_version() -> None:
    assert pipeline.PipelineParameters('app', 'python3.7').py_major_minor == '3.7'

def test_can_generate_github_source(pipeline_params: PipelineParameters) -> None:
    template: Dict[str, Any] = {}
    pipeline_params.code_source = 'github'
    pipeline.GithubSource().add_to_template(template, pipeline_params)
    cfn_params = template['Parameters']
    assert set(cfn_params) == set(['GithubOwner', 'GithubRepoName', 'GithubPersonalToken'])

def test_can_create_buildspec_v2() -> None:
    params = pipeline.PipelineParameters('myapp', 'python3.7')
    buildspec = pipeline.create_buildspec_v2(params)
    assert buildspec['phases']['install']['runtime-versions'] == {'python': '3.7'}

def test_build_extractor() -> None:
    template = {'Resources': {'AppPackageBuild': {'Properties': {'Source': {'BuildSpec': 'foobar'}}}}}
    extract = pipeline.BuildSpecExtractor()
    extracted = extract.extract_buildspec(template)
    assert extracted == 'foobar'
    assert 'BuildSpec' not in template['Resources']['AppPackageBuild']['Properties']['Source']
