"""Module containing tasks and flows for interacting with dbt CLI"""
import os
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Optional, Union
import yaml
from dbt.artifacts.schemas.freshness.v3.freshness import FreshnessResult
from dbt.artifacts.schemas.run.v5.run import RunResult, RunResultOutput
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.contracts.results import ExecutionResult, NodeStatus
from prefect_shell.commands import ShellOperation
from pydantic import Field
from prefect import task
from prefect.artifacts import acreate_markdown_artifact
from prefect.logging import get_run_logger
from prefect.states import Failed
from prefect.utilities.asyncutils import sync_compatible
from prefect.utilities.filesystem import relative_path_to_current_platform
from prefect_dbt.cli.credentials import DbtCliProfile

@task
@sync_compatible
async def trigger_dbt_cli_command(
    command: str,
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-cli-command-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Union[str, List[str], Failed]:
    logger = get_run_logger()
    if profiles_dir is None:
        profiles_dir = os.getenv('DBT_PROFILES_DIR', str(Path.home()) + '/.dbt')
    profiles_path = profiles_dir + '/profiles.yml'
    logger.debug(f'Using this profiles path: {profiles_path}')
    if overwrite_profiles or not Path(profiles_path).expanduser().exists():
        if dbt_cli_profile is None:
            raise ValueError(f'Profile not found. Provide `dbt_cli_profile` or ensure profiles.yml exists at {profiles_path}.')
        profile = dbt_cli_profile.get_profile()
        Path(profiles_dir).expanduser().mkdir(exist_ok=True)
        with open(profiles_path, 'w+') as f:
            yaml.dump(profile, f, default_flow_style=False)
        logger.info(f'Wrote profile to {profiles_path}')
    elif dbt_cli_profile is not None:
        raise ValueError(f'Since overwrite_profiles is False and profiles_path ({profiles_path}) already exists, the profile within dbt_cli_profile could not be used; if the existing profile is satisfactory, do not pass dbt_cli_profile')
    cli_args = [arg for arg in command.split() if arg != 'dbt']
    cli_args.append('--profiles-dir')
    cli_args.append(profiles_dir)
    if project_dir is not None:
        project_dir = Path(project_dir).expanduser()
        cli_args.append('--project-dir')
        cli_args.append(project_dir)
    if extra_command_args:
        for value in extra_command_args:
            cli_args.append(value)
    callbacks = []
    if stream_output:

        def _stream_output(event: Any) -> None:
            if event.info.level != 'debug':
                logger.info(event.info.msg)
        callbacks.append(_stream_output)
    dbt_runner_client = dbtRunner(callbacks=callbacks)
    logger.info(f'Running dbt command: {cli_args}')
    result = dbt_runner_client.invoke(cli_args)
    if result.exception is not None:
        logger.error(f'dbt task failed with exception: {result.exception}')
        raise result.exception
    if create_summary_artifact and isinstance(result.result, ExecutionResult):
        run_results = consolidate_run_results(result)
        markdown = create_summary_markdown(run_results, command)
        artifact_id = await acreate_markdown_artifact(markdown=markdown, key=summary_artifact_key)
        if not artifact_id:
            logger.error(f'Summary Artifact was not created for dbt {command} task')
        else:
            logger.info(f'dbt {command} task completed successfully with artifact {artifact_id}')
    else:
        logger.debug(f'Artifacts were not created for dbt {command} this task due to create_artifact=False or the dbt command did not return any RunExecutionResults. See https://docs.getdbt.com/reference/programmatic-invocations for more details on dbtRunnerResult.')
    if isinstance(result.result, ExecutionResult) and (not result.success):
        return Failed(message=f'dbt task result success: {result.success} with exception: {result.exception}')
    return result

class DbtCoreOperation(ShellOperation):
    profiles_dir: Optional[Union[str, PosixPath]] = Field(default=None, description="The directory to search for the profiles.yml file. Setting this appends the `--profiles-dir` option to the dbt commands provided. If this is not set, will try using the DBT_PROFILES_DIR environment variable, but if that's also not set, will use the default directory `$HOME/.dbt/`.")
    project_dir: Optional[Union[str, PosixPath]] = Field(default=None, description='The directory to search for the dbt_project.yml file. Default is the current working directory and its parents.')
    overwrite_profiles: bool = Field(default=False, description='Whether the existing profiles.yml file under profiles_dir should be overwritten with a new profile.')
    dbt_cli_profile: Optional[DbtCliProfile] = Field(default=None, description='Profiles class containing the profile written to profiles.yml. Note! This is optional and will raise an error if profiles.yml already exists under profile_dir and overwrite_profiles is set to False.')

    def _find_valid_profiles_dir(self) -> PosixPath:
        profiles_dir = self.profiles_dir
        if profiles_dir is None:
            if self.env.get('DBT_PROFILES_DIR') is not None:
                profiles_dir = self.env['DBT_PROFILES_DIR']
            else:
                profiles_dir = os.getenv('DBT_PROFILES_DIR', Path.home() / '.dbt')
        profiles_dir = relative_path_to_current_platform(Path(profiles_dir).expanduser())
        profiles_path = profiles_dir / 'profiles.yml'
        overwrite_profiles = self.overwrite_profiles
        dbt_cli_profile = self.dbt_cli_profile
        if not profiles_path.exists() or overwrite_profiles:
            if dbt_cli_profile is None:
                raise ValueError('Since overwrite_profiles is True or profiles_path is empty, need `dbt_cli_profile` to write a profile')
            profile = dbt_cli_profile.get_profile()
            profiles_dir.mkdir(exist_ok=True)
            with open(profiles_path, 'w+') as f:
                yaml.dump(profile, f, default_flow_style=False)
        elif dbt_cli_profile is not None:
            raise ValueError(f"Since overwrite_profiles is False and profiles_path {profiles_path} already exists, the profile within dbt_cli_profile couldn't be used; if the existing profile is satisfactory, do not set dbt_cli_profile")
        return profiles_dir

    def _append_dirs_to_commands(self, profiles_dir: PosixPath) -> List[str]:
        project_dir = self.project_dir
        commands = []
        for command in self.commands:
            command += f' --profiles-dir {profiles_dir}'
            if project_dir is not None:
                project_dir = Path(project_dir).expanduser()
                command += f' --project-dir {project_dir}'
            commands.append(command)
        return commands

    def _compile_kwargs(self, **open_kwargs: Any) -> Dict[str, Any]:
        profiles_dir = self._find_valid_profiles_dir()
        commands = self._append_dirs_to_commands(profiles_dir=profiles_dir)
        modified_self = self.copy()
        modified_self.commands = commands
        return super(type(self), modified_self)._compile_kwargs(**open_kwargs)

@sync_compatible
@task
async def run_dbt_build(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-build-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Union[str, List[str], Failed]:
    results = await trigger_dbt_cli_command.fn(command='build', profiles_dir=profiles_dir, project_dir=project_dir, overwrite_profiles=overwrite_profiles, dbt_cli_profile=dbt_cli_profile, create_summary_artifact=create_summary_artifact, summary_artifact_key=summary_artifact_key, extra_command_args=extra_command_args, stream_output=stream_output)
    return results

@sync_compatible
@task
async def run_dbt_model(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-run-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Union[str, List[str], Failed]:
    results = await trigger_dbt_cli_command.fn(command='run', profiles_dir=profiles_dir, project_dir=project_dir, overwrite_profiles=overwrite_profiles, dbt_cli_profile=dbt_cli_profile, create_summary_artifact=create_summary_artifact, summary_artifact_key=summary_artifact_key, extra_command_args=extra_command_args, stream_output=stream_output)
    return results

@sync_compatible
@task
async def run_dbt_test(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-test-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Union[str, List[str], Failed]:
    results = await trigger_dbt_cli_command.fn(command='test', profiles_dir=profiles_dir, project_dir=project_dir, overwrite_profiles=overwrite_profiles, dbt_cli_profile=dbt_cli_profile, create_summary_artifact=create_summary_artifact, summary_artifact_key=summary_artifact_key, extra_command_args=extra_command_args, stream_output=stream_output)
    return results

@sync_compatible
@task
async def run_dbt_snapshot(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-snapshot-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Union[str, List[str], Failed]:
    results = await trigger_dbt_cli_command.fn(command='snapshot', profiles_dir=profiles_dir, project_dir=project_dir, overwrite_profiles=overwrite_profiles, dbt_cli_profile=dbt_cli_profile, create_summary_artifact=create_summary_artifact, summary_artifact_key=summary_artifact_key, extra_command_args=extra_command_args, stream_output=stream_output)
    return results

@sync_compatible
@task
async def run_dbt_seed(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-seed-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Union[str, List[str], Failed]:
    results = await trigger_dbt_cli_command.fn(command='seed', profiles_dir=profiles_dir, project_dir=project_dir, overwrite_profiles=overwrite_profiles, dbt_cli_profile=dbt_cli_profile, create_summary_artifact=create_summary_artifact, summary_artifact_key=summary_artifact_key, extra_command_args=extra_command_args, stream_output=stream_output)
    return results

@sync_compatible
@task
async def run_dbt_source_freshness(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-source-freshness-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Union[str, List[str], Failed]:
    results = await trigger_dbt_cli_command.fn(command='source freshness', profiles_dir=profiles_dir, project_dir=project_dir, overwrite_profiles=overwrite_profiles, dbt_cli_profile=dbt_cli_profile, create_summary_artifact=create_summary_artifact, summary_artifact_key=summary_artifact_key, extra_command_args=extra_command_args, stream_output=stream_output)
    return results

def create_summary_markdown(run_results: Dict[str, List[Any]], command: str) -> str:
    prefix = 'dbt' if not command.startswith('dbt') else ''
    markdown = f'# {prefix} {command} Task Summary\n'
    markdown += _create_node_summary_table_md(run_results=run_results)
    if run_results['Error'] != [] or run_results['Fail'] != [] or run_results['Skipped'] != [] or (run_results['Warn'] != []):
        markdown += '\n\n ## Unsuccessful Nodes ❌\n\n'
        markdown += _create_unsuccessful_markdown(run_results=run_results)
    if run_results['Success'] != []:
        successful_runs_str = ''
        for r in run_results['Success']:
            if isinstance(r, (RunResult, FreshnessResult)):
                successful_runs_str += f'* {r.node.name}\n'
            elif isinstance(r, RunResultOutput):
                successful_runs_str += f'* {r.unique_id}\n'
            else:
                successful_runs_str += f'* {r}\n'
        markdown += f'\n## Successful Nodes ✅\n\n{successful_runs_str}\n\n'
    return markdown

def _create_node_info_md(node_name: str, resource_type: str, message: str, path: str, compiled_code: Optional[str]) -> str:
    markdown = f'\n**{node_name}**\n\nType: {resource_type}\n\nMessage: \n\n> {message}\n\n\nPath: {path}\n\n'
    if compiled_code:
        markdown += f'\nCompiled code:\n\n