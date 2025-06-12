"""Module containing tasks and flows for interacting with dbt CLI"""
import os
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Optional, Union, Callable
import yaml
from dbt.artifacts.schemas.freshness.v3.freshness import FreshnessResult
from dbt.artifacts.schemas.run.v5.run import RunResult, RunResultOutput
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.contracts.results import ExecutionResult, NodeStatus
from prefect_shell.commands import ShellOperation
from pydantic import Field
from prefect import task, Flow
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
) -> Union[dbtRunnerResult, Failed]:
    """
    Task for running dbt commands.

    If no profiles.yml file is found or if overwrite_profiles flag is set to True, this
    will first generate a profiles.yml file in the profiles_dir directory. Then run the dbt
    CLI shell command.

    Args:
        command: The dbt command to be executed.
        profiles_dir: The directory to search for the profiles.yml file. Setting this
            appends the `--profiles-dir` option to the command provided. If this is not set,
            will try using the DBT_PROFILES_DIR environment variable, but if that's also not
            set, will use the default directory `$HOME/.dbt/`.
        project_dir: The directory to search for the dbt_project.yml file.
            Default is the current working directory and its parents.
        overwrite_profiles: Whether the existing profiles.yml file under profiles_dir
            should be overwritten with a new profile.
        dbt_cli_profile: Profiles class containing the profile written to profiles.yml.
            Note! This is optional and will raise an error if profiles.yml already exists
            under profile_dir and overwrite_profiles is set to False.
        create_summary_artifact: If True, creates a Prefect artifact on the task run
            with the dbt results using the specified artifact key.
            Defaults to False.
        summary_artifact_key: The key under which to store the dbt results artifact in Prefect.
            Defaults to 'dbt-cli-command-summary'.
        extra_command_args: Additional command arguments to pass to the dbt command.
            These arguments get appended to the command that gets passed to the dbtRunner client.
            Example: extra_command_args=["--model", "foo_model"]
        stream_output: If True, the output from the dbt command will be logged in Prefect
            as it happens.
            Defaults to True.

    Returns:
        last_line_cli_output (dbtRunnerResult or Failed): The dbtRunnerResult object or a Failed state.
    """
    logger = get_run_logger()
    if profiles_dir is None:
        profiles_dir = os.getenv('DBT_PROFILES_DIR', str(Path.home()) + '/.dbt')
    profiles_path = os.path.join(profiles_dir, 'profiles.yml')
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
    
    cli_args: List[str] = [arg for arg in command.split() if arg != 'dbt']
    cli_args.append('--profiles-dir')
    cli_args.append(profiles_dir)
    if project_dir is not None:
        project_dir_path = Path(project_dir).expanduser()
        cli_args.append('--project-dir')
        cli_args.append(str(project_dir_path))
    if extra_command_args:
        cli_args.extend(extra_command_args)
    
    callbacks: List[Callable[[Any], None]] = []
    if stream_output:

        def _stream_output(event: Any) -> None:
            if event.info.level != 'debug':
                logger.info(event.info.msg)

        callbacks.append(_stream_output)
    
    dbt_runner_client: dbtRunner = dbtRunner(callbacks=callbacks)
    logger.info(f'Running dbt command: {cli_args}')
    result: dbtRunnerResult = dbt_runner_client.invoke(cli_args)
    
    if result.exception is not None:
        logger.error(f'dbt task failed with exception: {result.exception}')
        raise result.exception
    
    if create_summary_artifact and isinstance(result.result, ExecutionResult):
        run_results: Dict[str, List[Union[RunResult, FreshnessResult, RunResultOutput]]] = consolidate_run_results(result)
        markdown: str = create_summary_markdown(run_results, command)
        artifact_id: Optional[str] = await acreate_markdown_artifact(markdown=markdown, key=summary_artifact_key)
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
    """
    A block representing a dbt operation, containing multiple dbt and shell commands.

    For long-lasting operations, use the trigger method and utilize the block as a
    context manager for automatic closure of processes when context is exited.
    If not, manually call the close method to close processes.

    For short-lasting operations, use the run method. Context is automatically managed
    with this method.

    Attributes:
        commands: A list of commands to execute sequentially.
        stream_output: Whether to stream output.
        env: A dictionary of environment variables to set for the shell operation.
        working_dir: The working directory context the commands will be executed within.
        shell: The shell to use to execute the commands.
        extension: The extension to use for the temporary file.
            if unset defaults to `.ps1` on Windows and `.sh` on other platforms.
        profiles_dir: The directory to search for the profiles.yml file.
            Setting this appends the `--profiles-dir` option to the dbt commands
            provided. If this is not set, will try using the DBT_PROFILES_DIR
            environment variable, but if that's also not
            set, will use the default directory `$HOME/.dbt/`.
        project_dir: The directory to search for the dbt_project.yml file.
            Default is the current working directory and its parents.
        overwrite_profiles: Whether the existing profiles.yml file under profiles_dir
            should be overwritten with a new profile.
        dbt_cli_profile: Profiles class containing the profile written to profiles.yml.
            Note! This is optional and will raise an error if profiles.yml already
            exists under profile_dir and overwrite_profiles is set to False.

    Examples:
        Load a configured block.
        