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
    project_dir: Optional[Union[str, Path]] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = "dbt-cli-command-summary",
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True,
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
        last_line_cli_output (str): The last line of the CLI output will be returned
            if `return_all` in `shell_run_command_kwargs` is False. This is the default
            behavior.
        full_cli_output (List[str]): Full CLI output will be returned if `return_all`
            in `shell_run_command_kwargs` is True.
    """
    logger = get_run_logger()
    if profiles_dir is None:
        profiles_dir = os.getenv("DBT_PROFILES_DIR", str(Path.home()) + "/.dbt")
    profiles_path: str = profiles_dir + "/profiles.yml"
    logger.debug(f"Using this profiles path: {profiles_path}")
    if overwrite_profiles or not Path(profiles_path).expanduser().exists():
        if dbt_cli_profile is None:
            raise ValueError(
                f"Profile not found. Provide `dbt_cli_profile` or ensure profiles.yml exists at {profiles_path}."
            )
        profile: Dict[str, Any] = dbt_cli_profile.get_profile()
        Path(profiles_dir).expanduser().mkdir(exist_ok=True)
        with open(profiles_path, "w+") as f:
            yaml.dump(profile, f, default_flow_style=False)
        logger.info(f"Wrote profile to {profiles_path}")
    elif dbt_cli_profile is not None:
        raise ValueError(
            f"Since overwrite_profiles is False and profiles_path ({profiles_path}) already exists, the profile within dbt_cli_profile could not be used; if the existing profile is satisfactory, do not pass dbt_cli_profile"
        )
    cli_args: List[Union[str, Path]] = [arg for arg in command.split() if arg != "dbt"]
    cli_args.append("--profiles-dir")
    cli_args.append(profiles_dir)
    if project_dir is not None:
        project_dir = Path(project_dir).expanduser()
        cli_args.append("--project-dir")
        cli_args.append(project_dir)
    if extra_command_args:
        for value in extra_command_args:
            cli_args.append(value)
    callbacks: List[Callable[[Any], None]] = []
    if stream_output:

        def _stream_output(event: Any) -> None:
            if event.info.level != "debug":
                logger.info(event.info.msg)

        callbacks.append(_stream_output)
    dbt_runner_client = dbtRunner(callbacks=callbacks)
    logger.info(f"Running dbt command: {cli_args}")
    result: dbtRunnerResult = dbt_runner_client.invoke(cli_args)
    if result.exception is not None:
        logger.error(f"dbt task failed with exception: {result.exception}")
        raise result.exception
    if create_summary_artifact and isinstance(result.result, ExecutionResult):
        run_results = consolidate_run_results(result)
        markdown = create_summary_markdown(run_results, command)
        artifact_id = await acreate_markdown_artifact(
            markdown=markdown, key=summary_artifact_key
        )
        if not artifact_id:
            logger.error(f"Summary Artifact was not created for dbt {command} task")
        else:
            logger.info(
                f"dbt {command} task completed successfully with artifact {artifact_id}"
            )
    else:
        logger.debug(
            f"Artifacts were not created for dbt {command} this task due to create_artifact=False or the dbt command did not return any RunExecutionResults. See https://docs.getdbt.com/reference/programmatic-invocations for more details on dbtRunnerResult."
        )
    if isinstance(result.result, ExecutionResult) and (not result.success):
        return Failed(
            message=f"dbt task result success: {result.success} with exception: {result.exception}"
        )
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
    """

    _block_type_name = "dbt Core Operation"
    _logo_url = "https://images.ctfassets.net/gm98wzqotmnx/5zE9lxfzBHjw3tnEup4wWL/9a001902ed43a84c6c96d23b24622e19/dbt-bit_tm.png?h=250"
    _documentation_url = "https://docs.prefect.io/integrations/prefect-dbt"
    profiles_dir: Optional[Union[str, Path]] = Field(
        default=None,
        description="The directory to search for the profiles.yml file. Setting this appends the `--profiles-dir` option to the dbt commands provided. If this is not set, will try using the DBT_PROFILES_DIR environment variable, but if that's also not set, will use the default directory `$HOME/.dbt/`.",
    )
    project_dir: Optional[Union[str, Path]] = Field(
        default=None,
        description="The directory to search for the dbt_project.yml file. Default is the current working directory and its parents.",
    )
    overwrite_profiles: bool = Field(
        default=False,
        description="Whether the existing profiles.yml file under profiles_dir should be overwritten with a new profile.",
    )
    dbt_cli_profile: Optional[DbtCliProfile] = Field(
        default=None,
        description="Profiles class containing the profile written to profiles.yml. Note! This is optional and will raise an error if profiles.yml already exists under profile_dir and overwrite_profiles is set to False.",
    )

    def _find_valid_profiles_dir(self) -> Path:
        """
        Ensure that there is a profiles.yml available for use.
        """
        profiles_dir: Union[str, Path, None] = self.profiles_dir
        if profiles_dir is None:
            if self.env.get("DBT_PROFILES_DIR") is not None:
                profiles_dir = self.env["DBT_PROFILES_DIR"]
            else:
                profiles_dir = os.getenv("DBT_PROFILES_DIR", Path.home() / ".dbt")
        profiles_dir = relative_path_to_current_platform(Path(profiles_dir).expanduser())
        profiles_path: Path = profiles_dir / "profiles.yml"
        overwrite_profiles: bool = self.overwrite_profiles
        dbt_cli_profile: Optional[DbtCliProfile] = self.dbt_cli_profile
        if not profiles_path.exists() or overwrite_profiles:
            if dbt_cli_profile is None:
                raise ValueError(
                    "Since overwrite_profiles is True or profiles_path is empty, need `dbt_cli_profile` to write a profile"
                )
            profile: Dict[str, Any] = dbt_cli_profile.get_profile()
            profiles_dir.mkdir(exist_ok=True)
            with open(profiles_path, "w+") as f:
                yaml.dump(profile, f, default_flow_style=False)
        elif dbt_cli_profile is not None:
            raise ValueError(
                f"Since overwrite_profiles is False and profiles_path {profiles_path} already exists, the profile within dbt_cli_profile couldn't be used; if the existing profile is satisfactory, do not set dbt_cli_profile"
            )
        return profiles_dir

    def _append_dirs_to_commands(self, profiles_dir: Path) -> List[str]:
        """
        Append profiles_dir and project_dir options to dbt commands.
        """
        project_dir = self.project_dir
        commands: List[str] = []
        for command in self.commands:
            command += f" --profiles-dir {profiles_dir}"
            if project_dir is not None:
                project_dir = Path(project_dir).expanduser()
                command += f" --project-dir {project_dir}"
            commands.append(command)
        return commands

    def _compile_kwargs(self, **open_kwargs: Any) -> Dict[str, Any]:
        """
        Helper method to compile the kwargs for `open_process` so it's not repeated
        across the run and trigger methods.
        """
        profiles_dir: Path = self._find_valid_profiles_dir()
        commands: List[str] = self._append_dirs_to_commands(profiles_dir=profiles_dir)
        modified_self = self.copy()
        modified_self.commands = commands
        return super(type(self), modified_self)._compile_kwargs(**open_kwargs)


@sync_compatible
@task
async def run_dbt_build(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = "dbt-build-task-summary",
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True,
) -> Union[dbtRunnerResult, Failed]:
    """
    Executes the 'dbt build' command within a Prefect task,
    and optionally creates a Prefect artifact summarizing the dbt build results.
    """
    results = await trigger_dbt_cli_command.fn(
        command="build",
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        overwrite_profiles=overwrite_profiles,
        dbt_cli_profile=dbt_cli_profile,
        create_summary_artifact=create_summary_artifact,
        summary_artifact_key=summary_artifact_key,
        extra_command_args=extra_command_args,
        stream_output=stream_output,
    )
    return results


@sync_compatible
@task
async def run_dbt_model(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = "dbt-run-task-summary",
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True,
) -> Union[dbtRunnerResult, Failed]:
    """
    Executes the 'dbt run' command within a Prefect task,
    and optionally creates a Prefect artifact summarizing the dbt model results.
    """
    results = await trigger_dbt_cli_command.fn(
        command="run",
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        overwrite_profiles=overwrite_profiles,
        dbt_cli_profile=dbt_cli_profile,
        create_summary_artifact=create_summary_artifact,
        summary_artifact_key=summary_artifact_key,
        extra_command_args=extra_command_args,
        stream_output=stream_output,
    )
    return results


@sync_compatible
@task
async def run_dbt_test(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = "dbt-test-task-summary",
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True,
) -> Union[dbtRunnerResult, Failed]:
    """
    Executes the 'dbt test' command within a Prefect task,
    and optionally creates a Prefect artifact summarizing the dbt test results.
    """
    results = await trigger_dbt_cli_command.fn(
        command="test",
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        overwrite_profiles=overwrite_profiles,
        dbt_cli_profile=dbt_cli_profile,
        create_summary_artifact=create_summary_artifact,
        summary_artifact_key=summary_artifact_key,
        extra_command_args=extra_command_args,
        stream_output=stream_output,
    )
    return results


@sync_compatible
@task
async def run_dbt_snapshot(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = "dbt-snapshot-task-summary",
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True,
) -> Union[dbtRunnerResult, Failed]:
    """
    Executes the 'dbt snapshot' command within a Prefect task,
    and optionally creates a Prefect artifact summarizing the dbt snapshot results.
    """
    results = await trigger_dbt_cli_command.fn(
        command="snapshot",
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        overwrite_profiles=overwrite_profiles,
        dbt_cli_profile=dbt_cli_profile,
        create_summary_artifact=create_summary_artifact,
        summary_artifact_key=summary_artifact_key,
        extra_command_args=extra_command_args,
        stream_output=stream_output,
    )
    return results


@sync_compatible
@task
async def run_dbt_seed(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = "dbt-seed-task-summary",
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True,
) -> Union[dbtRunnerResult, Failed]:
    """
    Executes the 'dbt seed' command within a Prefect task,
    and optionally creates a Prefect artifact summarizing the dbt seed results.
    """
    results = await trigger_dbt_cli_command.fn(
        command="seed",
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        overwrite_profiles=overwrite_profiles,
        dbt_cli_profile=dbt_cli_profile,
        create_summary_artifact=create_summary_artifact,
        summary_artifact_key=summary_artifact_key,
        extra_command_args=extra_command_args,
        stream_output=stream_output,
    )
    return results


@sync_compatible
@task
async def run_dbt_source_freshness(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional[DbtCliProfile] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = "dbt-source-freshness-task-summary",
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True,
) -> Union[dbtRunnerResult, Failed]:
    """
    Executes the 'dbt source freshness' command within a Prefect task,
    and optionally creates a Prefect artifact summarizing the dbt source freshness results.
    """
    results = await trigger_dbt_cli_command.fn(
        command="source freshness",
        profiles_dir=profiles_dir,
        project_dir=project_dir,
        overwrite_profiles=overwrite_profiles,
        dbt_cli_profile=dbt_cli_profile,
        create_summary_artifact=create_summary_artifact,
        summary_artifact_key=summary_artifact_key,
        extra_command_args=extra_command_args,
        stream_output=stream_output,
    )
    return results


def create_summary_markdown(run_results: Dict[str, List[Any]], command: str) -> str:
    """
    Creates a Prefect task artifact summarizing the results
    of the above predefined prefrect-dbt task.
    """
    prefix = "dbt" if not command.startswith("dbt") else ""
    markdown = f"# {prefix} {command} Task Summary\n"
    markdown += _create_node_summary_table_md(run_results=run_results)
    if (
        run_results["Error"] != []
        or run_results["Fail"] != []
        or run_results["Skipped"] != []
        or (run_results["Warn"] != [])
    ):
        markdown += "\n\n ## Unsuccessful Nodes ❌\n\n"
        markdown += _create_unsuccessful_markdown(run_results=run_results)
    if run_results["Success"] != []:
        successful_runs_str = ""
        for r in run_results["Success"]:
            if isinstance(r, (RunResult, FreshnessResult)):
                successful_runs_str += f"* {r.node.name}\n"
            elif isinstance(r, RunResultOutput):
                successful_runs_str += f"* {r.unique_id}\n"
            else:
                successful_runs_str += f"* {r}\n"
        markdown += f"\n## Successful Nodes ✅\n\n{successful_runs_str}\n\n"
    return markdown


def _create_node_info_md(
    node_name: str,
    resource_type: str,
    message: str,
    path: Union[str, Path],
    compiled_code: Optional[str],
) -> str:
    """
    Creates template for unsuccessful node information
    """
    markdown = (
        f"\n**{node_name}**\n\nType: {resource_type}\n\nMessage: \n\n> {message}\n\n\nPath: {path}\n\n"
    )
    if compiled_code:
        markdown += f"\nCompiled code:\n\n```sql\n{compiled_code}\n```\n        "
    return markdown


def _create_node_summary_table_md(run_results: Dict[str, List[Any]]) -> str:
    """
    Creates a table for node summary
    """
    markdown = f"\n| Successes | Errors | Failures | Skips | Warnings |\n| :-------: | :----: | :------: | :---: | :------: |\n| {len(run_results['Success'])} |  {len(run_results['Error'])} | {len(run_results['Fail'])} | {len(run_results['Skipped'])} | {len(run_results['Warn'])} |\n    "
    return markdown


def _create_unsuccessful_markdown(run_results: Dict[str, List[Any]]) -> str:
    """
    Creates markdown summarizing the results
    of unsuccessful nodes, including compiled code.
    """
    markdown = ""
    if len(run_results["Error"]) > 0:
        markdown += "\n### Errored Nodes:\n"
        for n in run_results["Error"]:
            markdown += _create_node_info_md(
                n.node.name,
                n.node.resource_type,
                n.message,
                n.node.path,
                n.node.compiled_code
                if n.node.resource_type not in ["seed", "source"]
                else None,
            )
    if len(run_results["Fail"]) > 0:
        markdown += "\n### Failed Nodes:\n"
        for n in run_results["Fail"]:
            markdown += _create_node_info_md(
                n.node.name,
                n.node.resource_type,
                n.message,
                n.node.path,
                n.node.compiled_code
                if n.node.resource_type not in ["seed", "source"]
                else None,
            )
    if len(run_results["Skipped"]) > 0:
        markdown += "\n### Skipped Nodes:\n"
        for n in run_results["Skipped"]:
            markdown += _create_node_info_md(
                n.node.name,
                n.node.resource_type,
                n.message,
                n.node.path,
                n.node.compiled_code
                if n.node.resource_type not in ["seed", "source"]
                else None,
            )
    if len(run_results["Warn"]) > 0:
        markdown += "\n### Warned Nodes:\n"
        for n in run_results["Warn"]:
            markdown += _create_node_info_md(
                n.node.name,
                n.node.resource_type,
                n.message,
                n.node.path,
                n.node.compiled_code
                if n.node.resource_type not in ["seed", "source"]
                else None,
            )
    return markdown


def consolidate_run_results(
    results: dbtRunnerResult,
) -> Dict[str, List[Union[RunResult, RunResultOutput, FreshnessResult]]]:
    run_results: Dict[str, List[Union[RunResult, RunResultOutput, FreshnessResult]]] = {
        "Success": [],
        "Fail": [],
        "Skipped": [],
        "Error": [],
        "Warn": [],
    }
    if results.exception is None and isinstance(results.result, ExecutionResult):
        for r in results.result.results:
            if r.status == NodeStatus.Fail:
                run_results["Fail"].append(r)
            elif r.status == NodeStatus.Error or r.status == NodeStatus.RuntimeErr:
                run_results["Error"].append(r)
            elif r.status == NodeStatus.Skipped:
                run_results["Skipped"].append(r)
            elif r.status == NodeStatus.Success or r.status == NodeStatus.Pass:
                run_results["Success"].append(r)
            elif r.status == NodeStatus.Warn:
                run_results["Warn"].append(r)
    return run_results