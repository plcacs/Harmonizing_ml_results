from typing import Any, Dict, List, Optional, Union

@task
@sync_compatible
async def trigger_dbt_cli_command(
    command: str,
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional['DbtCliProfile'] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-cli-command-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Any:
    # ...

@sync_compatible
@task
async def run_dbt_build(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional['DbtCliProfile'] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-build-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Any:
    # ...

@sync_compatible
@task
async def run_dbt_model(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional['DbtCliProfile'] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-run-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Any:
    # ...

@sync_compatible
@task
async def run_dbt_test(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional['DbtCliProfile'] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-test-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Any:
    # ...

@sync_compatible
@task
async def run_dbt_snapshot(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional['DbtCliProfile'] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-snapshot-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Any:
    # ...

@sync_compatible
@task
async def run_dbt_seed(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional['DbtCliProfile'] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-seed-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Any:
    # ...

@sync_compatible
@task
async def run_dbt_source_freshness(
    profiles_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    overwrite_profiles: bool = False,
    dbt_cli_profile: Optional['DbtCliProfile'] = None,
    create_summary_artifact: bool = False,
    summary_artifact_key: str = 'dbt-source-freshness-task-summary',
    extra_command_args: Optional[List[str]] = None,
    stream_output: bool = True
) -> Any:
    # ...

def create_summary_markdown(
    run_results: Dict[str, Any],
    command: str
) -> str:
    # ...

def consolidate_run_results(
    results: 'dbtRunnerResult'
) -> Dict[str, Any]:
    # ...

class DbtCoreOperation(ShellOperation):
    # ...
