from pathlib import Path
from typing import Any, Dict, List, Optional, Union

async def trigger_dbt_cli_command(command: str, profiles_dir: Optional[str] = None, project_dir: Optional[str] = None, overwrite_profiles: bool = False, dbt_cli_profile: Optional[Any] = None, create_summary_artifact: bool = False, summary_artifact_key: str = 'dbt-cli-command-summary', extra_command_args: Optional[List[str]] = None, stream_output: bool = True) -> Union[str, List[str]]:
    ...

class DbtCoreOperation(ShellOperation):
    commands: List[str]
    stream_output: bool
    env: Dict[str, str]
    working_dir: str
    shell: str
    extension: str
    profiles_dir: Optional[str]
    project_dir: Optional[str]
    overwrite_profiles: bool
    dbt_cli_profile: Optional[Any]

    def _find_valid_profiles_dir(self) -> str:
        ...

    def _append_dirs_to_commands(self, profiles_dir: str) -> List[str]:
        ...

    def _compile_kwargs(self, **open_kwargs) -> Dict[str, Any]:
        ...

async def run_dbt_build(profiles_dir: Optional[str] = None, project_dir: Optional[str] = None, overwrite_profiles: bool = False, dbt_cli_profile: Optional[Any] = None, create_summary_artifact: bool = False, summary_artifact_key: str = 'dbt-build-task-summary', extra_command_args: Optional[List[str]] = None, stream_output: bool = True) -> Any:
    ...

async def run_dbt_model(profiles_dir: Optional[str] = None, project_dir: Optional[str] = None, overwrite_profiles: bool = False, dbt_cli_profile: Optional[Any] = None, create_summary_artifact: bool = False, summary_artifact_key: str = 'dbt-run-task-summary', extra_command_args: Optional[List[str]] = None, stream_output: bool = True) -> Any:
    ...

async def run_dbt_test(profiles_dir: Optional[str] = None, project_dir: Optional[str] = None, overwrite_profiles: bool = False, dbt_cli_profile: Optional[Any] = None, create_summary_artifact: bool = False, summary_artifact_key: str = 'dbt-test-task-summary', extra_command_args: Optional[List[str]] = None, stream_output: bool = True) -> Any:
    ...

async def run_dbt_snapshot(profiles_dir: Optional[str] = None, project_dir: Optional[str] = None, overwrite_profiles: bool = False, dbt_cli_profile: Optional[Any] = None, create_summary_artifact: bool = False, summary_artifact_key: str = 'dbt-snapshot-task-summary', extra_command_args: Optional[List[str]] = None, stream_output: bool = True) -> Any:
    ...

async def run_dbt_seed(profiles_dir: Optional[str] = None, project_dir: Optional[str] = None, overwrite_profiles: bool = False, dbt_cli_profile: Optional[Any] = None, create_summary_artifact: bool = False, summary_artifact_key: str = 'dbt-seed-task-summary', extra_command_args: Optional[List[str]] = None, stream_output: bool = True) -> Any:
    ...

async def run_dbt_source_freshness(profiles_dir: Optional[str] = None, project_dir: Optional[str] = None, overwrite_profiles: bool = False, dbt_cli_profile: Optional[Any] = None, create_summary_artifact: bool = False, summary_artifact_key: str = 'dbt-source-freshness-task-summary', extra_command_args: Optional[List[str]] = None, stream_output: bool = True) -> Any:
    ...

def create_summary_markdown(run_results: Dict[str, List[Any]], command: str) -> str:
    ...

def _create_node_info_md(node_name: str, resource_type: str, message: str, path: str, compiled_code: Optional[str]) -> str:
    ...

def _create_node_summary_table_md(run_results: Dict[str, List[Any]]) -> str:
    ...

def _create_unsuccessful_markdown(run_results: Dict[str, List[Any]]) -> str:
    ...

def consolidate_run_results(results: Any) -> Dict[str, List[Any]]:
    ...
