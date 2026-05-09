@task
@sync_compatible
async def trigger_dbt_cli_command(
    command: str, 
    profiles_dir: str = None, 
    project_dir: str = None, 
    overwrite_profiles: bool = False, 
    dbt_cli_profile: DbtCliProfile = None, 
    create_summary_artifact: bool = False, 
    summary_artifact_key: str = 'dbt-cli-command-summary', 
    extra_command_args: List[str] = None, 
    stream_output: bool = True
) -> Union[str, List[str]]:
    ...

class DbtCoreOperation(ShellOperation):
    profiles_dir: str = Field(default=None, description="The directory to search for the profiles.yml file. Setting this appends the `--profiles-dir` option to the dbt commands provided. If this is not set, will try using the DBT_PROFILES_DIR environment variable, but if that's also not set, will use the default directory `$HOME/.dbt/`.")
    project_dir: str = Field(default=None, description='The directory to search for the dbt_project.yml file. Default is the current working directory and its parents.')
    overwrite_profiles: bool = Field(default=False, description='Whether the existing profiles.yml file under profiles_dir should be overwritten with a new profile.')
    dbt_cli_profile: DbtCliProfile = Field(default=None, description='Profiles class containing the profile written to profiles.yml. Note! This is optional and will raise an error if profiles.yml already exists under profile_dir and overwrite_profiles is set to False.')

    def _find_valid_profiles_dir(self) -> str:
        ...

    def _append_dirs_to_commands(self, profiles_dir: str) -> List[str]:
        ...

    def _compile_kwargs(self, **open_kwargs: Any) -> Dict[str, Any]:
        ...

@sync_compatible
@task
async def run_dbt_build(
    profiles_dir: str = None, 
    project_dir: str = None, 
    overwrite_profiles: bool = False, 
    dbt_cli_profile: DbtCliProfile = None, 
    create_summary_artifact: bool = False, 
    summary_artifact_key: str = 'dbt-build-task-summary', 
    extra_command_args: List[str] = None, 
    stream_output: bool = True
) -> Union[str, List[str]]:
    ...

@sync_compatible
@task
async def run_dbt_model(
    profiles_dir: str = None, 
    project_dir: str = None, 
    overwrite_profiles: bool = False, 
    dbt_cli_profile: DbtCliProfile = None, 
    create_summary_artifact: bool = False, 
    summary_artifact_key: str = 'dbt-run-task-summary', 
    extra_command_args: List[str] = None, 
    stream_output: bool = True
) -> Union[str, List[str]]:
    ...

@sync_compatible
@task
async def run_dbt_test(
    profiles_dir: str = None, 
    project_dir: str = None, 
    overwrite_profiles: bool = False, 
    dbt_cli_profile: DbtCliProfile = None, 
    create_summary_artifact: bool = False, 
    summary_artifact_key: str = 'dbt-test-task-summary', 
    extra_command_args: List[str] = None, 
    stream_output: bool = True
) -> Union[str, List[str]]:
    ...

@sync_compatible
@task
async def run_dbt_snapshot(
    profiles_dir: str = None, 
    project_dir: str = None, 
    overwrite_profiles: bool = False, 
    dbt_cli_profile: DbtCliProfile = None, 
    create_summary_artifact: bool = False, 
    summary_artifact_key: str = 'dbt-snapshot-task-summary', 
    extra_command_args: List[str] = None, 
    stream_output: bool = True
) -> Union[str, List[str]]:
    ...

@sync_compatible
@task
async def run_dbt_seed(
    profiles_dir: str = None, 
    project_dir: str = None, 
    overwrite_profiles: bool = False, 
    dbt_cli_profile: DbtCliProfile = None, 
    create_summary_artifact: bool = False, 
    summary_artifact_key: str = 'dbt-seed-task-summary', 
    extra_command_args: List[str] = None, 
    stream_output: bool = True
) -> Union[str, List[str]]:
    ...

@sync_compatible
@task
async def run_dbt_source_freshness(
    profiles_dir: str = None, 
    project_dir: str = None, 
    overwrite_profiles: bool = False, 
    dbt_cli_profile: DbtCliProfile = None, 
    create_summary_artifact: bool = False, 
    summary_artifact_key: str = 'dbt-source-freshness-task-summary', 
    extra_command_args: List[str] = None, 
    stream_output: bool = True
) -> Union[str, List[str]]:
    ...

def create_summary_markdown(run_results: Dict[str, List[Any]], command: str) -> str:
    ...

def _create_node_info_md(node_name: str, resource_type: str, message: str, path: str, compiled_code: str) -> str:
    ...

def _create_node_summary_table_md(run_results: Dict[str, List[Any]]) -> str:
    ...

def _create_unsuccessful_markdown(run_results: Dict[str, List[Any]]) -> str:
    ...

def consolidate_run_results(results: dbtRunnerResult) -> Dict[str, List[Any]]:
    ...
