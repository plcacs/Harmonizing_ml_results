from pathlib import Path
from typing import List, Optional
import click
import pytest
from dbt.cli.exceptions import DbtUsageException
from dbt.cli.flags import Flags
from dbt.cli.main import cli
from dbt.cli.types import Command
from dbt.contracts.project import ProjectFlags
from dbt.tests.util import rm_file, write_file
from dbt_common.exceptions import DbtInternalError
from dbt_common.helper_types import WarnErrorOptions

class TestFlags:

    def make_dbt_context(self, context_name: str, args: List[str], parent=None) -> click.Context:
    
    def run_context(self) -> click.Context:
    
    def project_flags(self) -> ProjectFlags:
    
    def test_cli_args_unmodified(self) -> None:
    
    def test_which(self, run_context: click.Context) -> None:
    
    def test_cli_group_flags_from_params(self, run_context: click.Context, param: click.Parameter) -> None:
    
    def test_log_path_default(self, run_context: click.Context) -> None:
    
    def test_log_file_max_size_default(self, run_context: click.Context) -> None:
    
    def test_anonymous_usage_state(self, monkeypatch, run_context: click.Context, set_stats_param, do_not_track, expected_anonymous_usage_stats) -> None:
    
    def test_resource_types(self, monkeypatch) -> None:
    
    def test_empty_project_flags_uses_default(self, run_context: click.Context, project_flags: ProjectFlags) -> None:
    
    def test_none_project_flags_uses_default(self, run_context: click.Context) -> None:
    
    def test_prefer_project_flags_to_default(self, run_context: click.Context, project_flags: ProjectFlags) -> None:
    
    def test_prefer_param_value_to_project_flags(self) -> None:
    
    def test_prefer_env_to_project_flags(self, monkeypatch, project_flags: ProjectFlags) -> None:
    
    def test_mutually_exclusive_options_passed_separately(self) -> None:
    
    def test_mutually_exclusive_options_from_cli(self) -> None:
    
    def test_mutually_exclusive_options_from_project_flags(self, warn_error, project_flags: ProjectFlags) -> None:
    
    def test_mutually_exclusive_options_from_envvar(self, warn_error, monkeypatch) -> None:
    
    def test_mutually_exclusive_options_from_cli_and_project_flags(self, warn_error, project_flags: ProjectFlags) -> None:
    
    def test_mutually_exclusive_options_from_cli_and_envvar(self, warn_error, monkeypatch) -> None:
    
    def test_mutually_exclusive_options_from_project_flags_and_envvar(self, project_flags: ProjectFlags, warn_error, monkeypatch) -> None:
    
    def test_no_color_interaction(self, cli_colors, cli_colors_file, flag_colors, flag_colors_file) -> None:
    
    def test_log_level_interaction(self, cli_log_level, cli_log_level_file, flag_log_level, flag_log_level_file) -> None:
    
    def test_log_format_interaction(self, cli_log_format, cli_log_format_file, flag_log_format, flag_log_format_file) -> None:
    
    def test_log_settings_from_config(self) -> None:
    
    def test_log_file_settings_from_config(self) -> None:
    
    def test_duplicate_flags_raises_error(self) -> None:
    
    def test_global_flag_at_child_context(self) -> None:
    
    def test_global_flag_with_env_var(self, monkeypatch) -> None:
    
    def test_set_project_only_flags(self, project_flags: ProjectFlags, run_context: click.Context) -> None:
    
    def _create_flags_from_dict(self, cmd: Command, d: dict) -> Flags:
    
    def test_from_dict__run(self) -> None:
    
    def test_from_dict__build(self) -> None:
    
    def test_from_dict__seed(self) -> None:
    
    def test_from_dict__which_fails(self) -> None:
    
    def test_from_dict_0_value(self) -> None:

def test_project_flag_defaults() -> None:
