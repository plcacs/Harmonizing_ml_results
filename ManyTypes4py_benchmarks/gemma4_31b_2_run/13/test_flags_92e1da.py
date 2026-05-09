from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple
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

    def make_dbt_context(self, context_name: str, args: List[str], parent: Any = None) -> Any:
        ctx = cli.make_context(context_name, args.copy(), parent)
        return ctx

    @pytest.fixture(scope='class')
    def run_context(self) -> Any:
        return self.make_dbt_context('run', ['run'])

    @pytest.fixture
    def project_flags(self) -> ProjectFlags:
        return ProjectFlags()

    def test_cli_args_unmodified(self) -> None:
        args = ['--target', 'my_target']
        args_before = args.copy()
        self.make_dbt_context('context', args)
        assert args == args_before

    def test_which(self, run_context: Any) -> None:
        flags = Flags(run_context)
        assert flags.WHICH == 'run'

    @pytest.mark.parametrize('param', cli.params)
    def test_cli_group_flags_from_params(self, run_context: Any, param: Any) -> None:
        flags = Flags(run_context)
        if 'DEPRECATED_' in param.name.upper():
            assert not hasattr(flags, param.name.upper())
            return
        if param.name.upper() in ('VERSION', 'LOG_PATH'):
            return
        assert hasattr(flags, param.name.upper())
        assert getattr(flags, param.name.upper()) == run_context.params[param.name.lower()]

    def test_log_path_default(self, run_context: Any) -> None:
        flags = Flags(run_context)
        assert hasattr(flags, 'LOG_PATH')
        assert getattr(flags, 'LOG_PATH') == Path('logs')

    def test_log_file_max_size_default(self, run_context: Any) -> None:
        flags = Flags(run_context)
        assert hasattr(flags, 'LOG_FILE_MAX_BYTES')
        assert getattr(flags, 'LOG_FILE_MAX_BYTES') == 10 * 1024 * 1024

    @pytest.mark.parametrize('set_stats_param,do_not_track,expected_anonymous_usage_stats', [('default', '1', False), ('default', 't', False), ('default', 'true', False), ('default', 'y', False), ('default', 'yes', False), ('default', 'false', True), ('default', 'anything', True), (True, '1', False), (True, 't', False), (True, 'true', False), (True, 'y', False), (True, 'yes', False), (True, 'false', True), (True, 'anything', True), (True, '2', True), (False, '1', False), (False, 't', False), (False, 'true', False), (False, 'y', False), (False, 'yes', False), (False, 'false', False), (False, 'anything', False), (False, '2', False)])
    def test_anonymous_usage_state(self, monkeypatch: Any, run_context: Any, set_stats_param: Any, do_not_track: str, expected_anonymous_usage_stats: bool) -> None:
        monkeypatch.setenv('DO_NOT_TRACK', do_not_track)
        if set_stats_param != 'default':
            run_context.params['send_anonymous_usage_stats'] = set_stats_param
        flags = Flags(run_context)
        assert flags.SEND_ANONYMOUS_USAGE_STATS == expected_anonymous_usage_stats

    def test_resource_types(self, monkeypatch: Any) -> None:
        monkeypatch.setenv('DBT_RESOURCE_TYPES', 'model')
        build_context = self.make_dbt_context('build', ['build'])
        build_context.params['resource_types'] = ('unit_test',)
        flags = Flags(build_context)
        assert flags.resource_types == ('unit_test',)

    def test_empty_project_flags_uses_default(self, run_context: Any, project_flags: ProjectFlags) -> None:
        flags = Flags(run_context, project_flags)
        assert flags.USE_COLORS == run_context.params['use_colors']

    def test_none_project_flags_uses_default(self, run_context: Any) -> None:
        flags = Flags(run_context, None)
        assert flags.USE_COLORS == run_context.params['use_colors']

    def test_prefer_project_flags_to_default(self, run_context: Any, project_flags: ProjectFlags) -> None:
        project_flags.use_colors = False
        assert run_context.params['use_colors'] is not project_flags.use_colors
        flags = Flags(run_context, project_flags)
        assert flags.USE_COLORS == project_flags.use_colors

    def test_prefer_param_value_to_project_flags(self) -> None:
        project_flags = ProjectFlags(use_colors=False)
        context = self.make_dbt_context('run', ['--use-colors', 'True', 'run'])
        flags = Flags(context, project_flags)
        assert flags.USE_COLORS

    def test_prefer_env_to_project_flags(self, monkeypatch: Any, project_flags: ProjectFlags) -> None:
        project_flags.use_colors = False
        monkeypatch.setenv('DBT_USE_COLORS', 'True')
        context = self.make_dbt_context('run', ['run'])
        flags = Flags(context, project_flags)
        assert flags.USE_COLORS

    def test_mutually_exclusive_options_passed_separately(self) -> None:
        """Assert options that are mutually exclusive can be passed separately without error"""
        warn_error_context = self.make_dbt_context('run', ['--warn-error', 'run'])
        flags = Flags(warn_error_context)
        assert flags.WARN_ERROR
        warn_error_options_context = self.make_dbt_context('run', ['--warn-error-options', '{"include": "all"}', 'run'])
        flags = Flags(warn_error_options_context)
        assert flags.WARN_ERROR_OPTIONS == WarnErrorOptions(include='all')

    def test_mutually_exclusive_options_from_cli(self) -> None:
        context = self.make_dbt_context('run', ['--warn-error', '--warn-error-options', '{"include": "all"}', 'run'])
        with pytest.raises(DbtUsageException):
            Flags(context)

    @pytest.mark.parametrize('warn_error', [True, False])
    def test_mutually_exclusive_options_from_project_flags(self, warn_error: bool, project_flags: ProjectFlags) -> None:
        project_flags.warn_error = warn_error
        context = self.make_dbt_context('run', ['--warn-error-options', '{"include": "all"}', 'run'])
        with pytest.raises(DbtUsageException):
            Flags(context, project_flags)

    @pytest.mark.parametrize('warn_error', ['True', 'False'])
    def test_mutually_exclusive_options_from_envvar(self, warn_error: str, monkeypatch: Any) -> None:
        monkeypatch.setenv('DBT_WARN_ERROR', warn_error)
        monkeypatch.setenv('DBT_WARN_ERROR_OPTIONS', '{"include":"all"}')
        context = self.make_dbt_context('run', ['run'])
        with pytest.raises(DbtUsageException):
            Flags(context)

    @pytest.mark.parametrize('warn_error', [True, False])
    def test_mutually_exclusive_options_from_cli_and_project_flags(self, warn_error: bool, project_flags: ProjectFlags) -> None:
        project_flags.warn_error = warn_error
        context = self.make_dbt_context('run', ['--warn-error-options', '{"include": "all"}', 'run'])
        with pytest.raises(DbtUsageException):
            Flags(context, project_flags)

    @pytest.mark.parametrize('warn_error', ['True', 'False'])
    def test_mutually_exclusive_options_from_cli_and_envvar(self, warn_error: str, monkeypatch: Any) -> None:
        monkeypatch.setenv('DBT_WARN_ERROR', warn_error)
        context = self.make_dbt_context('run', ['--warn-error-options', '{"include": "all"}', 'run'])
        with pytest.raises(DbtUsageException):
            Flags(context)

    @pytest.mark.parametrize('warn_error', ['True', 'False'])
    def test_mutually_exclusive_options_from_project_flags_and_envvar(self, project_flags: ProjectFlags, warn_error: str, monkeypatch: Any) -> None:
        project_flags.warn_error = warn_error
        monkeypatch.setenv('DBT_WARN_ERROR_OPTIONS', '{"include": "all"}')
        context = self.make_dbt_context('run', ['run'])
        with pytest.raises(DbtUsageException):
            Flags(context, project_flags)

    @pytest.mark.parametrize('cli_colors,cli_colors_file,flag_colors,flag_colors_file', [(None, None, True, True), (True, None, True, True), (None, True, True, True), (False, None, False, False), (None, False, True, False), (True, True, True, True), (False, False, False, False), (True, False, True, False), (False, True, False, True)])
    def test_no_color_interaction(self, cli_colors: Optional[bool], cli_colors_file: Optional[bool], flag_colors: bool, flag_colors_file: bool) -> None:
        cli_params = []
        if cli_colors is not None:
            cli_params.append('--use-colors' if cli_colors else '--no-use-colors')
        if cli_colors_file is not None:
            cli_params.append('--use-colors-file' if cli_colors_file else '--no-use-colors-file')
        cli_params.append('run')
        context = self.make_dbt_context('run', cli_params)
        flags = Flags(context, None)
        assert flags.USE_COLORS == flag_colors
        assert flags.USE_COLORS_FILE == flag_colors_file

    @pytest.mark.parametrize('cli_log_level,cli_log_level_file,flag_log_level,flag_log_level_file', [(None, None, 'info', 'debug'), ('error', None, 'error', 'error'), ('info', None, 'info', 'info'), ('debug', 'warn', 'debug', 'warn')])
    def test_log_level_interaction(self, cli_log_level: Optional[str], cli_log_level_file: Optional[str], flag_log_level: str, flag_log_level_file: str) -> None:
        cli_params = []
        if cli_log_level is not None:
            cli_params.append('--log-level')
            cli_params.append(cli_log_level)
        if cli_log_level_file is not None:
            cli_params.append('--log-level-file')
            cli_params.append(cli_log_level_file)
        cli_params.append('run')
        context = self.make_dbt_context('run', cli_params)
        flags = Flags(context, None)
        assert flags.LOG_LEVEL == flag_log_level
        assert flags.LOG_LEVEL_FILE == flag_log_level_file

    @pytest.mark.parametrize('cli_log_format,cli_log_format_file,flag_log_format,flag_log_format_file', [(None, None, 'default', 'debug'), ('json', None, 'json', 'json'), (None, 'json', 'default', 'json'), ('debug', 'text', 'debug', 'text')])
    def test_log_format_interaction(self, cli_log_format: Optional[str], cli_log_format_file: Optional[str], flag_log_format: str, flag_log_format_file: str) -> None:
        cli_params = []
        if cli_log_format is not None:
            cli_params.append('--log-format')
            cli_params.append(cli_log_format)
        if cli_log_format_file is not None:
            cli_params.append('--log-format-file')
            cli_params.append(cli_log_format_file)
        cli_params.append('run')
        context = self.make_dbt_context('run', cli_params)
        flags = Flags(context, None)
        assert flags.LOG_FORMAT == flag_log_format
        assert flags.LOG_FORMAT_FILE == flag_log_format_file

    def test_log_settings_from_config(self) -> None:
        """Test that values set in ProjectFlags for log settings will set flags as expected"""
        context = self.make_dbt_context('run', ['run'])
        config = ProjectFlags(log_format='json', log_level='warn', use_colors=False)
        flags = Flags(context, config)
        assert flags.LOG_FORMAT == 'json'
        assert flags.LOG_FORMAT_FILE == 'json'
        assert flags.LOG_LEVEL == 'warn'
        assert flags.LOG_LEVEL_FILE == 'warn'
        assert flags.USE_COLORS is False
        assert flags.USE_COLORS_FILE is False

    def test_log_file_settings_from_config(self) -> None:
        """Test that values set in ProjectFlags for log *file* settings will set flags as expected, leaving the console
        logging flags with their default values"""
        context = self.make_dbt_context('run', ['run'])
        config = ProjectFlags(log_format_file='json', log_level_file='warn', use_colors_file=False)
        flags = Flags(context, config)
        assert flags.LOG_FORMAT == 'default'
        assert flags.LOG_FORMAT_FILE == 'json'
        assert flags.LOG_LEVEL == 'info'
        assert flags.LOG_LEVEL_FILE == 'warn'
        assert flags.USE_COLORS is True
        assert flags.USE_COLORS_FILE is False

    def test_duplicate_flags_raises_error(self) -> None:
        parent_context = self.make_dbt_context('parent', ['--version-check'])
        context = self.make_dbt_context('child', ['--version-check'], parent_context)
        with pytest.raises(DbtUsageException):
            Flags(context)

    def test_global_flag_at_child_context(self) -> None:
        parent_context_a = self.make_dbt_context('parent_context_a', ['--no-use-colors'])
        child_context_a = self.make_dbt_context('child_context_a', ['run'], parent_context_a)
        flags_a = Flags(child_context_a)
        parent_context_b = self.make_dbt_context('parent_context_b', ['run'])
        child_context_b = self.make_dbt_context('child_context_b', ['--no-use-colors'], parent_context_b)
        flags_b = Flags(child_context_b)
        assert flags_a.USE_COLORS == flags_b.USE_COLORS

    def test_global_flag_with_env_var(self, monkeypatch: Any) -> None:
        monkeypatch.setenv('DBT_QUIET', '0')
        parent_context = self.make_dbt_context('parent', ['--no-use-colors'])
        child_context = self.make_dbt_context('child', ['--quiet'], parent_context)
        flags = Flags(child_context)
        assert flags.QUIET is True
        parent_context = self.make_dbt_context('parent', ['--quiet'])
        child_context = self.make_dbt_context('child', ['--no-use-colors'], parent_context)
        flags = Flags(child_context)
        assert flags.QUI