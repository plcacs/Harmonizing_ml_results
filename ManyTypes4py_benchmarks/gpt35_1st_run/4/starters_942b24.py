from __future__ import annotations
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import warnings
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
import click
import requests
import yaml
from attrs import define, field
from packaging.version import parse
import kedro

if TYPE_CHECKING:
    from collections import OrderedDict
    from importlib_metadata import EntryPoints

TOOLS_ARG_HELP: str = "\nSelect which tools you'd like to include. By default, none are included.\n\n\nTools\n\n1) Linting: Provides a basic linting setup with Ruff\n\n2) Testing: Provides basic testing setup with pytest\n\n3) Custom Logging: Provides more logging options\n\n4) Documentation: Basic documentation setup with Sphinx\n\n5) Data Structure: Provides a directory structure for storing data\n\n6) PySpark: Provides set up configuration for working with PySpark\n\n7) Kedro Viz: Provides Kedro's native visualisation tool \n\n\nExample usage:\n\nkedro new --tools=lint,test,log,docs,data,pyspark,viz (or any subset of these options)\n\nkedro new --tools=all\n\nkedro new --tools=none\n\nFor more information on using tools, see https://docs.kedro.org/en/stable/starters/new_project_tools.html\n"
CONFIG_ARG_HELP: str = "Non-interactive mode, using a configuration yaml file. This file\nmust supply  the keys required by the template's prompts.yml. When not using a starter,\nthese are `project_name`, `repo_name` and `python_package`."
CHECKOUT_ARG_HELP: str = 'An optional tag, branch or commit to checkout in the starter repository.'
DIRECTORY_ARG_HELP: str = 'An optional directory inside the repository where the starter resides.'
NAME_ARG_HELP: str = 'The name of your new Kedro project.'
STARTER_ARG_HELP: str = 'Specify the starter template to use when creating the project.\nThis can be the path to a local directory, a URL to a remote VCS repository supported\nby `cookiecutter` or one of the aliases listed in ``kedro starter list``.\n'
EXAMPLE_ARG_HELP: str = 'Enter y to enable, n to disable the example pipeline.'
TELEMETRY_ARG_HELP: str = 'Allow or not allow Kedro to collect usage analytics.\nWe cannot see nor store information contained into a Kedro project. Opt in with "yes"\nand out with "no".\n'

@define(order=True)
class KedroStarterSpec:
    directory: Optional[str] = None
    origin: str = field(init=False)

KEDRO_PATH: Path = Path(kedro.__file__).parent
TEMPLATE_PATH: Path = KEDRO_PATH / 'templates' / 'project'

def _get_latest_starters_version() -> str:
    if 'KEDRO_STARTERS_VERSION' not in os.environ:
        GITHUB_TOKEN: Optional[str] = os.getenv('GITHUB_TOKEN')
        headers: Dict[str, str] = {}
        if GITHUB_TOKEN:
            headers['Authorization'] = f'token {GITHUB_TOKEN}'
        try:
            response = requests.get('https://api.github.com/repos/kedro-org/kedro-starters/releases/latest', headers=headers, timeout=10)
            response.raise_for_status()
            latest_release: Dict[str, Any] = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f'Error fetching kedro-starters latest release version: {e}')
            return ''
        os.environ['KEDRO_STARTERS_VERSION'] = latest_release['tag_name']
        return str(latest_release['tag_name'])
    else:
        return str(os.getenv('KEDRO_STARTERS_VERSION'))

def _kedro_version_equal_or_lower_to_starters(version: str) -> bool:
    starters_version: str = _get_latest_starters_version()
    return parse(version) <= parse(starters_version)

_STARTERS_REPO: str = 'git+https://github.com/kedro-org/kedro-starters.git'
_OFFICIAL_STARTER_SPECS: List[KedroStarterSpec] = [
    KedroStarterSpec('astro-airflow-iris', _STARTERS_REPO, 'astro-airflow-iris'),
    KedroStarterSpec('spaceflights-pandas', _STARTERS_REPO, 'spaceflights-pandas'),
    KedroStarterSpec('spaceflights-pandas-viz', _STARTERS_REPO, 'spaceflights-pandas-viz'),
    KedroStarterSpec('spaceflights-pyspark', _STARTERS_REPO, 'spaceflights-pyspark'),
    KedroStarterSpec('spaceflights-pyspark-viz', _STARTERS_REPO, 'spaceflights-pyspark-viz'),
    KedroStarterSpec('databricks-iris', _STARTERS_REPO, 'databricks-iris')
]

for starter_spec in _OFFICIAL_STARTER_SPECS:
    starter_spec.origin = 'kedro'

_OFFICIAL_STARTER_SPECS_DICT: Dict[str, KedroStarterSpec] = {spec.alias: spec for spec in _OFFICIAL_STARTER_SPECS}

TOOLS_SHORTNAME_TO_NUMBER: Dict[str, str] = {'lint': '1', 'test': '2', 'tests': '2', 'log': '3', 'logs': '3', 'docs': '4', 'doc': '4', 'data': '5', 'pyspark': '6', 'viz': '7'}
NUMBER_TO_TOOLS_NAME: Dict[str, str] = {'1': 'Linting', '2': 'Testing', '3': 'Custom Logging', '4': 'Documentation', '5': 'Data Structure', '6': 'PySpark', '7': 'Kedro Viz'}

def _validate_flag_inputs(flag_inputs: Dict[str, Optional[str]]) -> None:
    if flag_inputs.get('checkout') and (not flag_inputs.get('starter')):
        raise KedroCliError('Cannot use the --checkout flag without a --starter value.')
    if flag_inputs.get('directory') and (not flag_inputs.get('starter')):
        raise KedroCliError('Cannot use the --directory flag without a --starter value.')
    if (flag_inputs.get('tools') or flag_inputs.get('example')) and flag_inputs.get('starter'):
        raise KedroCliError('Cannot use the --starter flag with the --example and/or --tools flag.')

def _validate_input_with_regex_pattern(pattern_name: str, input: str) -> None:
    VALIDATION_PATTERNS: Dict[str, Dict[str, str]] = {
        'yes_no': {'regex': '(?i)^\\s*(y|yes|n|no)\\s*$', 'error_message': f"'{input}' is an invalid value for example pipeline. It must contain only y, n, YES, or NO (case insensitive)."},
        'project_name': {'regex': '^[\\w -]{2,}$', 'error_message': f"'{input}' is an invalid value for project name. It must contain only alphanumeric symbols, spaces, underscores and hyphens and be at least 2 characters long"},
        'tools': {'regex': '^(\n                all|none|                        # A: "all" or "none" or\n                (\\ *\\d+                          # B: any number of spaces followed by one or more digits\n                (\\ *-\\ *\\d+)?                    # C: zero or one instances of: a hyphen followed by one or more digits, spaces allowed\n                (\\ *,\\ *\\d+(\\ *-\\ *\\d+)?)*       # D: any number of instances of: a comma followed by B and C, spaces allowed\n                \\ *)?)                           # E: zero or one instances of (B,C,D) as empty strings are also permissible\n                $', 'error_message': f"'{input}' is an invalid value for project tools. Please select valid options for tools using comma-separated values, ranges, or 'all/none'."}
    }
    if not re.match(VALIDATION_PATTERNS[pattern_name]['regex'], input, flags=re.X):
        click.secho(VALIDATION_PATTERNS[pattern_name]['error_message'], fg='red', err=True)
        sys.exit(1)

def _parse_yes_no_to_bool(value: str) -> bool:
    return value.strip().lower() in ['y', 'yes'] if value is not None else False

def _validate_selected_tools(selected_tools: str) -> None:
    valid_tools: List[str] = [*list(TOOLS_SHORTNAME_TO_NUMBER), 'all', 'none']
    if selected_tools is not None:
        tools: List[str] = re.sub('\\s', '', selected_tools).split(',')
        for tool in tools:
            if tool not in valid_tools:
                click.secho('Please select from the available tools: lint, test, log, docs, data, pyspark, viz, all, none', fg='red', err=True)
                sys.exit(1)
        if ('none' in tools or 'all' in tools) and len(tools) > 1:
            click.secho("Tools options 'all' and 'none' cannot be used with other options", fg='red', err=True)
            sys.exit(1)

def _print_selection_and_prompt_info(selected_tools: str, example_pipeline: str, interactive: bool) -> None:
    if selected_tools == "['None']":
        click.secho('You have selected no project tools', fg='green')
    else:
        click.secho(f'You have selected the following project tools: {selected_tools}', fg='green')
    if example_pipeline == 'True':
        click.secho('It has been created with an example pipeline.', fg='green')
    else:
        warnings.warn("Your project does not contain any pipelines with nodes. Please ensure that at least one pipeline has been defined before executing 'kedro run'.", UserWarning)
    if interactive:
        click.secho('\nTo skip the interactive flow you can run `kedro new` with\nkedro new --name=<your-project-name> --tools=<your-project-tools> --example=<yes/no>', fg='green')

@click.group(context_settings=CONTEXT_SETTINGS, name='Kedro')
def create_cli() -> None:
    pass

@create_cli.group()
def starter() -> None:
    """Commands for working with project starters."""

@command_with_verbosity(create_cli, short_help='Create a new kedro project.')
@click.option('--config', '-c', 'config_path', type=click.Path(exists=True), help=CONFIG_ARG_HELP)
@click.option('--starter', '-s', 'starter_alias', help=STARTER_ARG_HELP)
@click.option('--checkout', help=CHECKOUT_ARG_HELP)
@click.option('--directory', help=DIRECTORY_ARG_HELP)
@click.option('--name', '-n', 'project_name', help=NAME_ARG_HELP)
@click.option('--tools', '-t', 'selected_tools', help=TOOLS_ARG_HELP)
@click.option('--example', '-e', 'example_pipeline', help=EXAMPLE_ARG_HELP)
@click.option('--telemetry', '-tc', 'telemetry_consent', help=TELEMETRY_ARG_HELP, type=click.Choice(['yes', 'no', 'y', 'n'], case_sensitive=False))
def new(config_path: str, starter_alias: str, selected_tools: str, project_name: str, checkout: str, directory: str, example_pipeline: str, telemetry_consent: str, **kwargs: Any) -> None:
    """Create a new kedro project."""
    flag_inputs: Dict[str, Optional[str]] = {'config': config_path, 'starter': starter_alias, 'tools': selected_tools, 'name': project_name, 'checkout': checkout, 'directory': directory, 'example': example_pipeline, 'telemetry_consent': telemetry_consent}
    _validate_flag_inputs(flag_inputs)
    starters_dict: Dict[str, KedroStarterSpec] = _get_starters_dict()
    if starter_alias in starters_dict:
        if directory:
            raise KedroCliError('Cannot use the --directory flag with a --starter alias.')
        spec = starters_dict[starter_alias]
        template_path = spec.template_path
        directory = spec.directory
        checkout = _select_checkout_branch_for_cookiecutter(checkout)
    elif starter_alias is not None:
        template_path = starter_alias
    else:
        template_path = str(TEMPLATE_PATH)
    if selected_tools is not None:
        selected_tools = selected_tools.lower()
    tmpdir = tempfile.mkdtemp()
    cookiecutter_dir = _get_cookiecutter_dir(template_path, checkout, directory, tmpdir)
    prompts_required = _get_prompts_required_and_clear_from_CLI_provided(cookiecutter_dir, selected_tools, project_name, example_pipeline)
    cookiecutter_context = None
    if not config_path:
        cookiecutter_context = _make_cookiecutter_context_for_prompts(cookiecutter_dir)
    shutil.rmtree(tmpdir, onerror=_remove_readonly)
    extra_context = _get_extra_context(prompts_required=prompts_required, config_path=config_path, cookiecutter_context=cookiecutter_context, selected_tools=selected_tools, project_name=project_name, example_pipeline=example_pipeline, starter_alias=starter_alias)
    cookiecutter_args, project_template = _make_cookiecutter_args_and_fetch_template(config=extra_context, checkout=checkout, directory=directory, template_path=template_path)
    if telemetry_consent is not None:
        telemetry_consent = 'true' if _parse_yes_no_to_bool(telemetry_consent) else 'false'
    _create_project(project_template, cookiecutter_args, telemetry_consent)
    if not starter_alias:
        interactive_flow = prompts_required and (not config_path)
        _print_selection_and_prompt_info(extra_context['tools'], extra_context['example_pipeline'], interactive_flow)

@starter.command('list')
def list_starters() -> None:
    """List all official project starters available."""
    starters_dict: Dict[str, KedroStarterSpec] = _get_starters_dict()
    sorted_starters_dict: Dict[str, Dict[str, KedroStarterSpec]] = {origin: dict(sorted(starters_dict_by_origin)) for origin, starters_dict_by_origin in groupby(starters_dict.items(), lambda item: item[1].origin)}
    sorted_starters_dict = dict(sorted(sorted_starters_dict.items(), key=lambda x: x == 'kedro'))
    for origin, starters_spec in sorted_starters_dict.items():
        click.secho(f'\nStarters from {origin}\n', fg='yellow')
        click.echo(yaml.safe_dump(_starter_spec_to_dict(starters_spec), sort_keys=False))

def _get_cookiecutter_dir(template_path: str, checkout: str, directory: str, tmpdir: str) -> Path:
    """Gives a path to the cookiecutter directory. If template_path is a repo then
    clones it to ``tmpdir``; if template_path is a file path then directly uses that
    path without copying anything.
    """
    from cookiecutter.exceptions import RepositoryCloneFailed, RepositoryNotFound
    try:
        cookiecutter_dir, _ = determine_repo_dir(template=template_path, abbreviations={}, clone_to_dir=Path(tmpdir).resolve(), checkout=checkout, no_input=True, directory=directory)
    except (RepositoryNotFound, RepositoryCloneFailed) as exc:
        error_message = f'Kedro project template not found at {template_path}.'
        if checkout:
            error_message += f' Specified tag {checkout}. The following tags are available: ' + ', '.join(_get_available_tags(template_path))
        official_starters: List[str] = sorted(_OFFICIAL_STARTER_SPECS_DICT)
        raise KedroCliError(f'{error_message}. The aliases for the official Kedro starters are: \n{yaml.safe_dump(official_starters, sort_keys=False)}') from exc
    return Path(cookiecutter_dir)

def _get_prompts_required_and_clear_from_CLI_provided(cookiecutter_dir: Path, selected_tools: str, project_name: str, example_pipeline: str) -> Dict[str, Any]:
    """Finds the information a user must supply according to prompts.yml,
    and clear it from what has already been provided via the CLI(validate it before)"""
    prompts_yml: Path = cookiecutter_dir / 'prompts.yml'
    if not prompts_yml.is_file():
        return {}
    try:
        with prompts_yml.open('r') as prompts_file:
            prompts_required: Dict[str, Any] = yaml.safe_load(prompts_file)
    except Exception as exc:
        raise KedroCliError('Failed to generate project: could not load prompts.yml.') from exc
    if selected_tools is not None:
        _validate_selected_tools(selected_tools)
        del prompts_required['tools']
    if project_name is not None:
        _validate_input_with_regex_pattern('project_name', project_name)
        del prompts_required['project_name']
    if example_pipeline is not None:
        _validate_input_with_regex_pattern('yes_no', example_pipeline)
        del prompts_required['example_pipeline']
    return prompts_required

def _get_available_tags(template_path: str) -> List[str]:
    import git
    try:
        tags: str = git.cmd.Git().ls_remote('--tags', template_path.replace('git+', ''))
        unique_tags: Set[str] = {tag.split('/')[-1].replace('^{}', '') for tag in tags.split('\n')}
    except git.GitCommandError:
        return []
    return sorted(unique_tags)

def _get_starters_dict() -> Dict[str, KedroStarterSpec]:
    """This function lists all the starter aliases declared in
    the core repo and in plugins entry points.

    For example, the output for official kedro starters looks like:
    {"astro-airflow-iris":
        KedroStarterSpec(
            name="astro-airflow-iris",
            template_path="git+https://github.com/kedro-org/kedro-starters.git",
            directory="astro-airflow-iris",
            origin="kedro"
        ),
    }
    """
    starter_specs: Dict[str, KedroStarterSpec] = _OFFICIAL_STARTER_SPECS_DICT
    for starter_entry_point in _get_entry_points(name='starters'):
        origin: str = starter_entry_point.module.split('.')[0]
        specs: List[KedroStarterSpec] = _safe_load_entry_point(starter_entry_point) or []
        for spec in specs:
            if not isinstance(spec, KedroStarterSpec):
                click.secho(f"The starter configuration loaded from module {origin}should be a 'KedroStarterSpec', got '{type(spec)}' instead", fg='red')
            elif spec.alias in starter_specs:
                click.secho(f'Starter alias `{spec.alias}` from `{origin}` has been ignored as it is already defined by`{starter_specs[spec.alias].origin}`', fg='red')
            else:
                spec.origin = origin
                starter_specs[spec.alias] = spec