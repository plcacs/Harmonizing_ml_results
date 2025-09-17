#!/usr/bin/env python
"""kedro is a CLI for managing Kedro projects.

This module implements commands available from the kedro CLI for creating
projects.
"""
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import click
import requests
import yaml
from attrs import define, field
from packaging.version import parse

import kedro
from kedro import __version__ as version
from kedro.framework.cli.utils import (
    CONTEXT_SETTINGS,
    KedroCliError,
    _clean_pycache,
    _get_entry_points,
    _safe_load_entry_point,
    command_with_verbosity,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from importlib_metadata import EntryPoints

TOOLS_ARG_HELP: str = (
    "\nSelect which tools you'd like to include. By default, none are included.\n\n\nTools\n\n1) Linting: Provides a basic linting setup with Ruff\n\n2) Testing: Provides basic testing setup with pytest\n\n3) Custom Logging: Provides more logging options\n\n4) Documentation: Basic documentation setup with Sphinx\n\n5) Data Structure: Provides a directory structure for storing data\n\n6) PySpark: Provides set up configuration for working with PySpark\n\n7) Kedro Viz: Provides Kedro's native visualisation tool \n\n\nExample usage:\n\nkedro new --tools=lint,test,log,docs,data,pyspark,viz (or any subset of these options)\n\nkedro new --tools=all\n\nkedro new --tools=none\n\nFor more information on using tools, see https://docs.kedro.org/en/stable/starters/new_project_tools.html\n"
)
CONFIG_ARG_HELP: str = (
    "Non-interactive mode, using a configuration yaml file. This file\nmust supply  the keys required by the template's prompts.yml. When not using a starter,\nthese are `project_name`, `repo_name` and `python_package`."
)
CHECKOUT_ARG_HELP: str = "An optional tag, branch or commit to checkout in the starter repository."
DIRECTORY_ARG_HELP: str = "An optional directory inside the repository where the starter resides."
NAME_ARG_HELP: str = "The name of your new Kedro project."
STARTER_ARG_HELP: str = (
    "Specify the starter template to use when creating the project.\nThis can be the path to a local directory, a URL to a remote VCS repository supported\nby `cookiecutter` or one of the aliases listed in ``kedro starter list``.\n"
)
EXAMPLE_ARG_HELP: str = "Enter y to enable, n to disable the example pipeline."
TELEMETRY_ARG_HELP: str = (
    "Allow or not allow Kedro to collect usage analytics.\nWe cannot see nor store information contained into a Kedro project. Opt in with \"yes\"\nand out with \"no\".\n"
)


@define(order=True)
class KedroStarterSpec:
    """Specification of custom kedro starter template

    Args:
        alias: alias of the starter which shows up on `kedro starter list` and is used
            by the starter argument of `kedro new`
        template_path: path to a directory or a URL to a remote VCS repository supported
            by `cookiecutter`
        directory: optional directory inside the repository where the starter resides.
        origin: reserved field used by kedro internally to determine where the starter
            comes from, users do not need to provide this field.
    """
    alias: str
    template_path: Union[str, Path]
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
            response = requests.get(
                'https://api.github.com/repos/kedro-org/kedro-starters/releases/latest',
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            latest_release = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f'Error fetching kedro-starters latest release version: {e}')
            return ''
        os.environ['KEDRO_STARTERS_VERSION'] = latest_release['tag_name']
        return str(latest_release['tag_name'])
    else:
        return str(os.getenv('KEDRO_STARTERS_VERSION'))


def _kedro_version_equal_or_lower_to_starters(ver: str) -> bool:
    starters_version: str = _get_latest_starters_version()
    return parse(ver) <= parse(starters_version)


_STARTERS_REPO: str = 'git+https://github.com/kedro-org/kedro-starters.git'
_OFFICIAL_STARTER_SPECS: List[KedroStarterSpec] = [
    KedroStarterSpec('astro-airflow-iris', _STARTERS_REPO, 'astro-airflow-iris'),
    KedroStarterSpec('spaceflights-pandas', _STARTERS_REPO, 'spaceflights-pandas'),
    KedroStarterSpec('spaceflights-pandas-viz', _STARTERS_REPO, 'spaceflights-pandas-viz'),
    KedroStarterSpec('spaceflights-pyspark', _STARTERS_REPO, 'spaceflights-pyspark'),
    KedroStarterSpec('spaceflights-pyspark-viz', _STARTERS_REPO, 'spaceflights-pyspark-viz'),
    KedroStarterSpec('databricks-iris', _STARTERS_REPO, 'databricks-iris'),
]
for starter_spec in _OFFICIAL_STARTER_SPECS:
    starter_spec.origin = 'kedro'
_OFFICIAL_STARTER_SPECS_DICT: Dict[str, KedroStarterSpec] = {spec.alias: spec for spec in _OFFICIAL_STARTER_SPECS}
TOOLS_SHORTNAME_TO_NUMBER: Dict[str, str] = {
    'lint': '1',
    'test': '2',
    'tests': '2',
    'log': '3',
    'logs': '3',
    'docs': '4',
    'doc': '4',
    'data': '5',
    'pyspark': '6',
    'viz': '7',
}
NUMBER_TO_TOOLS_NAME: Dict[str, str] = {
    '1': 'Linting',
    '2': 'Testing',
    '3': 'Custom Logging',
    '4': 'Documentation',
    '5': 'Data Structure',
    '6': 'PySpark',
    '7': 'Kedro Viz',
}


def _validate_flag_inputs(flag_inputs: Dict[str, Any]) -> None:
    if flag_inputs.get('checkout') and (not flag_inputs.get('starter')):
        raise KedroCliError('Cannot use the --checkout flag without a --starter value.')
    if flag_inputs.get('directory') and (not flag_inputs.get('starter')):
        raise KedroCliError('Cannot use the --directory flag without a --starter value.')
    if (flag_inputs.get('tools') or flag_inputs.get('example')) and flag_inputs.get('starter'):
        raise KedroCliError('Cannot use the --starter flag with the --example and/or --tools flag.')


def _validate_input_with_regex_pattern(pattern_name: str, input_val: str) -> None:
    VALIDATION_PATTERNS: Dict[str, Dict[str, str]] = {
        'yes_no': {
            'regex': r'(?i)^\s*(y|yes|n|no)\s*$',
            'error_message': f"'{input_val}' is an invalid value for example pipeline. It must contain only y, n, YES, or NO (case insensitive).",
        },
        'project_name': {
            'regex': r'^[\w -]{2,}$',
            'error_message': f"'{input_val}' is an invalid value for project name. It must contain only alphanumeric symbols, spaces, underscores and hyphens and be at least 2 characters long",
        },
        'tools': {
            'regex': r'''^(
                all|none|                        # A: "all" or "none" or
                (\ *\d+                          # B: any number of spaces followed by one or more digits
                (\ *-\ *\d+)?                    # C: zero or one instances of: a hyphen followed by one or more digits, spaces allowed
                (\ *,\ *\d+(\ *-\ *\d+)?)*       # D: any number of instances of: a comma followed by B and C, spaces allowed
                \ *?)                           # E: zero or one instances of (B,C,D) as empty strings are also permissible
                $''',
            'error_message': f"'{input_val}' is an invalid value for project tools. Please select valid options for tools using comma-separated values, ranges, or 'all/none'.",
        },
    }
    if not re.match(VALIDATION_PATTERNS[pattern_name]['regex'], input_val, flags=re.X):
        click.secho(VALIDATION_PATTERNS[pattern_name]['error_message'], fg='red', err=True)
        sys.exit(1)


def _parse_yes_no_to_bool(value: Optional[str]) -> Optional[bool]:
    return value.strip().lower() in ['y', 'yes'] if value is not None else None


def _validate_selected_tools(selected_tools: Optional[str]) -> None:
    valid_tools: List[str] = [*list(TOOLS_SHORTNAME_TO_NUMBER), 'all', 'none']
    if selected_tools is not None:
        tools: List[str] = re.sub(r'\s', '', selected_tools).split(',')
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
        warnings.warn(
            "Your project does not contain any pipelines with nodes. Please ensure that at least one pipeline has been defined before executing 'kedro run'.",
            UserWarning,
        )
    if interactive:
        click.secho(
            '\nTo skip the interactive flow you can run `kedro new` with\nkedro new --name=<your-project-name> --tools=<your-project-tools> --example=<yes/no>',
            fg='green',
        )


@click.group(context_settings=CONTEXT_SETTINGS, name='Kedro')
def create_cli() -> None:
    pass


@create_cli.group()
def starter() -> None:
    """Commands for working with project starters."""
    pass


@command_with_verbosity(create_cli, short_help='Create a new kedro project.')
@click.option('--config', '-c', 'config_path', type=click.Path(exists=True), help=CONFIG_ARG_HELP)
@click.option('--starter', '-s', 'starter_alias', help=STARTER_ARG_HELP)
@click.option('--checkout', help=CHECKOUT_ARG_HELP)
@click.option('--directory', help=DIRECTORY_ARG_HELP)
@click.option('--name', '-n', 'project_name', help=NAME_ARG_HELP)
@click.option('--tools', '-t', 'selected_tools', help=TOOLS_ARG_HELP)
@click.option('--example', '-e', 'example_pipeline', help=EXAMPLE_ARG_HELP)
@click.option('--telemetry', '-tc', 'telemetry_consent', help=TELEMETRY_ARG_HELP, type=click.Choice(['yes', 'no', 'y', 'n'], case_sensitive=False))
def new(
    config_path: Optional[str],
    starter_alias: Optional[str],
    selected_tools: Optional[str],
    project_name: Optional[str],
    checkout: Optional[str],
    directory: Optional[str],
    example_pipeline: Optional[str],
    telemetry_consent: Optional[str],
    **kwargs: Any,
) -> None:
    """Create a new kedro project."""
    flag_inputs: Dict[str, Any] = {
        'config': config_path,
        'starter': starter_alias,
        'tools': selected_tools,
        'name': project_name,
        'checkout': checkout,
        'directory': directory,
        'example': example_pipeline,
        'telemetry_consent': telemetry_consent,
    }
    _validate_flag_inputs(flag_inputs)
    starters_dict: Dict[str, KedroStarterSpec] = _get_starters_dict()
    if starter_alias in starters_dict:
        if directory:
            raise KedroCliError('Cannot use the --directory flag with a --starter alias.')
        spec: KedroStarterSpec = starters_dict[starter_alias]
        template_path: Union[str, Path] = spec.template_path
        directory = spec.directory
        checkout = _select_checkout_branch_for_cookiecutter(checkout)
    elif starter_alias is not None:
        template_path = starter_alias
    else:
        template_path = str(TEMPLATE_PATH)
    if selected_tools is not None:
        selected_tools = selected_tools.lower()
    tmpdir: str = tempfile.mkdtemp()
    cookiecutter_dir: Path = _get_cookiecutter_dir(template_path, checkout, directory, tmpdir)
    prompts_required: Dict[str, Any] = _get_prompts_required_and_clear_from_CLI_provided(cookiecutter_dir, selected_tools, project_name, example_pipeline)
    cookiecutter_context: Optional[Dict[str, Any]] = None
    if not config_path:
        cookiecutter_context = _make_cookiecutter_context_for_prompts(cookiecutter_dir)
    shutil.rmtree(tmpdir, onerror=_remove_readonly)
    extra_context: Dict[str, Any] = _get_extra_context(
        prompts_required=prompts_required,
        config_path=config_path,
        cookiecutter_context=cookiecutter_context,
        selected_tools=selected_tools,
        project_name=project_name,
        example_pipeline=example_pipeline,
        starter_alias=starter_alias,
    )
    cookiecutter_args, project_template = _make_cookiecutter_args_and_fetch_template(
        config=extra_context, checkout=checkout, directory=directory, template_path=str(template_path)
    )
    if telemetry_consent is not None:
        telemetry_consent = 'true' if _parse_yes_no_to_bool(telemetry_consent) else 'false'
    _create_project(project_template, cookiecutter_args, telemetry_consent)
    if not starter_alias:
        interactive_flow: bool = bool(prompts_required) and (not config_path)
        _print_selection_and_prompt_info(extra_context['tools'], extra_context['example_pipeline'], interactive_flow)


@starter.command('list')
def list_starters() -> None:
    """List all official project starters available."""
    starters_dict: Dict[str, KedroStarterSpec] = _get_starters_dict()
    sorted_starters_dict: Dict[str, Dict[str, KedroStarterSpec]] = {
        origin: dict(sorted(starters_dict_by_origin))
        for origin, starters_dict_by_origin in groupby(
            starters_dict.items(), lambda item: item[1].origin
        )
    }
    sorted_starters_dict = dict(sorted(sorted_starters_dict.items(), key=lambda x: x[0] == 'kedro'))
    for origin, starters_spec in sorted_starters_dict.items():
        click.secho(f'\nStarters from {origin}\n', fg='yellow')
        click.echo(yaml.safe_dump(_starter_spec_to_dict(starters_spec), sort_keys=False))


def _get_cookiecutter_dir(
    template_path: Union[str, Path],
    checkout: Optional[str],
    directory: Optional[str],
    tmpdir: str,
) -> Path:
    """Gives a path to the cookiecutter directory. If template_path is a repo then
    clones it to ``tmpdir``; if template_path is a file path then directly uses that
    path without copying anything.
    """
    from cookiecutter.exceptions import RepositoryCloneFailed, RepositoryNotFound
    from cookiecutter.repository import determine_repo_dir

    try:
        cookiecutter_dir, _ = determine_repo_dir(
            template=template_path,
            abbreviations={},
            clone_to_dir=Path(tmpdir).resolve(),
            checkout=checkout,
            no_input=True,
            directory=directory,
        )
    except (RepositoryNotFound, RepositoryCloneFailed) as exc:
        error_message: str = f'Kedro project template not found at {template_path}.'
        if checkout:
            error_message += f' Specified tag {checkout}. The following tags are available: ' + ', '.join(_get_available_tags(str(template_path)))
        official_starters: List[str] = sorted(list(_OFFICIAL_STARTER_SPECS_DICT))
        raise KedroCliError(
            f'{error_message}. The aliases for the official Kedro starters are: \n{yaml.safe_dump(official_starters, sort_keys=False)}'
        ) from exc
    return Path(cookiecutter_dir)


def _get_prompts_required_and_clear_from_CLI_provided(
    cookiecutter_dir: Path,
    selected_tools: Optional[str],
    project_name: Optional[str],
    example_pipeline: Optional[str],
) -> Dict[str, Any]:
    """Finds the information a user must supply according to prompts.yml,
    and clear it from what has already been provided via the CLI(validate it before)
    """
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
        unique_tags: set = {tag.split('/')[-1].replace('^{}', '') for tag in tags.split('\n') if tag}
    except git.GitCommandError:
        return []
    return sorted(unique_tags)


def _get_starters_dict() -> Dict[str, KedroStarterSpec]:
    """This function lists all the starter aliases declared in
    the core repo and in plugins entry points.

    For example, the output for official kedro starters looks like:
    {"astro-airflow-iris":
        KedroStarterSpec(
            alias="astro-airflow-iris",
            template_path="git+https://github.com/kedro-org/kedro-starters.git",
            directory="astro-airflow-iris",
            origin="kedro"
        ),
    }
    """
    starter_specs: Dict[str, KedroStarterSpec] = _OFFICIAL_STARTER_SPECS_DICT.copy()
    for starter_entry_point in _get_entry_points(name='starters'):
        origin: str = starter_entry_point.module.split('.')[0]
        specs: Any = _safe_load_entry_point(starter_entry_point) or []
        for spec in specs:
            if not isinstance(spec, KedroStarterSpec):
                click.secho(f"The starter configuration loaded from module {origin} should be a 'KedroStarterSpec', got '{type(spec)}' instead", fg='red')
            elif spec.alias in starter_specs:
                click.secho(f'Starter alias `{spec.alias}` from `{origin}` has been ignored as it is already defined by `{starter_specs[spec.alias].origin}`', fg='red')
            else:
                spec.origin = origin
                starter_specs[spec.alias] = spec
    return starter_specs


def _get_extra_context(
    prompts_required: Dict[str, Any],
    config_path: Optional[str],
    cookiecutter_context: Optional[Dict[str, Any]],
    selected_tools: Optional[str],
    project_name: Optional[str],
    example_pipeline: Optional[str],
    starter_alias: Optional[str],
) -> Dict[str, Any]:
    """Generates a config dictionary that will be passed to cookiecutter as `extra_context`, based
    on CLI flags, user prompts, configuration file or Default values.
    It is crucial to return a dictionary with string values, otherwise, there will be issues with Cookiecutter.
    """
    if config_path:
        extra_context: Dict[str, Any] = _fetch_validate_parse_config_from_file(config_path, prompts_required, starter_alias)
    else:
        extra_context = _fetch_validate_parse_config_from_user_prompts(prompts_required, cookiecutter_context or {})
    if selected_tools is not None:
        tools_numbers: List[str] = _convert_tool_short_names_to_numbers(selected_tools)
        extra_context['tools'] = _convert_tool_numbers_to_readable_names(tools_numbers)
    if project_name is not None:
        extra_context['project_name'] = project_name
    if example_pipeline is not None:
        extra_context['example_pipeline'] = str(_parse_yes_no_to_bool(example_pipeline))
    extra_context.setdefault('kedro_version', version)
    extra_context.setdefault('tools', str(['None']))
    extra_context.setdefault('example_pipeline', 'False')
    return extra_context


def _convert_tool_short_names_to_numbers(selected_tools: str) -> List[str]:
    """Prepares tools selection from the CLI or config input to the correct format
    to be put in the project configuration, if it exists.
    Replaces tool strings with the corresponding prompt number.
    """
    if selected_tools.lower() == 'none':
        return []
    if selected_tools.lower() == 'all':
        return list(NUMBER_TO_TOOLS_NAME.keys())
    tools: List[str] = []
    for tool in selected_tools.lower().split(','):
        tool_short_name: str = tool.strip()
        if tool_short_name in TOOLS_SHORTNAME_TO_NUMBER:
            tools.append(TOOLS_SHORTNAME_TO_NUMBER[tool_short_name])
    tools = sorted(list(set(tools)))
    return tools


def _convert_tool_numbers_to_readable_names(tools_numbers: List[str]) -> str:
    """Transform the list of tool numbers into a list of readable names, using 'None' for empty lists.
    Then, convert the result into a string format to prevent issues with Cookiecutter.
    """
    tools_names: List[str] = [NUMBER_TO_TOOLS_NAME[tool] for tool in tools_numbers]
    if tools_names == []:
        tools_names = ['None']
    return str(tools_names)


def _fetch_validate_parse_config_from_file(config_path: str, prompts_required: Dict[str, Any], starter_alias: Optional[str]) -> Dict[str, Any]:
    """Obtains configuration for a new kedro project non-interactively from a file.
    """
    try:
        with open(config_path, encoding='utf-8') as config_file:
            config: Dict[str, Any] = yaml.safe_load(config_file)
        if KedroCliError.VERBOSE_ERROR:
            click.echo(config_path + ':')
            click.echo(yaml.dump(config, default_flow_style=False))
    except Exception as exc:
        raise KedroCliError(f'Failed to generate project: could not load config at {config_path}.') from exc
    if starter_alias and ('tools' in config or 'example_pipeline' in config):
        raise KedroCliError('The --starter flag can not be used with `example_pipeline` and/or `tools` keys in the config file.')
    _validate_config_file_against_prompts(config, prompts_required)
    _validate_input_with_regex_pattern('project_name', config.get('project_name', 'New Kedro Project'))
    example_pipeline: str = config.get('example_pipeline', 'no')
    _validate_input_with_regex_pattern('yes_no', example_pipeline)
    config['example_pipeline'] = str(_parse_yes_no_to_bool(example_pipeline))
    tools_short_names: str = config.get('tools', 'none').lower()
    _validate_selected_tools(tools_short_names)
    tools_numbers: List[str] = _convert_tool_short_names_to_numbers(tools_short_names)
    config['tools'] = _convert_tool_numbers_to_readable_names(tools_numbers)
    return config


def _fetch_validate_parse_config_from_user_prompts(prompts: Dict[str, Any], cookiecutter_context: Dict[str, Any]) -> Dict[str, Any]:
    """Interactively obtains information from user prompts.
    """
    if not cookiecutter_context:
        raise Exception('No cookiecutter context available.')
    config: Dict[str, Any] = {}
    for variable_name, prompt_dict in prompts.items():
        prompt = _Prompt(**prompt_dict)
        default_value: str = cookiecutter_context.get(variable_name) or ''
        user_input: str = click.prompt(str(prompt), default=default_value, show_default=True, type=str).strip()
        if user_input:
            prompt.validate(user_input)
            config[variable_name] = user_input
    if 'tools' in config:
        tools_numbers: List[str] = _parse_tools_input(config['tools'])
        _validate_tool_selection(tools_numbers)
        config['tools'] = _convert_tool_numbers_to_readable_names(tools_numbers)
    if 'example_pipeline' in config:
        example_pipeline_bool: Optional[bool] = _parse_yes_no_to_bool(config['example_pipeline'])
        config['example_pipeline'] = str(example_pipeline_bool)
    return config


def _make_cookiecutter_context_for_prompts(cookiecutter_dir: Path) -> Dict[str, Any]:
    from cookiecutter.generate import generate_context
    cookiecutter_context: Dict[str, Any] = generate_context(cookiecutter_dir / 'cookiecutter.json')
    return cookiecutter_context.get('cookiecutter', {})


def _select_checkout_branch_for_cookiecutter(checkout: Optional[str]) -> str:
    if checkout:
        return checkout
    elif _kedro_version_equal_or_lower_to_starters(version):
        return version
    else:
        return 'main'


def _make_cookiecutter_args_and_fetch_template(
    config: Dict[str, Any],
    checkout: Optional[str],
    directory: Optional[str],
    template_path: str,
) -> Tuple[Dict[str, Any], str]:
    """Creates a dictionary of arguments to pass to cookiecutter and returns project template path.
    """
    cookiecutter_args: Dict[str, Any] = {
        'output_dir': config.get('output_dir', str(Path.cwd().resolve())),
        'no_input': True,
        'extra_context': config,
    }
    if directory:
        cookiecutter_args['directory'] = directory
    tools: str = config['tools']
    example_pipeline: str = config['example_pipeline']
    starter_path: str = 'git+https://github.com/kedro-org/kedro-starters.git'
    cookiecutter_args['checkout'] = checkout
    if 'PySpark' in tools and 'Kedro Viz' in tools:
        cookiecutter_args['directory'] = 'spaceflights-pyspark-viz'
    elif 'PySpark' in tools:
        cookiecutter_args['directory'] = 'spaceflights-pyspark'
    elif 'Kedro Viz' in tools:
        cookiecutter_args['directory'] = 'spaceflights-pandas-viz'
    elif example_pipeline == 'True':
        cookiecutter_args['directory'] = 'spaceflights-pandas'
    else:
        starter_path = template_path
    return (cookiecutter_args, starter_path)


def _validate_config_file_against_prompts(config: Dict[str, Any], prompts: Dict[str, Any]) -> None:
    """Checks that the configuration file contains all needed variables.
    """
    if not config:
        raise KedroCliError('Config file is empty.')
    additional_keys: Dict[str, str] = {'tools': 'none', 'example_pipeline': 'no'}
    missing_keys: set = set(prompts) - set(config)
    missing_mandatory_keys: set = missing_keys - set(additional_keys)
    if missing_mandatory_keys:
        click.echo(yaml.dump(config, default_flow_style=False))
        raise KedroCliError(f'{", ".join(missing_mandatory_keys)} not found in config file.')
    for key, default_value in additional_keys.items():
        if key in missing_keys:
            click.secho(f"The `{key}` key not found in the config file, default value '{default_value}' is being used.", fg='yellow')
    if 'output_dir' in config and (not Path(config['output_dir']).exists()):
        raise KedroCliError(f"'{config['output_dir']}' is not a valid output directory. It must be a relative or absolute path to an existing directory.")


def _validate_tool_selection(tools: List[str]) -> None:
    for tool in tools[::-1]:
        if tool not in NUMBER_TO_TOOLS_NAME:
            message: str = f"'{tool}' is not a valid selection.\nPlease select from the available tools: 1, 2, 3, 4, 5, 6, 7."
            click.secho(message, fg='red', err=True)
            sys.exit(1)


def _parse_tools_input(tools_str: str) -> List[str]:
    """Parse the tools input string.
    """
    def _validate_range(start: str, end: str) -> None:
        if int(start) > int(end):
            message: str = f"'{start}-{end}' is an invalid range for project tools.\nPlease ensure range values go from smaller to larger."
            click.secho(message, fg='red', err=True)
            sys.exit(1)
        if int(end) > len(NUMBER_TO_TOOLS_NAME):
            message: str = f"'{end}' is not a valid selection.\nPlease select from the available tools: 1, 2, 3, 4, 5, 6, 7."
            click.secho(message, fg='red', err=True)
            sys.exit(1)

    if not tools_str:
        return []
    tools_str = tools_str.lower()
    if tools_str == 'all':
        return list(NUMBER_TO_TOOLS_NAME.keys())
    if tools_str == 'none':
        return []
    tools_choices: List[str] = tools_str.replace(' ', '').split(',')
    selected: List[str] = []
    for choice in tools_choices:
        if '-' in choice:
            start, end = choice.split('-')
            _validate_range(start, end)
            selected.extend([str(i) for i in range(int(start), int(end) + 1)])
        else:
            selected.append(choice.strip())
    return selected


def _create_project(template_path: str, cookiecutter_args: Dict[str, Any], telemetry_consent: Optional[str]) -> None:
    """Creates a new kedro project using cookiecutter.
    """
    from cookiecutter.main import cookiecutter
    try:
        result_path: str = cookiecutter(template=template_path, **cookiecutter_args)
        if telemetry_consent is not None:
            with open(result_path + '/.telemetry', 'w') as telemetry_file:
                telemetry_file.write('consent: ' + telemetry_consent)
    except Exception as exc:
        raise KedroCliError('Failed to generate project when running cookiecutter.') from exc
    _clean_pycache(Path(result_path))
    extra_context: Dict[str, Any] = cookiecutter_args['extra_context']
    project_name: str = extra_context.get('project_name', 'New Kedro Project')
    click.secho(f"\nCongratulations!\nYour project '{project_name}' has been created in the directory \n{result_path}\n", fg='green')


class _Prompt:
    """Represent a single CLI prompt for `kedro new`"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.title: str = kwargs['title']
        except KeyError as exc:
            raise KedroCliError('Each prompt must have a title field to be valid.') from exc
        self.text: str = kwargs.get('text', '')
        self.regexp: Optional[str] = kwargs.get('regex_validator', None)
        self.error_message: str = kwargs.get('error_message', '')

    def __str__(self) -> str:
        title: str = self.title.strip().title()
        title = click.style(title, fg='cyan', bold=True)
        title_line: str = '=' * len(self.title)
        title_line = click.style(title_line, fg='cyan', bold=True)
        text: str = self.text.strip()
        prompt_lines: List[str] = [title, title_line, text]
        prompt_text: str = '\n'.join((str(line).strip() for line in prompt_lines))
        return f'\n{prompt_text}\n'

    def validate(self, user_input: str) -> None:
        """Validate a given prompt value against the regex validator"""
        if self.regexp and (not re.match(self.regexp, user_input)):
            message: str = f"'{user_input}' is an invalid value for {self.title.lower()}."
            click.secho(message, fg='red', err=True)
            click.secho(self.error_message, fg='red', err=True)
            sys.exit(1)


def _remove_readonly(func: Callable[[str], Any], path: str, excinfo: Any) -> None:
    """Remove readonly files on Windows
    See: https://docs.python.org/3/library/shutil.html?highlight=shutil#rmtree-example
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _starter_spec_to_dict(starter_specs: Dict[str, KedroStarterSpec]) -> Dict[str, Any]:
    """Convert a dictionary of starters spec to a nicely formatted dictionary"""
    format_dict: Dict[str, Any] = {}
    for alias, spec in starter_specs.items():
        format_dict[alias] = {}
        format_dict[alias]['template_path'] = spec.template_path
        if spec.directory:
            format_dict[alias]['directory'] = spec.directory
    return format_dict

if __name__ == '__main__':
    create_cli()