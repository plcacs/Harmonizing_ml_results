import configparser
import fnmatch
import os
import posixpath
import re
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Pattern, Set, Tuple, Type, Union
from warnings import warn
from . import sorting, stdlibs
from .exceptions import FormattingPluginDoesNotExist, InvalidSettingsPath, ProfileDoesNotExist, SortingFunctionDoesNotExist, UnsupportedSettings
from .profiles import profiles as profiles
from .sections import DEFAULT as SECTION_DEFAULTS
from .sections import FIRSTPARTY, FUTURE, LOCALFOLDER, STDLIB, THIRDPARTY
from .utils import Trie
from .wrap_modes import WrapModes
from .wrap_modes import from_string as wrap_mode_from_string
if TYPE_CHECKING:
    import tomllib
else:
    from ._vendored import tomli as tomllib
_SHEBANG_RE: re.Pattern = re.compile(b'^#!.*\\bpython[23w]?\\b')
CYTHON_EXTENSIONS: FrozenSet[str] = frozenset({'pyx', 'pxd'})
SUPPORTED_EXTENSIONS: FrozenSet[str] = frozenset({'py', 'pyi', *CYTHON_EXTENSIONS})
BLOCKED_EXTENSIONS: FrozenSet[str] = frozenset({'pex'})
FILE_SKIP_COMMENTS: Tuple[str, str] = ('isort:skip_file', 'isort: skip_file')
MAX_CONFIG_SEARCH_DEPTH: int = 25
STOP_CONFIG_SEARCH_ON_DIRS: Tuple[str] = ('.git', '.hg')
VALID_PY_TARGETS: Tuple[str] = tuple((target.replace('py', '') for target in dir(stdlibs) if not target.startswith('_')))
CONFIG_SOURCES: Tuple[str] = ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
DEFAULT_SKIP: FrozenSet[str] = frozenset({'.venv', 'venv', '.tox', '.eggs', '.git', '.hg', '.mypy_cache', '.nox', '.svn', '.bzr', '_build', 'buck-out', 'build', 'dist', '.pants.d', '.direnv', 'node_modules', '__pypackages__', '.pytype'})
CONFIG_SECTIONS: Dict[str, Tuple[str, ...]] = {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
FALLBACK_CONFIG_SECTIONS: Tuple[str] = ('isort', 'tool:isort', 'tool.isort')
IMPORT_HEADING_PREFIX: str = 'import_heading_'
IMPORT_FOOTER_PREFIX: str = 'import_footer_'
KNOWN_PREFIX: str = 'known_'
KNOWN_SECTION_MAPPING: Dict[str, str] = {STDLIB: 'STANDARD_LIBRARY', FUTURE: 'FUTURE_LIBRARY', FIRSTPARTY: 'FIRST_PARTY', THIRDPARTY: 'THIRD_PARTY', LOCALFOLDER: 'LOCAL_FOLDER'}
RUNTIME_SOURCE: str = 'runtime'
DEPRECATED_SETTINGS: Tuple[str] = ('not_skip', 'keep_direct_and_as_imports')
_STR_BOOLEAN_MAPPING: Dict[str, bool] = {'y': True, 'yes': True, 't': True, 'on': True, '1': True, 'true': True, 'n': False, 'no': False, 'f': False, 'off': False, '0': False, 'false': False}

@dataclass(frozen=True)
class _Config:
    py_version: str = '3'
    force_to_top: FrozenSet[str] = frozenset()
    skip: FrozenSet[str] = DEFAULT_SKIP
    extend_skip: FrozenSet[str] = frozenset()
    skip_glob: FrozenSet[str] = frozenset()
    extend_skip_glob: FrozenSet[str] = frozenset()
    skip_gitignore: bool = False
    line_length: int = 79
    wrap_length: int = 0
    line_ending: str = ''
    sections: FrozenSet[str] = SECTION_DEFAULTS
    no_sections: bool = False
    known_future_library: FrozenSet[str] = frozenset(('__future__',))
    known_third_party: FrozenSet[str] = frozenset()
    known_first_party: FrozenSet[str] = frozenset()
    known_local_folder: FrozenSet[str] = frozenset()
    known_standard_library: FrozenSet[str] = frozenset()
    extra_standard_library: FrozenSet[str] = frozenset()
    known_other: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    multi_line_output: WrapModes = WrapModes.GRID
    forced_separate: Tuple[str] = ()
    indent: str = ' ' * 4
    comment_prefix: str = '  #'
    length_sort: bool = False
    length_sort_straight: bool = False
    length_sort_sections: FrozenSet[str] = frozenset()
    add_imports: FrozenSet[str] = frozenset()
    remove_imports: FrozenSet[str] = frozenset()
    append_only: bool = False
    reverse_relative: bool = False
    force_single_line: bool = False
    single_line_exclusions: Tuple[str] = ()
    default_section: str = THIRDPARTY
    import_headings: Dict[str, str] = field(default_factory=dict)
    import_footers: Dict[str, str] = field(default_factory=dict)
    balanced_wrapping: bool = False
    use_parentheses: bool = False
    order_by_type: bool = True
    atomic: bool = False
    lines_before_imports: int = -1
    lines_after_imports: int = -1
    lines_between_sections: int = 1
    lines_between_types: int = 0
    combine_as_imports: bool = False
    combine_star: bool = False
    include_trailing_comma: bool = False
    from_first: bool = False
    verbose: bool = False
    quiet: bool = False
    force_adds: bool = False
    force_alphabetical_sort_within_sections: bool = False
    force_alphabetical_sort: bool = False
    force_grid_wrap: int = 0
    force_sort_within_sections: bool = False
    lexicographical: bool = False
    group_by_package: bool = False
    ignore_whitespace: bool = False
    no_lines_before: FrozenSet[str] = frozenset()
    no_inline_sort: bool = False
    ignore_comments: bool = False
    case_sensitive: bool = False
    sources: Tuple[str] = ()
    virtual_env: str = ''
    conda_env: str = ''
    ensure_newline_before_comments: bool = False
    directory: str = ''
    profile: str = ''
    honor_noqa: bool = False
    src_paths: Tuple[str] = ()
    old_finders: bool = False
    remove_redundant_aliases: bool = False
    float_to_top: bool = False
    filter_files: bool = False
    formatter: str = ''
    formatting_function: Optional[Callable] = None
    color_output: bool = False
    treat_comments_as_code: FrozenSet[str] = frozenset()
    treat_all_comments_as_code: bool = False
    supported_extensions: FrozenSet[str] = SUPPORTED_EXTENSIONS
    blocked_extensions: FrozenSet[str] = BLOCKED_EXTENSIONS
    constants: FrozenSet[str] = frozenset()
    classes: FrozenSet[str] = frozenset()
    variables: FrozenSet[str] = frozenset()
    dedup_headings: bool = False
    only_sections: bool = False
    only_modified: bool = False
    combine_straight_imports: bool = False
    auto_identify_namespace_packages: bool = True
    namespace_packages: FrozenSet[str] = frozenset()
    follow_links: bool = True
    indented_import_headings: bool = True
    honor_case_in_force_sorted_sections: bool = False
    sort_relative_in_force_sorted_sections: bool = False
    overwrite_in_place: bool = False
    reverse_sort: bool = False
    star_first: bool = False
    import_dependencies: Dict[str, str] = {}
    git_ls_files: Dict[str, str] = field(default_factory=dict)
    format_error: str = '{error}: {message}'
    format_success: str = '{success}: {message}'
    sort_order: str = 'natural'
    sort_reexports: bool = False
    split_on_trailing_comma: bool = False

    def __post_init__(self) -> None:
        py_version = self.py_version
        if py_version == 'auto':
            py_version = f'{sys.version_info.major}{sys.version_info.minor}'
        if py_version not in VALID_PY_TARGETS:
            raise ValueError(f'The python version {py_version} is not supported. You can set a python version with the -py or --python-version flag. The following versions are supported: {VALID_PY_TARGETS}')
        if py_version != 'all':
            object.__setattr__(self, 'py_version', f'py{py_version}')
        if not self.known_standard_library:
            object.__setattr__(self, 'known_standard_library', frozenset(getattr(stdlibs, self.py_version).stdlib))
        if self.multi_line_output == WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA:
            vertical_grid_grouped = WrapModes.VERTICAL_GRID_GROUPED
            object.__setattr__(self, 'multi_line_output', vertical_grid_grouped)
        if self.force_alphabetical_sort:
            object.__setattr__(self, 'force_alphabetical_sort_within_sections', True)
            object.__setattr__(self, 'no_sections', True)
            object.__setattr__(self, 'lines_between_types', 1)
            object.__setattr__(self, 'from_first', True)
        if self.wrap_length > self.line_length:
            raise ValueError(f'wrap_length must be set lower than or equal to line_length: {self.wrap_length} > {self.line_length}.')

    def __hash__(self) -> int:
        return id(self)
_DEFAULT_SETTINGS: Dict[str, Any] = {**vars(_Config()), 'source': 'defaults'}

class Config(_Config):

    def __init__(self, settings_file: str = '', settings_path: str = '', config: Optional[_Config] = None, **config_overrides: Any) -> None:
        self._known_patterns: Optional[List[Tuple[re.Pattern, str]]] = None
        self._section_comments: Optional[Tuple[str]] = None
        self._section_comments_end: Optional[Tuple[str]] = None
        self._skips: Optional[FrozenSet[str]] = None
        self._skip_globs: Optional[FrozenSet[str]] = None
        self._sorting_function: Optional[Callable] = None
        if config:
            config_vars = vars(config).copy()
            config_vars.update(config_overrides)
            config_vars['py_version'] = config_vars['py_version'].replace('py', '')
            config_vars.pop('_known_patterns')
            config_vars.pop('_section_comments')
            config_vars.pop('_section_comments_end')
            config_vars.pop('_skips')
            config_vars.pop('_skip_globs')
            super().__init__(**config_vars)
            return
        quiet = config_overrides.get('quiet', False)
        sources = [_DEFAULT_SETTINGS]
        if settings_file:
            config_settings = _get_config_data(settings_file, CONFIG_SECTIONS.get(os.path.basename(settings_file), FALLBACK_CONFIG_SECTIONS))
            project_root = os.path.dirname(settings_file)
            if not config_settings and (not quiet):
                warn(f'A custom settings file was specified: {settings_file} but no configuration was found inside. This can happen when [settings] is used as the config header instead of [isort]. See: https://pycqa.github.io/isort/docs/configuration/config_files#custom-config-files for more information.')
        elif settings_path:
            if not os.path.exists(settings_path):
                raise InvalidSettingsPath(settings_path)
            settings_path = os.path.abspath(settings_path)
            project_root, config_settings = _find_config(settings_path)
        else:
            config_settings = {}
            project_root = os.getcwd()
        profile_name = config_overrides.get('profile', config_settings.get('profile', ''))
        profile = {}
        if profile_name:
            if profile_name not in profiles:
                import pkg_resources
                for plugin in pkg_resources.iter_entry_points('isort.profiles'):
                    profiles.setdefault(plugin.name, plugin.load())
            if profile_name not in profiles:
                raise ProfileDoesNotExist(profile_name)
            profile = profiles[profile_name].copy()
            profile['source'] = f'{profile_name} profile'
            sources.append(profile)
        if config_settings:
            sources.append(config_settings)
        if config_overrides:
            config_overrides['source'] = RUNTIME_SOURCE
            sources.append(config_overrides)
        combined_config = {**profile, **config_settings, **config_overrides}
        if 'indent' in combined_config:
            indent = str(combined_config['indent'])
            if indent.isdigit():
                indent = ' ' * int(indent)
            else:
                indent = indent.strip("'").strip('"')
                if indent.lower() == 'tab':
                    indent = '\t'
            combined_config['indent'] = indent
        known_other = {}
        import_headings = {}
        import_footers = {}
        for key, value in tuple(combined_config.items()):
            if key.startswith(KNOWN_PREFIX) and key not in ('known_standard_library', 'known_future_library', 'known_third_party', 'known_first_party', 'known_local_folder'):
                import_heading = key[len(KNOWN_PREFIX):].lower()
                maps_to_section = import_heading.upper()
                combined_config.pop(key)
                if maps_to_section in KNOWN_SECTION_MAPPING:
                    section_name = f'known_{KNOWN_SECTION_MAPPING[maps_to_section].lower()}'
                    if section_name in combined_config and (not quiet):
                        warn(f"Can't set both {key} and {section_name} in the same config file.\nDefault to {section_name} if unsure.\n\nSee: https://pycqa.github.io/isort/#custom-sections-and-ordering.")
                    else:
                        combined_config[section_name] = frozenset(value)
                else:
                    known_other[import_heading] = frozenset(value)
                    if maps_to_section not in combined_config.get('sections', ()) and (not quiet):
                        warn(f'`{key}` setting is defined, but {maps_to_section} is not included in `sections` config option: {combined_config.get('sections', SECTION_DEFAULTS)}.\n\nSee: https://pycqa.github.io/isort/#custom-sections-and-ordering.')
            if key.startswith(IMPORT_HEADING_PREFIX):
                import_headings[key[len(IMPORT_HEADING_PREFIX):].lower()] = str(value)
            if key.startswith(IMPORT_FOOTER_PREFIX):
                import_footers[key[len(IMPORT_FOOTER_PREFIX):].lower()] = str(value)
            default_value = _DEFAULT_SETTINGS.get(key, None)
            if default_value is None:
                continue
            combined_config[key] = type(default_value)(value)
        for section in combined_config.get('sections', ()):
            if section in SECTION_DEFAULTS:
                continue
            if not section.lower() in known_other:
                config_keys = ', '.join(known_other.keys())
                warn(f'`sections` setting includes {section}, but no known_{section.lower()} is defined. The following known_SECTION config options are defined: {config_keys}.')
        if 'directory' not in combined_config:
            combined_config['directory'] = os.path.dirname(config_settings['source']) if config_settings.get('source', None) else os.getcwd()
        path_root = Path(combined_config.get('directory', project_root)).resolve()
        path_root = path_root if path_root.is_dir() else path_root.parent
        if 'src_paths' not in combined_config:
            combined_config['src_paths'] = (path_root / 'src', path_root)
        else:
            src_paths = []
            for src_path in combined_config.get('src_paths', ()):
                full_paths = path_root.glob(src_path) if '*' in str(src_path) else [path_root / src_path]
                for path in full_paths:
                    if path not in src_paths:
                        src_paths.append(path)
            combined_config['src_paths'] = tuple(src_paths)
        if 'formatter' in combined_config:
            import pkg_resources
            for plugin in pkg_resources.iter_entry_points('isort.formatters'):
                if plugin.name == combined_config['formatter']:
                    combined_config['formatting_function'] = plugin.load()
                    break
            else:
                raise FormattingPluginDoesNotExist(combined_config['formatter'])
        combined_config.pop('source', None)
        combined_config.pop('sources', None)
        combined_config.pop('runtime_src_paths', None)
        deprecated_options_used = [option for option in combined_config if option in DEPRECATED_SETTINGS]
        if deprecated_options_used:
            for deprecated_option in deprecated_options_used:
                combined_config.pop(deprecated_option)
            if not quiet:
                warn(f'W0503: Deprecated config options were used: {', '.join(deprecated_options_used)}.Please see the 5.0.0 upgrade guide: https://pycqa.github.io/isort/docs/upgrade_guides/5.0.0.html')
        if known_other:
            combined_config['known_other'] = known_other
        if import_headings:
            for import_heading_key in import_headings:
                combined_config.pop(f'{IMPORT_HEADING_PREFIX}{import_heading_key}')
            combined_config['import_headings'] = import_headings
        if import_footers:
            for import_footer_key in import_footers:
                combined_config.pop(f'{IMPORT_FOOTER_PREFIX}{import_footer_key}')
            combined_config['import_footers'] = import_footers
        unsupported_config_errors = {}
        for option in set(combined_config.keys()).difference(getattr(_Config, '__dataclass_fields__', {}).keys()):
            for source in reversed(sources):
                if option in source:
                    unsupported_config_errors[option] = {'value': source[option], 'source': source['source']}
        if unsupported_config_errors:
            raise UnsupportedSettings(unsupported_config_errors)
        super().__init__(sources=tuple(sources), **combined_config)

    def is_supported_filetype(self, file_name: str) -> bool:
        _root, ext = os.path.splitext(file_name)
        ext = ext.lstrip('.')
        if ext in self.supported_extensions:
            return True
        if ext in self.blocked_extensions:
            return False
        if file_name.endswith('~'):
            return False
        try:
            if stat.S_ISFIFO(os.stat(file_name).st_mode):
                return False
        except OSError:
            pass
        try:
            with open(file_name, 'rb') as fp:
                line = fp.readline(100)
        except OSError:
            return False
        return bool(_