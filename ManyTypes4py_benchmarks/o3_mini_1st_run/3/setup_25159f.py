#!/usr/bin/env python3
import ast
import copy
import json
import logging
import os
import string
import sys
import warnings
from collections import OrderedDict
from distutils import dir_util
from distutils.util import convert_path
from functools import lru_cache
from marko.block import Heading, FencedCode, LinkRefDef, BlankLine
from marko.ext.gfm import gfm
from marko.ext.gfm.elements import Table
from marko.inline import CodeSpan
from pathlib import Path
from ruamel.yaml import YAML
from setuptools import setup, find_packages, Command
from setuptools.command.build_py import build_py
from typing import Dict, List, Sequence, Optional, Tuple, Any
from pysetup.constants import PHASE0, ETH2_SPEC_COMMENT_PREFIX
from pysetup.helpers import combine_spec_objects, dependency_order_class_objects, objects_to_spec, parse_config_vars
from pysetup.md_doc_paths import get_md_doc_paths
from pysetup.spec_builders import spec_builders
from pysetup.typing import BuildTarget, ProtocolDefinition, SpecObject, VariableDefinition

warnings.filterwarnings('ignore', message='Normalizing .* to .*')

class PyspecFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(('running ', 'creating '))

logging.getLogger().addFilter(PyspecFilter())

@lru_cache(maxsize=None)
def _get_name_from_heading(heading: Heading) -> Optional[str]:
    last_child = heading.children[-1]
    if isinstance(last_child, CodeSpan):
        return last_child.children  # expected to be a str
    return None

@lru_cache(maxsize=None)
def _get_source_from_code_block(block: FencedCode) -> str:
    return block.children[0].children.strip()

@lru_cache(maxsize=None)
def _get_function_name_from_source(source: str) -> str:
    fn = ast.parse(source).body[0]
    return fn.name

@lru_cache(maxsize=None)
def _get_self_type_from_source(source: str) -> Optional[str]:
    fn = ast.parse(source).body[0]
    args = fn.args.args
    if len(args) == 0:
        return None
    if args[0].arg != 'self':
        return None
    if args[0].annotation is None:
        return None
    return args[0].annotation.id

@lru_cache(maxsize=None)
def _get_class_info_from_source(source: str) -> Tuple[str, Optional[str]]:
    class_def = ast.parse(source).body[0]
    base = class_def.bases[0]
    if isinstance(base, ast.Name):
        parent_class = base.id
    elif isinstance(base, ast.Subscript):
        parent_class = base.value.id
    else:
        parent_class = None
    return (class_def.name, parent_class)

@lru_cache(maxsize=None)
def _is_constant_id(name: str) -> bool:
    if name[0] not in string.ascii_uppercase + '_':
        return False
    return all(map(lambda c: c in string.ascii_uppercase + '_' + string.digits, name[1:]))

@lru_cache(maxsize=None)
def _load_kzg_trusted_setups(preset_name: str) -> Tuple[str, str, str]:
    trusted_setups_file_path = str(Path(__file__).parent) + '/presets/' + preset_name + '/trusted_setups/trusted_setup_4096.json'
    with open(trusted_setups_file_path, 'r') as f:
        json_data = json.load(f)
        trusted_setup_G1_monomial = json_data['g1_monomial']
        trusted_setup_G1_lagrange = json_data['g1_lagrange']
        trusted_setup_G2_monomial = json_data['g2_monomial']
    return (trusted_setup_G1_monomial, trusted_setup_G1_lagrange, trusted_setup_G2_monomial)

@lru_cache(maxsize=None)
def _load_curdleproofs_crs(preset_name: str) -> Any:
    """
    NOTE: File generated from https://github.com/asn-d6/curdleproofs/blob/8e8bf6d4191fb6a844002f75666fb7009716319b/tests/crs.rs#L53-L67
    """
    file_path = str(Path(__file__).parent) + '/presets/' + preset_name + '/trusted_setups/curdleproofs_crs.json'
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data

ALL_KZG_SETUPS: Dict[str, Tuple[str, str, str]] = {
    'minimal': _load_kzg_trusted_setups('minimal'),
    'mainnet': _load_kzg_trusted_setups('mainnet')
}
ALL_CURDLEPROOFS_CRS: Dict[str, Any] = {
    'minimal': _load_curdleproofs_crs('minimal'),
    'mainnet': _load_curdleproofs_crs('mainnet')
}

@lru_cache(maxsize=None)
def _get_eth2_spec_comment(child: LinkRefDef) -> Optional[str]:
    _, _, title = child._parse_info  # type: ignore
    if not (title[0] == '(' and title[len(title) - 1] == ')'):
        return None
    title = title[1:len(title) - 1]
    if not title.startswith(ETH2_SPEC_COMMENT_PREFIX):
        return None
    return title[len(ETH2_SPEC_COMMENT_PREFIX):].strip()

@lru_cache(maxsize=None)
def _parse_value(name: str, typed_value: str, type_hint: Optional[str] = None) -> VariableDefinition:
    comment: Optional[str] = None
    if name in ('ROOT_OF_UNITY_EXTENDED', 'ROOTS_OF_UNITY_EXTENDED', 'ROOTS_OF_UNITY_REDUCED'):
        comment = 'noqa: E501'
    typed_value = typed_value.strip()
    if '(' not in typed_value:
        return VariableDefinition(type_name=None, value=typed_value, comment=comment, type_hint=type_hint)
    i = typed_value.index('(')
    type_name = typed_value[:i]
    return VariableDefinition(type_name=type_name, value=typed_value[i + 1:-1], comment=comment, type_hint=type_hint)

def _update_constant_vars_with_kzg_setups(constant_vars: Dict[str, VariableDefinition], preset_name: str) -> None:
    comment: str = 'noqa: E501'
    kzg_setups: Tuple[str, str, str] = ALL_KZG_SETUPS[preset_name]
    constant_vars['KZG_SETUP_G1_MONOMIAL'] = VariableDefinition(
        constant_vars['KZG_SETUP_G1_MONOMIAL'].value, str(kzg_setups[0]), comment, None)
    constant_vars['KZG_SETUP_G1_LAGRANGE'] = VariableDefinition(
        constant_vars['KZG_SETUP_G1_LAGRANGE'].value, str(kzg_setups[1]), comment, None)
    constant_vars['KZG_SETUP_G2_MONOMIAL'] = VariableDefinition(
        constant_vars['KZG_SETUP_G2_MONOMIAL'].value, str(kzg_setups[2]), comment, None)

@lru_cache(maxsize=None)
def parse_markdown(content: str) -> Any:
    return gfm.parse(content)

def get_spec(file_name: str, preset: Dict[str, Any], config: Dict[str, Any], preset_name: str) -> SpecObject:
    functions: Dict[str, str] = {}
    protocols: Dict[str, ProtocolDefinition] = {}
    constant_vars: Dict[str, VariableDefinition] = {}
    preset_vars: Dict[str, VariableDefinition] = {}
    config_vars: Dict[str, VariableDefinition] = {}
    ssz_dep_constants: Dict[str, str] = {}
    func_dep_presets: Dict[str, str] = {}
    ssz_objects: Dict[str, str] = {}
    dataclasses: Dict[str, str] = {}
    custom_types: Dict[str, str] = {}
    with open(file_name) as source_file:
        document = parse_markdown(source_file.read())
    current_name: Optional[str] = None
    should_skip: bool = False
    for child in document.children:
        if isinstance(child, BlankLine):
            continue
        if should_skip:
            should_skip = False
            continue
        if isinstance(child, Heading):
            current_name = _get_name_from_heading(child)
        elif isinstance(child, FencedCode):
            if child.lang != 'python':
                continue
            source: str = _get_source_from_code_block(child)
            if source.startswith('def'):
                current_name = _get_function_name_from_source(source)
                self_type_name: Optional[str] = _get_self_type_from_source(source)
                function_def: str = '\n'.join((line.rstrip() for line in source.splitlines()))
                if self_type_name is None:
                    functions[current_name] = function_def
                else:
                    if self_type_name not in protocols:
                        protocols[self_type_name] = ProtocolDefinition(functions={})
                    protocols[self_type_name].functions[current_name] = function_def
            elif source.startswith('@dataclass'):
                dataclasses[current_name] = '\n'.join((line.rstrip() for line in source.splitlines()))
            elif source.startswith('class'):
                class_name, parent_class = _get_class_info_from_source(source)
                try:
                    assert class_name == current_name
                except Exception:
                    print('class_name', class_name)
                    print('current_name', current_name)
                    raise
                if parent_class:
                    assert parent_class == 'Container'
                ssz_objects[current_name] = '\n'.join((line.rstrip() for line in source.splitlines()))
            else:
                raise Exception('unrecognized python code element: ' + source)
        elif isinstance(child, Table):
            for row in child.children:
                cells = row.children
                if len(cells) >= 2:
                    name_cell = cells[0]
                    name = name_cell.children[0].children
                    value_cell = cells[1]
                    value = value_cell.children[0].children
                    description: Optional[str] = None
                    if len(cells) >= 3:
                        description_cell = cells[2]
                        if len(description_cell.children) > 0:
                            description = description_cell.children[0].children
                            if isinstance(description, list):
                                description = description[0].children
                    if isinstance(name, list):
                        name = name[0].children
                    if isinstance(value, list):
                        value = value[0].children
                    if description is not None and description.startswith('<!-- predefined-type -->'):
                        continue
                    if not _is_constant_id(name):
                        if value.startswith(('uint', 'Bytes', 'ByteList', 'Union', 'Vector', 'List', 'ByteVector')):
                            custom_types[name] = value
                        continue
                    if value.startswith('get_generalized_index'):
                        ssz_dep_constants[name] = value
                        continue
                    if description is not None and description.startswith('<!-- predefined -->'):
                        func_dep_presets[name] = value
                    value_def: VariableDefinition = _parse_value(name, value)
                    if name in preset:
                        preset_vars[name] = VariableDefinition(value_def.type_name, preset[name], value_def.comment, None)
                    elif name in config:
                        config_vars[name] = VariableDefinition(value_def.type_name, config[name], value_def.comment, None)
                    else:
                        if name in ('ENDIANNESS', 'KZG_ENDIANNESS'):
                            value_def = _parse_value(name, value, type_hint='Final')
                        constant_vars[name] = value_def
        elif isinstance(child, LinkRefDef):
            comment: Optional[str] = _get_eth2_spec_comment(child)
            if comment == 'skip':
                should_skip = True
    if any(('KZG_SETUP' in name for name in constant_vars)):
        _update_constant_vars_with_kzg_setups(constant_vars, preset_name)
    if any(('CURDLEPROOFS_CRS' in name for name in constant_vars)):
        constant_vars['CURDLEPROOFS_CRS'] = VariableDefinition(
            None, 'curdleproofs.CurdleproofsCrs.from_json(json.dumps(' + str(ALL_CURDLEPROOFS_CRS[str(preset_name)]).replace('0x', '') + '))', 'noqa: E501', None)
    return SpecObject(
        functions=functions,
        protocols=protocols,
        custom_types=custom_types,
        constant_vars=constant_vars,
        preset_vars=preset_vars,
        config_vars=config_vars,
        ssz_dep_constants=ssz_dep_constants,
        func_dep_presets=func_dep_presets,
        ssz_objects=ssz_objects,
        dataclasses=dataclasses
    )

@lru_cache(maxsize=None)
def load_preset(preset_files: Tuple[Path, ...]) -> Dict[str, Any]:
    """
    Loads the a directory of preset files, merges the result into one preset.
    """
    preset: Dict[str, Any] = {}
    for fork_file in preset_files:
        yaml_inst = YAML(typ='base')
        fork_preset = yaml_inst.load(fork_file)
        if fork_preset is None:
            continue
        if not set(fork_preset.keys()).isdisjoint(preset.keys()):
            duplicates = set(fork_preset.keys()).intersection(set(preset.keys()))
            raise Exception(f"duplicate config var(s) in preset files: {', '.join(duplicates)}")
        preset.update(fork_preset)
    assert preset != {}
    return parse_config_vars(preset)

@lru_cache(maxsize=None)
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Loads the given configuration file.
    """
    yaml_inst = YAML(typ='base')
    config_data = yaml_inst.load(config_path)
    return parse_config_vars(config_data)

def build_spec(fork: str, preset_name: str, source_files: Sequence[str],
               preset_files: Sequence[Path], config_file: Path) -> str:
    preset: Dict[str, Any] = load_preset(tuple(preset_files))
    config: Dict[str, Any] = load_config(config_file)
    all_specs: List[SpecObject] = [get_spec(spec, preset, config, preset_name) for spec in source_files]
    spec_object: SpecObject = all_specs[0]
    for value in all_specs[1:]:
        spec_object = combine_spec_objects(spec_object, value)
    class_objects: Dict[str, str] = {**spec_object.ssz_objects, **spec_object.dataclasses}
    new_objects: Dict[str, str] = {}
    while OrderedDict(new_objects) != OrderedDict(class_objects):
        new_objects = copy.deepcopy(class_objects)
        dependency_order_class_objects(class_objects, spec_object.custom_types)
    return objects_to_spec(preset_name, spec_object, fork, class_objects)

class PySpecCommand(Command):
    """Convert spec markdown files to a spec python file"""
    description: str = 'Convert spec markdown files to a spec python file'
    user_options: List[Tuple[str, Optional[str], str]] = [
        ('spec-fork=', None, 'Spec fork to tag build with. Used to select md-docs defaults.'),
        ('md-doc-paths=', None, 'List of paths of markdown files to build spec with'),
        ('build-targets=', None, 'Names, directory paths of compile-time presets, and default config paths.'),
        ('out-dir=', None, 'Output directory to write spec package to')
    ]

    def initialize_options(self) -> None:
        """Set default values for options."""
        self.spec_fork: str = PHASE0
        self.md_doc_paths: str = ''
        self.out_dir: str = 'pyspec_output'
        self.build_targets: str = (
            '\n                minimal:presets/minimal:configs/minimal.yaml\n                mainnet:presets/mainnet:configs/mainnet.yaml\n        '
        )

    def finalize_options(self) -> None:
        """Post-process options."""
        if len(self.md_doc_paths) == 0:
            self.md_doc_paths = get_md_doc_paths(self.spec_fork)
            if len(self.md_doc_paths) == 0:
                raise Exception('no markdown files specified, and spec fork "%s" is unknown' % self.spec_fork)
        self.parsed_md_doc_paths: List[str] = self.md_doc_paths.split()
        for filename in self.parsed_md_doc_paths:
            if not os.path.exists(filename):
                raise Exception('Pyspec markdown input file "%s" does not exist.' % filename)
        self.parsed_build_targets: List[BuildTarget] = []
        for target in self.build_targets.split():
            target = target.strip()
            data = target.split(':')
            if len(data) != 3:
                raise Exception('invalid target, expected "name:preset_dir:config_file" format, but got: %s' % target)
            name, preset_dir_path, config_path = data
            if any((c not in string.digits + string.ascii_letters for c in name)):
                raise Exception('invalid target name: "%s"' % name)
            if not os.path.exists(preset_dir_path):
                raise Exception('Preset dir "%s" does not exist' % preset_dir_path)
            _, _, preset_file_names = next(os.walk(preset_dir_path))
            preset_paths: List[Path] = [Path(preset_dir_path) / name for name in preset_file_names]
            if not os.path.exists(config_path):
                raise Exception('Config file "%s" does not exist' % config_path)
            from pysetup.typing import BuildTarget  # local import if needed
            self.parsed_build_targets.append(BuildTarget(name, preset_paths, Path(config_path)))

    def run(self) -> None:
        if not getattr(self, 'dry_run', False):
            dir_util.mkpath(self.out_dir)
        print(f'Building pyspec: {self.spec_fork}')
        for name, preset_paths, config_path in self.parsed_build_targets:
            spec_str: str = build_spec(spec_builders[self.spec_fork].fork, name, self.parsed_md_doc_paths, preset_paths, config_path)
            if getattr(self, 'dry_run', False):
                self.announce(f'dry run successfully prepared contents for spec. out dir: "{self.out_dir}", spec fork: "{self.spec_fork}", build target: "{name}"')
                self.debug_print(spec_str)
            else:
                with open(os.path.join(self.out_dir, name + '.py'), 'w') as out:
                    out.write(spec_str)
        if not getattr(self, 'dry_run', False):
            with open(os.path.join(self.out_dir, '__init__.py'), 'w') as out:
                out.write('from . import mainnet as spec  # noqa:F401\n')

class BuildPyCommand(build_py):
    """Customize the build command to run the spec-builder on setup.py build"""

    def initialize_options(self) -> None:
        super(BuildPyCommand, self).initialize_options()

    def run_pyspec_cmd(self, spec_fork: str, **opts: Any) -> None:
        cmd_obj: Command = self.distribution.reinitialize_command('pyspec')
        setattr(cmd_obj, 'spec_fork', spec_fork)
        setattr(cmd_obj, 'out_dir', os.path.join(self.build_lib, 'eth2spec', spec_fork))
        for k, v in opts.items():
            setattr(cmd_obj, k, v)
        self.run_command('pyspec')

    def run(self) -> None:
        for spec_fork in spec_builders:
            self.run_pyspec_cmd(spec_fork=spec_fork)
        super(BuildPyCommand, self).run()

class PyspecDevCommand(Command):
    """Build the markdown files in-place to their source location for testing."""
    description: str = 'Build the markdown files in-place to their source location for testing.'
    user_options: List[Tuple[str, Optional[str], str]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run_pyspec_cmd(self, spec_fork: str, **opts: Any) -> None:
        cmd_obj: Command = self.distribution.reinitialize_command('pyspec')
        setattr(cmd_obj, 'spec_fork', spec_fork)
        eth2spec_dir: str = convert_path(self.distribution.package_dir['eth2spec'])
        setattr(cmd_obj, 'out_dir', os.path.join(eth2spec_dir, spec_fork))
        for k, v in opts.items():
            setattr(cmd_obj, k, v)
        self.run_command('pyspec')

    def run(self) -> None:
        for spec_fork in spec_builders:
            self.run_pyspec_cmd(spec_fork=spec_fork)

commands: Dict[str, Any] = {
    'pyspec': PySpecCommand,
    'build_py': BuildPyCommand,
    'pyspecdev': PyspecDevCommand
}

with open('README.md', 'rt', encoding='utf8') as f:
    readme: str = f.read()
with open(os.path.join('tests', 'core', 'pyspec', 'eth2spec', 'VERSION.txt')) as f:
    spec_version: str = f.read().strip()

setup(
    version=spec_version,
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/ethereum/consensus-specs',
    include_package_data=False,
    package_data={
        'configs': ['*.yaml'],
        'eth2spec': ['VERSION.txt'],
        'presets': ['**/*.yaml', '**/*.json'],
        'specs': ['**/*.md'],
        'sync': ['optimistic.md']
    },
    package_dir={
        'configs': 'configs',
        'eth2spec': 'tests/core/pyspec/eth2spec',
        'presets': 'presets',
        'specs': 'specs',
        'sync': 'sync'
    },
    packages=find_packages(where='tests/core/pyspec') + ['configs', 'presets', 'specs', 'presets', 'sync'],
    py_modules=['eth2spec'],
    cmdclass=commands
)