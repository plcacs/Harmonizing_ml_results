import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Protocol, Set, Tuple

import pathspec
from dbt.config import Project
from dbt.contracts.files import (
    AnySourceFile,
    FileHash,
    FilePath,
    FixtureSourceFile,
    ParseFileType,
    SchemaSourceFile,
    SourceFile,
)
from dbt.events.types import InputFileDiffError
from dbt.exceptions import ParsingError
from dbt.parser.common import schema_file_keys
from dbt.parser.schemas import yaml_from_file
from dbt.parser.search import filesystem_search
from dbt_common.clients.system import load_file_contents
from dbt_common.dataclass_schema import dbtClassMixin
from dbt_common.events.functions import fire_event

@dataclass
class InputFile(dbtClassMixin):
    path: str = ""
    content: str = ""
    modification_time: float = 0.0

@dataclass
class FileDiff(dbtClassMixin):
    deleted: List[str] = field(default_factory=list)
    changed: List[InputFile] = field(default_factory=list)
    added: List[InputFile] = field(default_factory=list)

def load_source_file(
    path: FilePath,
    parse_file_type: ParseFileType,
    project_name: str,
    saved_files: Optional[Mapping[str, SourceFile]]
) -> SourceFile:
    if parse_file_type == ParseFileType.Schema:
        sf_cls = SchemaSourceFile
    elif parse_file_type == ParseFileType.Fixture:
        sf_cls = FixtureSourceFile
    else:
        sf_cls = SourceFile
    source_file: SourceFile = sf_cls(
        path=path,
        checksum=FileHash.empty(),
        parse_file_type=parse_file_type,
        project_name=project_name,
    )
    skip_loading_schema_file: bool = False
    if parse_file_type == ParseFileType.Schema and saved_files and (source_file.file_id in saved_files):
        old_source_file: SourceFile = saved_files[source_file.file_id]
        if source_file.path.modification_time != 0.0 and old_source_file.path.modification_time == source_file.path.modification_time:
            source_file.checksum = old_source_file.checksum
            source_file.dfy = old_source_file.dfy
            skip_loading_schema_file = True
    if not skip_loading_schema_file:
        file_contents: str = load_file_contents(path.absolute_path, strip=True)
        source_file.contents = file_contents
        source_file.checksum = FileHash.from_contents(file_contents)
    if parse_file_type == ParseFileType.Schema and source_file.contents:
        dfy: Any = yaml_from_file(source_file)
        if dfy:
            validate_yaml(source_file.path.original_file_path, dfy)
            source_file.dfy = dfy
    return source_file

def validate_yaml(file_path: str, dct: Mapping[str, Any]) -> None:
    for key in schema_file_keys:
        if key in dct:
            if not isinstance(dct[key], list):
                msg = f"The schema file at {file_path} is invalid because the value of '{key}' is not a list"
                raise ParsingError(msg)
            for element in dct[key]:
                if not isinstance(element, dict):
                    msg = f"The schema file at {file_path} is invalid because a list element for '{key}' is not a dictionary"
                    raise ParsingError(msg)
                if 'name' not in element:
                    msg = f"The schema file at {file_path} is invalid because a list element for '{key}' does not have a name attribute."
                    raise ParsingError(msg)

def load_seed_source_file(match: FilePath, project_name: str) -> SourceFile:
    if match.seed_too_large():
        source_file: SourceFile = SourceFile.big_seed(match)
    else:
        file_contents: str = load_file_contents(match.absolute_path, strip=True)
        checksum: FileHash = FileHash.from_contents(file_contents)
        source_file = SourceFile(path=match, checksum=checksum)
        source_file.contents = ''
    source_file.parse_file_type = ParseFileType.Seed
    source_file.project_name = project_name
    return source_file

def get_source_files(
    project: Project,
    paths: List[str],
    extension: str,
    parse_file_type: ParseFileType,
    saved_files: Optional[Mapping[str, SourceFile]],
    ignore_spec: Optional[pathspec.PathSpec]
) -> List[SourceFile]:
    fp_list = filesystem_search(project, paths, extension, ignore_spec)
    fb_list: List[SourceFile] = []
    for fp in fp_list:
        if parse_file_type == ParseFileType.Seed:
            fb_list.append(load_seed_source_file(fp, project.project_name))
        else:
            if parse_file_type == ParseFileType.SingularTest:
                path_obj = pathlib.Path(fp.relative_path)
                if path_obj.parts[0] in ['generic', 'fixtures']:
                    continue
            file_obj = load_source_file(fp, parse_file_type, project.project_name, saved_files)
            if file_obj:
                fb_list.append(file_obj)
    return fb_list

def read_files_for_parser(
    project: Project,
    files: MutableMapping[str, SourceFile],
    parse_ft: ParseFileType,
    file_type_info: Mapping[str, Any],
    saved_files: Optional[Mapping[str, SourceFile]],
    ignore_spec: Optional[pathspec.PathSpec]
) -> List[str]:
    dirs: List[str] = file_type_info['paths']
    parser_files: List[str] = []
    for extension in file_type_info['extensions']:
        source_files: List[SourceFile] = get_source_files(project, dirs, extension, parse_ft, saved_files, ignore_spec)
        for sf in source_files:
            files[sf.file_id] = sf
            parser_files.append(sf.file_id)
    return parser_files

def generate_dbt_ignore_spec(project_root: str) -> Optional[pathspec.PathSpec]:
    ignore_file_path: str = os.path.join(project_root, '.dbtignore')
    ignore_spec: Optional[pathspec.PathSpec] = None
    if os.path.exists(ignore_file_path):
        with open(ignore_file_path) as f:
            ignore_spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, f)
    return ignore_spec

class ReadFiles(Protocol):
    def read_files(self) -> None:
        ...

@dataclass
class ReadFilesFromFileSystem:
    files: MutableMapping[str, SourceFile] = field(default_factory=dict)
    saved_files: MutableMapping[str, SourceFile] = field(default_factory=dict)
    project_parser_files: MutableMapping[str, MutableMapping[str, List[str]]] = field(default_factory=dict)
    all_projects: Mapping[str, Project] = field(default_factory=dict)

    def read_files(self) -> None:
        for project in self.all_projects.values():
            file_types: Mapping[ParseFileType, Dict[str, Any]] = get_file_types_for_project(project)
            self.read_files_for_project(project, file_types)

    def read_files_for_project(self, project: Project, file_types: Mapping[ParseFileType, Dict[str, Any]]) -> None:
        dbt_ignore_spec: Optional[pathspec.PathSpec] = generate_dbt_ignore_spec(project.project_root)
        project_files: MutableMapping[str, List[str]] = {}
        self.project_parser_files[project.project_name] = project_files
        for parse_ft, file_type_info in file_types.items():
            parser_files: List[str] = read_files_for_parser(
                project, self.files, parse_ft, file_type_info, self.saved_files, dbt_ignore_spec
            )
            project_files[file_type_info['parser']] = parser_files

@dataclass
class ReadFilesFromDiff:
    files: MutableMapping[str, SourceFile] = field(default_factory=dict)
    saved_files: MutableMapping[str, SourceFile] = field(default_factory=dict)
    project_parser_files: MutableMapping[str, Any] = field(default_factory=dict)
    project_file_types: MutableMapping[str, Any] = field(default_factory=dict)
    local_package_dirs: Optional[List[str]] = None
    file_diff: FileDiff = field(default_factory=FileDiff)
    all_projects: Mapping[str, Project] = field(default_factory=dict)
    root_project_name: str = ""

    def read_files(self) -> None:
        for file_id, source_file in self.saved_files.items():
            if isinstance(source_file, SchemaSourceFile):
                file_cls = SchemaSourceFile
            else:
                file_cls = SourceFile
            new_source_file: SourceFile = file_cls(
                path=source_file.path,
                checksum=source_file.checksum,
                project_name=source_file.project_name,
                parse_file_type=source_file.parse_file_type,
                contents=source_file.contents,
            )
            self.files[file_id] = new_source_file
        for input_file_path in self.file_diff.deleted:
            project_name: str = self.get_project_name(input_file_path)
            file_id: str = f'{project_name}://{input_file_path}'
            if file_id in self.files:
                self.files.pop(file_id)
            else:
                fire_event(InputFileDiffError(category='deleted file not found', file_id=file_id))
        for input_file in self.file_diff.changed:
            project_name = self.get_project_name(input_file.path)
            file_id = f'{project_name}://{input_file.path}'
            if file_id in self.files:
                source_file = self.files[file_id]
                source_file.contents = input_file.content
                source_file.checksum = FileHash.from_contents(input_file.content)
                source_file.path.modification_time = input_file.modification_time
                if isinstance(source_file, SchemaSourceFile) and source_file.contents:
                    dfy = yaml_from_file(source_file)
                    if dfy:
                        validate_yaml(source_file.path.original_file_path, dfy)
                        source_file.dfy = dfy
        for input_file in self.file_diff.added:
            project_name: str = self.get_project_name(input_file.path)
            input_file_path_obj: pathlib.PurePath = pathlib.PurePath(input_file.path)
            extension: str = input_file_path_obj.suffix
            searched_path: str = input_file_path_obj.parts[0]
            relative_path_parts = input_file_path_obj.parts[1:]
            relative_path: pathlib.PurePath = pathlib.PurePath('').joinpath(*relative_path_parts)
            input_file_path = FilePath(
                searched_path=searched_path,
                relative_path=str(relative_path),
                modification_time=input_file.modification_time,
                project_root=self.all_projects[project_name].project_root,
            )
            file_types, file_type_lookup = self.get_project_file_types(project_name)
            parse_ft_for_extension: Set[ParseFileType] = set()
            parse_ft_for_path: Set[ParseFileType] = set()
            if extension in file_type_lookup['extensions']:
                parse_ft_for_extension = file_type_lookup['extensions'][extension]
            if searched_path in file_type_lookup['paths']:
                parse_ft_for_path = file_type_lookup['paths'][searched_path]
            file_id: str = f'{project_name}://{input_file.path}'
            if len(parse_ft_for_extension) == 0 or len(parse_ft_for_path) == 0:
                fire_event(InputFileDiffError(category='not a project file', file_id=file_id))
                continue
            parse_ft_set: Set[ParseFileType] = parse_ft_for_extension.intersection(parse_ft_for_path)
            if len(parse_ft_set) != 1:
                fire_event(InputFileDiffError(category='unable to resolve diff file location', file_id=file_id))
                continue
            parse_ft: ParseFileType = parse_ft_set.pop()
            source_file_cls = SourceFile
            if parse_ft == ParseFileType.Schema:
                source_file_cls = SchemaSourceFile
            source_file: SourceFile = source_file_cls(
                path=input_file_path,
                contents=input_file.content,
                checksum=FileHash.from_contents(input_file.content),
                project_name=project_name,
                parse_file_type=parse_ft,
            )
            if source_file_cls == SchemaSourceFile:
                dfy = yaml_from_file(source_file)
                if dfy:
                    validate_yaml(source_file.path.original_file_path, dfy)
                    source_file.dfy = dfy
                else:
                    continue
            self.files[source_file.file_id] = source_file

    def get_project_name(self, path: str) -> str:
        return self.root_project_name

    def get_project_file_types(self, project_name: str) -> Tuple[
        Mapping[ParseFileType, Any], Mapping[str, Dict[str, Set[ParseFileType]]]
    ]:
        if project_name not in self.project_file_types:
            file_types: Mapping[ParseFileType, Any] = get_file_types_for_project(self.all_projects[project_name])
            file_type_lookup: Dict[str, Dict[str, Set[ParseFileType]]] = self.get_file_type_lookup(file_types)
            self.project_file_types[project_name] = {'file_types': file_types, 'file_type_lookup': file_type_lookup}
        file_types = self.project_file_types[project_name]['file_types']
        file_type_lookup = self.project_file_types[project_name]['file_type_lookup']
        return (file_types, file_type_lookup)

    def get_file_type_lookup(
        self, file_types: Mapping[ParseFileType, Any]
    ) -> Dict[str, Dict[str, Set[ParseFileType]]]:
        file_type_lookup: Dict[str, Dict[str, Set[ParseFileType]]] = {'paths': {}, 'extensions': {}}
        for parse_ft, file_type in file_types.items():
            for path_val in file_type['paths']:
                if path_val not in file_type_lookup['paths']:
                    file_type_lookup['paths'][path_val] = set()
                file_type_lookup['paths'][path_val].add(parse_ft)
            for extension in file_type['extensions']:
                if extension not in file_type_lookup['extensions']:
                    file_type_lookup['extensions'][extension] = set()
                file_type_lookup['extensions'][extension].add(parse_ft)
        return file_type_lookup

def get_file_types_for_project(project: Project) -> Mapping[ParseFileType, Dict[str, Any]]:
    file_types: Mapping[ParseFileType, Dict[str, Any]] = {
        ParseFileType.Macro: {
            'paths': project.macro_paths,
            'extensions': ['.sql'],
            'parser': 'MacroParser'
        },
        ParseFileType.Model: {
            'paths': project.model_paths,
            'extensions': ['.sql', '.py'],
            'parser': 'ModelParser'
        },
        ParseFileType.Snapshot: {
            'paths': project.snapshot_paths,
            'extensions': ['.sql'],
            'parser': 'SnapshotParser'
        },
        ParseFileType.Analysis: {
            'paths': project.analysis_paths,
            'extensions': ['.sql'],
            'parser': 'AnalysisParser'
        },
        ParseFileType.SingularTest: {
            'paths': project.test_paths,
            'extensions': ['.sql'],
            'parser': 'SingularTestParser'
        },
        ParseFileType.GenericTest: {
            'paths': project.generic_test_paths,
            'extensions': ['.sql'],
            'parser': 'GenericTestParser'
        },
        ParseFileType.Seed: {
            'paths': project.seed_paths,
            'extensions': ['.csv'],
            'parser': 'SeedParser'
        },
        ParseFileType.Documentation: {
            'paths': project.docs_paths,
            'extensions': ['.md'],
            'parser': 'DocumentationParser'
        },
        ParseFileType.Schema: {
            'paths': project.all_source_paths,
            'extensions': ['.yml', '.yaml'],
            'parser': 'SchemaParser'
        },
        ParseFileType.Fixture: {
            'paths': project.fixture_paths,
            'extensions': ['.csv', '.sql'],
            'parser': 'FixtureParser'
        },
    }
    return file_types