from typing import Dict, List, Optional, Protocol

import pathspec
import pathlib

from dbt.config import Project
from dbt.contracts.files import AnySourceFile, FileHash, FilePath, FixtureSourceFile, ParseFileType, SchemaSourceFile, SourceFile
from dbt.events.types import InputFileDiffError
from dbt.exceptions import ParsingError
from dbt.parser.common import schema_file_keys
from dbt.parser.schemas import yaml_from_file
from dbt.parser.search import filesystem_search
from dbt_common.clients.system import load_file_contents
from dbt_common.dataclass_schema import dbtClassMixin
from dbt_common.events.functions import fire_event

class ReadFiles(Protocol):
    def read_files(self) -> None:
        pass

def load_source_file(path: pathlib.Path, parse_file_type: ParseFileType, project_name: str, saved_files: Dict[str, AnySourceFile]) -> AnySourceFile:
    ...

def validate_yaml(file_path: str, dct: Dict[str, List[Dict[str, str]]]) -> None:
    ...

def load_seed_source_file(match: pathlib.Path, project_name: str) -> SourceFile:
    ...

def get_source_files(project: Project, paths: List[str], extension: str, parse_file_type: ParseFileType, saved_files: Dict[str, AnySourceFile], ignore_spec: Optional[pathspec.PathSpec]) -> List[AnySourceFile]:
    ...

def read_files_for_parser(project: Project, files: Dict[str, AnySourceFile], parse_ft: ParseFileType, file_type_info: Dict[str, List[str]], saved_files: Dict[str, AnySourceFile], ignore_spec: Optional[pathspec.PathSpec]) -> List[str]:
    ...

def generate_dbt_ignore_spec(project_root: str) -> Optional[pathspec.PathSpec]:
    ...

class ReadFilesFromFileSystem:
    files: Dict[str, AnySourceFile] = field(default_factory=dict)
    saved_files: Dict[str, AnySourceFile] = field(default_factory=dict)
    project_parser_files: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def read_files(self) -> None:
        ...

    def read_files_for_project(self, project: Project, file_types: Dict[ParseFileType, Dict[str, List[str]]]) -> None:
        ...

class ReadFilesFromDiff:
    files: Dict[str, AnySourceFile] = field(default_factory=dict)
    saved_files: Dict[str, AnySourceFile] = field(default_factory=dict)
    project_parser_files: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    project_file_types: Dict[str, Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    local_package_dirs: Optional[List[str]] = None

    def read_files(self) -> None:
        ...

    def get_project_name(self, path: str) -> str:
        ...

    def get_project_file_types(self, project_name: str) -> Tuple[Dict[ParseFileType, Dict[str, List[str]]], Dict[str, Dict[str, Set[ParseFileType]]]:
        ...

    def get_file_type_lookup(self, file_types: Dict[ParseFileType, Dict[str, List[str]]]) -> Dict[str, Dict[str, Set[ParseFileType]]]:
        ...

def get_file_types_for_project(project: Project) -> Dict[ParseFileType, Dict[str, List[str]]:
    ...
