from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from dbt.artifacts.resources import ColumnInfo, NodeVersion
from dbt.contracts.graph.nodes import UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import HasColumnDocs, HasColumnProps, HasColumnTests, UnparsedAnalysisUpdate, UnparsedColumn, UnparsedExposure, UnparsedMacroUpdate, UnparsedModelUpdate, UnparsedNodeUpdate, UnparsedSingularTestUpdate
from dbt.exceptions import ParsingError
from dbt.node_types import NodeType
from dbt.parser.search import FileBlock
from dbt_common.contracts.constraints import ColumnLevelConstraint, ConstraintType
from dbt_common.exceptions import DbtInternalError
from dbt_semantic_interfaces.type_enums import TimeGranularity

schema_file_keys_to_resource_types: Dict[str, NodeType] = {
    'models': NodeType.Model,
    'seeds': NodeType.Seed,
    'snapshots': NodeType.Snapshot,
    'sources': NodeType.Source,
    'macros': NodeType.Macro,
    'analyses': NodeType.Analysis,
    'exposures': NodeType.Exposure,
    'metrics': NodeType.Metric,
    'semantic_models': NodeType.SemanticModel,
    'saved_queries': NodeType.SavedQuery
}
resource_types_to_schema_file_keys: Dict[NodeType, str] = {v: k for k, v in schema_file_keys_to_resource_types.items()}
schema_file_keys: List[str] = list(schema_file_keys_to_resource_types.keys())

def trimmed(inp: str) -> str:
    if len(inp) < 50:
        return inp
    return inp[:44] + '...' + inp[-3:]

TestDef = Union[str, Dict[str, Any]]
Target = TypeVar('Target', UnparsedNodeUpdate, UnparsedMacroUpdate, UnparsedAnalysisUpdate, UnpatchedSourceDefinition, UnparsedExposure, UnparsedModelUpdate, UnparsedSingularTestUpdate)
ColumnTarget = TypeVar('ColumnTarget', UnparsedModelUpdate, UnparsedNodeUpdate, UnparsedAnalysisUpdate, UnpatchedSourceDefinition)
Versioned = TypeVar('Versioned', bound=UnparsedModelUpdate)
Testable = TypeVar('Testable', UnparsedNodeUpdate, UnpatchedSourceDefinition, UnparsedModelUpdate)

@dataclass
class YamlBlock(FileBlock):
    @classmethod
    def from_file_block(cls, src: FileBlock, data: Any) -> 'YamlBlock':
        return cls(file=src.file, data=data)

@dataclass
class TargetBlock(YamlBlock, Generic[Target]):
    target: Target

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def columns(self) -> List[Any]:
        return []

    @property
    def data_tests(self) -> List[Any]:
        return []

    @property
    def tests(self) -> List[Any]:
        return []

    @classmethod
    def from_yaml_block(cls, src: YamlBlock, target: Target) -> 'TargetBlock':
        return cls(file=src.file, data=src.data, target=target)

@dataclass
class TargetColumnsBlock(TargetBlock[ColumnTarget], Generic[ColumnTarget]):
    @property
    def columns(self) -> List[Any]:
        if self.target.columns is None:
            return []
        else:
            return self.target.columns

@dataclass
class TestBlock(TargetColumnsBlock[Testable], Generic[Testable]):
    @property
    def data_tests(self) -> List[Any]:
        if self.target.data_tests is None:
            return []
        else:
            return self.target.data_tests

    @property
    def quote_columns(self) -> Optional[bool]:
        return self.target.quote_columns

    @classmethod
    def from_yaml_block(cls, src: YamlBlock, target: Testable) -> 'TestBlock':
        return cls(file=src.file, data=src.data, target=target)

@dataclass
class VersionedTestBlock(TestBlock, Generic[Versioned]):
    @property
    def columns(self) -> List[Any]:
        if not self.target.versions:
            return super().columns
        else:
            raise DbtInternalError('.columns for VersionedTestBlock with versions')

    @property
    def data_tests(self) -> List[Any]:
        if not self.target.versions:
            return super().data_tests
        else:
            raise DbtInternalError('.data_tests for VersionedTestBlock with versions')

    @classmethod
    def from_yaml_block(cls, src: YamlBlock, target: Versioned) -> 'VersionedTestBlock':
        return cls(file=src.file, data=src.data, target=target)

@dataclass
class GenericTestBlock(TestBlock[Testable], Generic[Testable]):
    @classmethod
    def from_test_block(cls, src: TestBlock, data_test: Any, column_name: str, tags: List[str], version: Optional[NodeVersion]) -> 'GenericTestBlock':
        return cls(file=src.file, data=src.data, target=src.target, data_test=data_test, column_name=column_name, tags=tags, version=version)

class ParserRef:
    """A helper object to hold parse-time references."""

    def __init__(self) -> None:
        self.column_info: Dict[str, ColumnInfo] = {}

    def _add(self, column: UnparsedColumn) -> None:
        tags = getattr(column, 'tags', [])
        quote = None
        granularity = None
        if isinstance(column, UnparsedColumn):
            quote = column.quote
            granularity = TimeGranularity(column.granularity) if column.granularity else None
        if any((c for c in column.constraints if 'type' not in c or not ConstraintType.is_valid(c['type']))):
            raise ParsingError(f'Invalid constraint type on column {column.name}')
        self.column_info[column.name] = ColumnInfo(
            name=column.name,
            description=column.description,
            data_type=column.data_type,
            constraints=[ColumnLevelConstraint.from_dict(c) for c in column.constraints],
            meta=column.meta,
            tags=tags,
            quote=quote,
            _extra=column.extra,
            granularity=granularity
        )

    @classmethod
    def from_target(cls, target: ColumnTarget) -> 'ParserRef':
        refs = cls()
        for column in target.columns:
            refs._add(column)
        return refs

    @classmethod
    def from_versioned_target(cls, target: Versioned, version: NodeVersion) -> 'ParserRef':
        refs = cls()
        for base_column in target.get_columns_for_version(version):
            refs._add(base_column)
        return refs
