from copy import deepcopy
from datetime import datetime
from typing import Any, List, Optional
import pytest
from freezegun import freeze_time
from dbt.artifacts.resources import Defaults, Dimension, Entity, FileHash, Measure, TestMetadata
from dbt.artifacts.resources.v1.semantic_model import NodeRelation
from dbt.contracts.graph.model_config import TestConfig
from dbt.contracts.graph.nodes import ColumnInfo, ModelNode, ParsedNode, SemanticModel
from dbt.node_types import NodeType
from dbt_common.contracts.constraints import ColumnLevelConstraint, ConstraintType, ModelLevelConstraint
from dbt_semantic_interfaces.references import MeasureReference
from dbt_semantic_interfaces.type_enums import AggregationType, DimensionType, EntityType

class TestModelNode:
    @pytest.fixture(scope='class')
    def default_model_node(self) -> ModelNode:
        ...

    @pytest.mark.parametrize('deprecation_date,current_date,expected_is_past_deprecation_date', [(None, '2024-05-02', False), ('2024-05-01', '2024-05-02', True), ('2024-05-01', '2024-05-01', False), ('2024-05-01', '2024-04-30', False)])
    def test_is_past_deprecation_date(self, default_model_node: ModelNode, deprecation_date: Optional[str], current_date: str, expected_is_past_deprecation_date: bool) -> None:
        ...

    @pytest.mark.parametrize('model_constraints,columns,expected_all_constraints', [([], {}, []), ([ModelLevelConstraint(type=ConstraintType.foreign_key)], {}, [ModelLevelConstraint(type=ConstraintType.foreign_key)]), ([], {'id': ColumnInfo(name='id', constraints=[ColumnLevelConstraint(type=ConstraintType.foreign_key)])}, [ColumnLevelConstraint(type=ConstraintType.foreign_key)]), ([ModelLevelConstraint(type=ConstraintType.foreign_key)], {'id': ColumnInfo(name='id', constraints=[ColumnLevelConstraint(type=ConstraintType.foreign_key)])}, [ModelLevelConstraint(type=ConstraintType.foreign_key), ColumnLevelConstraint(type=ConstraintType.foreign_key)])])
    def test_all_constraints(self, default_model_node: ModelNode, model_constraints: List[ModelLevelConstraint], columns: dict, expected_all_constraints: List[Union[ModelLevelConstraint, ColumnLevelConstraint]]) -> None:
        ...

class TestSemanticModel:
    @pytest.fixture(scope='function')
    def dimensions(self) -> List[Dimension]:
        ...

    @pytest.fixture(scope='function')
    def entities(self) -> List[Entity]:
        ...

    @pytest.fixture(scope='function')
    def measures(self) -> List[Measure]:
        ...

    @pytest.fixture(scope='function')
    def default_semantic_model(self, dimensions: List[Dimension], entities: List[Entity], measures: List[Measure]) -> SemanticModel:
        ...

    def test_checked_agg_time_dimension_for_measure_via_defaults(self, default_semantic_model: SemanticModel) -> None:
        ...

    def test_checked_agg_time_dimension_for_measure_via_measure(self, default_semantic_model: SemanticModel) -> None:
        ...

    def test_checked_agg_time_dimension_for_measure_exception(self, default_semantic_model: SemanticModel) -> None:
        ...

    def test_semantic_model_same_contents(self, default_semantic_model: SemanticModel) -> None:
        ...

    def test_semantic_model_same_contents_update_model(self, default_semantic_model: SemanticModel) -> None:
        ...

    def test_semantic_model_same_contents_different_node_relation(self, default_semantic_model: SemanticModel) -> None:
        ...

def test_no_primary_key() -> None:
    ...

def test_primary_key_model_constraint() -> None:
    ...

def test_primary_key_column_constraint() -> None:
    ...

def test_unique_non_null_single() -> None:
    ...

def test_unique_non_null_multiple() -> None:
    ...

def test_enabled_unique_single() -> None:
    ...

def test_enabled_unique_multiple() -> None:
    ...

def test_enabled_unique_combo_single() -> None:
    ...

def test_enabled_unique_combo_multiple() -> None:
    ...

def test_disabled_unique_single() -> None:
    ...

def test_disabled_unique_multiple() -> None:
    ...

def test_disabled_unique_combo_single() -> None:
    ...

def test_disabled_unique_combo_multiple() -> None:
    ...

def assertSameContents(list1: List[Any], list2: List[Any]) -> None:
    ...

class TestParsedNode:
    @pytest.fixture(scope='class')
    def parsed_node(self) -> ParsedNode:
        ...

    def test_get_target_write_path(self, parsed_node: ParsedNode) -> None:
        ...

    def test_get_target_write_path_split(self, parsed_node: ParsedNode) -> None:
        ...