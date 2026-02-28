from typing import Any, Dict, List, Optional

def make_model(
    pkg: str,
    name: str,
    code: str,
    language: str = 'sql',
    refs: Optional[List] = None,
    sources: Optional[List] = None,
    tags: Optional[List] = None,
    path: Optional[str] = None,
    alias: Optional[str] = None,
    config_kwargs: Optional[Dict] = None,
    fqn_extras: Optional[List] = None,
    depends_on_macros: Optional[List] = None,
    version: Optional[str] = None,
    latest_version: Optional[str] = None,
    access: Optional[str] = None,
    patch_path: Optional[str] = None,
) -> ModelNode:
    # ...

def make_seed(
    pkg: str,
    name: str,
    path: Optional[str] = None,
    loader: Optional[str] = None,
    alias: Optional[str] = None,
    tags: Optional[List] = None,
    fqn_extras: Optional[List] = None,
    checksum: Optional[str] = None,
) -> SeedNode:
    # ...

def make_source(
    pkg: str,
    source_name: str,
    table_name: str,
    path: Optional[str] = None,
    loader: Optional[str] = None,
    identifier: Optional[str] = None,
    fqn_extras: Optional[List] = None,
) -> SourceDefinition:
    # ...

def make_macro(
    pkg: str,
    name: str,
    macro_sql: str,
    path: Optional[str] = None,
    depends_on_macros: Optional[List] = None,
) -> Macro:
    # ...

def make_unique_test(
    pkg: str,
    test_model: Any,
    column_name: str,
    path: Optional[str] = None,
    refs: Optional[List] = None,
    sources: Optional[List] = None,
    tags: Optional[List] = None,
) -> GenericTestNode:
    # ...

def make_not_null_test(
    pkg: str,
    test_model: Any,
    column_name: str,
    path: Optional[str] = None,
    refs: Optional[List] = None,
    sources: Optional[List] = None,
    tags: Optional[List] = None,
) -> GenericTestNode:
    # ...

def make_generic_test(
    pkg: str,
    test_name: str,
    test_model: Any,
    test_kwargs: Dict,
    path: Optional[str] = None,
    refs: Optional[List] = None,
    sources: Optional[List] = None,
    tags: Optional[List] = None,
    column_name: Optional[str] = None,
) -> GenericTestNode:
    # ...

def make_unit_test(
    pkg: str,
    test_name: str,
    test_model: Any,
) -> UnitTestDefinition:
    # ...

def make_singular_test(
    pkg: str,
    name: str,
    sql: str,
    refs: Optional[List] = None,
    sources: Optional[List] = None,
    tags: Optional[List] = None,
    path: Optional[str] = None,
    config_kwargs: Optional[Dict] = None,
) -> SingularTestNode:
    # ...

def make_exposure(
    pkg: str,
    name: str,
    path: Optional[str] = None,
    fqn_extras: Optional[List] = None,
    owner: Optional[Owner] = None,
) -> Exposure:
    # ...

def make_metric(
    pkg: str,
    name: str,
    path: Optional[str] = None,
) -> Metric:
    # ...

def make_group(
    pkg: str,
    name: str,
    path: Optional[str] = None,
) -> Group:
    # ...

def make_semantic_model(
    pkg: str,
    name: str,
    model: Any,
    path: Optional[str] = None,
) -> SemanticModel:
    # ...

def make_saved_query(
    pkg: str,
    name: str,
    metric: Metric,
    path: Optional[str] = None,
) -> SavedQuery:
    # ...

@pytest.fixture
def macro_test_unique() -> Macro:
    # ...

@pytest.fixture
def macro_default_test_unique() -> Macro:
    # ...

@pytest.fixture
def macro_test_not_null() -> Macro:
    # ...

@pytest.fixture
def macro_materialization_table_default() -> Macro:
    # ...

@pytest.fixture
def macro_default_test_not_null() -> Macro:
    # ...

@pytest.fixture
def seed() -> SeedNode:
    # ...

@pytest.fixture
def source() -> SourceDefinition:
    # ...

@pytest.fixture
def ephemeral_model(source: SourceDefinition) -> ModelNode:
    # ...

@pytest.fixture
def view_model(ephemeral_model: ModelNode) -> ModelNode:
    # ...

@pytest.fixture
def table_model(ephemeral_model: ModelNode) -> ModelNode:
    # ...

@pytest.fixture
def table_model_py(seed: SeedNode) -> ModelNode:
    # ...

@pytest.fixture
def table_model_csv(seed: SeedNode) -> ModelNode:
    # ...

@pytest.fixture
def ext_source() -> SourceDefinition:
    # ...

@pytest.fixture
def ext_source_2() -> SourceDefinition:
    # ...

@pytest.fixture
def ext_source_other() -> SourceDefinition:
    # ...

@pytest.fixture
def ext_source_other_2() -> SourceDefinition:
    # ...

@pytest.fixture
def ext_model(ext_source: SourceDefinition) -> ModelNode:
    # ...

@pytest.fixture
def union_model(seed: SeedNode, ext_source: SourceDefinition) -> ModelNode:
    # ...

@pytest.fixture
def versioned_model_v1(seed: SeedNode) -> ModelNode:
    # ...

@pytest.fixture
def versioned_model_v2(seed: SeedNode) -> ModelNode:
    # ...

@pytest.fixture
def versioned_model_v3(seed: SeedNode) -> ModelNode:
    # ...

@pytest.fixture
def versioned_model_v12_string(seed: SeedNode) -> ModelNode:
    # ...

@pytest.fixture
def versioned_model_v4_nested_dir(seed: SeedNode) -> ModelNode:
    # ...

@pytest.fixture
def table_id_unique(table_model: ModelNode) -> GenericTestNode:
    # ...

@pytest.fixture
def table_id_not_null(table_model: ModelNode) -> GenericTestNode:
    # ...

@pytest.fixture
def view_id_unique(view_model: ModelNode) -> GenericTestNode:
    # ...

@pytest.fixture
def ext_source_id_unique(ext_source: SourceDefinition) -> GenericTestNode:
    # ...

@pytest.fixture
def view_test_nothing(view_model: ModelNode) -> SingularTestNode:
    # ...

@pytest.fixture
def unit_test_table_model(table_model: ModelNode) -> UnitTestDefinition:
    # ...

@pytest.fixture
def namespaced_seed() -> SeedNode:
    # ...

@pytest.fixture
def namespace_model(source: SourceDefinition) -> ModelNode:
    # ...

@pytest.fixture
def namespaced_union_model(seed: SeedNode, ext_source: SourceDefinition) -> ModelNode:
    # ...

@pytest.fixture
def metric() -> Metric:
    # ...

@pytest.fixture
def saved_query() -> SavedQuery:
    # ...

@pytest.fixture
def semantic_model(table_model: ModelNode) -> SemanticModel:
    # ...

@pytest.fixture
def metricflow_time_spine_model() -> ModelNode:
    # ...

@pytest.fixture
def nodes(
    seed: SeedNode,
    ephemeral_model: ModelNode,
    view_model: ModelNode,
    table_model: ModelNode,
    table_model_py: ModelNode,
    table_model_csv: ModelNode,
    union_model: ModelNode,
    versioned_model_v1: ModelNode,
    versioned_model_v2: ModelNode,
    versioned_model_v3: ModelNode,
    versioned_model_v4_nested_dir: ModelNode,
    versioned_model_v12_string: ModelNode,
    ext_model: ModelNode,
    table_id_unique: GenericTestNode,
    table_id_not_null: GenericTestNode,
    view_id_unique: GenericTestNode,
    ext_source_id_unique: GenericTestNode,
    view_test_nothing: SingularTestNode,
    namespaced_seed: SeedNode,
    namespace_model: ModelNode,
    namespaced_union_model: ModelNode,
) -> List[GraphMemberNode]:
    # ...

@pytest.fixture
def sources(
    source: SourceDefinition,
    ext_source: SourceDefinition,
    ext_source_2: SourceDefinition,
    ext_source_other: SourceDefinition,
    ext_source_other_2: SourceDefinition,
) -> List[SourceDefinition]:
    # ...

@pytest.fixture
def macros(
    macro_test_unique: Macro,
    macro_default_test_unique: Macro,
    macro_test_not_null: Macro,
    macro_default_test_not_null: Macro,
    macro_materialization_table_default: Macro,
) -> List[Macro]:
    # ...

@pytest.fixture
def unit_tests(unit_test_table_model: UnitTestDefinition) -> List[UnitTestDefinition]:
    # ...

@pytest.fixture
def metrics(metric: Metric) -> List[Metric]:
    # ...

@pytest.fixture
def semantic_models(semantic_model: SemanticModel) -> List[SemanticModel]:
    # ...

@pytest.fixture
def saved_queries(saved_query: SavedQuery) -> List[SavedQuery]:
    # ...

@pytest.fixture
def files() -> Dict:
    # ...

def make_manifest(
    disabled: Dict,
    docs: List,
    exposures: List,
    files: Dict,
    groups: List,
    macros: List,
    metrics: List,
    nodes: List,
    saved_queries: List,
    selectors: Dict,
    semantic_models: List,
    sources: List,
    unit_tests: List,
) -> Manifest:
    # ...
