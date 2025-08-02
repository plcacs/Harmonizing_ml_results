from typing import Any, Dict, List, Optional, Union
import pytest
from dbt.artifacts.resources import (
    ExposureType,
    MacroDependsOn,
    MetricInputMeasure,
    MetricTypeParams,
    NodeRelation,
    Owner,
    QueryParams,
    RefArgs,
    TestConfig,
    TestMetadata,
    WhereFilter,
    WhereFilterIntersection,
)
from dbt.artifacts.resources.types import ModelLanguage
from dbt.artifacts.resources.v1.model import ModelConfig
from dbt.contracts.files import AnySourceFile, FileHash
from dbt.contracts.graph.manifest import Manifest, ManifestMetadata
from dbt.contracts.graph.nodes import (
    AccessType,
    DependsOn,
    Documentation,
    Exposure,
    GenericTestNode,
    GraphMemberNode,
    Group,
    Macro,
    ManifestNode,
    Metric,
    ModelNode,
    NodeConfig,
    SavedQuery,
    SeedNode,
    SemanticModel,
    SingularTestNode,
    SourceDefinition,
    UnitTestDefinition,
)
from dbt.contracts.graph.unparsed import UnitTestInputFixture, UnitTestOutputFixture
from dbt.node_types import NodeType
from dbt_semantic_interfaces.type_enums import MetricType


def func_vhyebr8c(
    pkg: str,
    name: str,
    code: str,
    language: str = "sql",
    refs: Optional[List[Any]] = None,
    sources: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
    path: Optional[str] = None,
    alias: Optional[str] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
    fqn_extras: Optional[List[str]] = None,
    depends_on_macros: Optional[List[str]] = None,
    version: Optional[Union[int, str]] = None,
    latest_version: Optional[Union[int, str]] = None,
    access: Optional[AccessType] = None,
    patch_path: Optional[str] = None,
) -> ModelNode:
    if refs is None:
        refs = []
    if sources is None:
        sources = []
    if tags is None:
        tags = []
    if path is None:
        if language == "sql":
            path = f"{name}.sql"
        elif language == "python":
            path = f"{name}.py"
        else:
            raise ValueError(f"Unknown language: {language}")
    if alias is None:
        alias = name
    if config_kwargs is None:
        config_kwargs = {}
    if depends_on_macros is None:
        depends_on_macros = []
    if fqn_extras is None:
        fqn_extras = []
    fqn = [pkg] + fqn_extras + [name]
    if version:
        fqn.append(f"v{version}")
    depends_on_nodes: List[str] = []
    source_values: List[List[str]] = []
    ref_values: List[RefArgs] = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, "version") else None
        ref_values.append(
            RefArgs(name=ref.name, package=ref.package_name, version=ref_version)
        )
        depends_on_nodes.append(ref.unique_id)
    for src in sources:
        source_values.append([src.source_name, src.name])
        depends_on_nodes.append(src.unique_id)
    return ModelNode(
        language=language,
        raw_code=code,
        database="dbt",
        schema="dbt_schema",
        alias=alias,
        name=name,
        fqn=fqn,
        unique_id=f"model.{pkg}.{name}" if not version else f"model.{pkg}.{name}.v{version}",
        package_name=pkg,
        path=path,
        original_file_path=f"models/{path}",
        config=NodeConfig(**config_kwargs),
        tags=tags,
        refs=ref_values,
        sources=source_values,
        depends_on=DependsOn(nodes=depends_on_nodes, macros=depends_on_macros),
        resource_type=NodeType.Model,
        checksum=FileHash.from_contents(""),
        version=version,
        latest_version=latest_version,
        access=access or AccessType.Protected,
        patch_path=patch_path,
    )


def func_4rgc9d6x(
    pkg: str,
    name: str,
    path: Optional[str] = None,
    loader: Optional[Any] = None,
    alias: Optional[str] = None,
    tags: Optional[List[str]] = None,
    fqn_extras: Optional[List[str]] = None,
    checksum: Optional[FileHash] = None,
) -> SeedNode:
    if alias is None:
        alias = name
    if tags is None:
        tags = []
    if path is None:
        path = f"{name}.csv"
    if fqn_extras is None:
        fqn_extras = []
    if checksum is None:
        checksum = FileHash.from_contents("")
    fqn = [pkg] + fqn_extras + [name]
    return SeedNode(
        database="dbt",
        schema="dbt_schema",
        alias=alias,
        name=name,
        fqn=fqn,
        unique_id=f"seed.{pkg}.{name}",
        package_name=pkg,
        path=path,
        original_file_path=f"data/{path}",
        tags=tags,
        resource_type=NodeType.Seed,
        checksum=checksum,
    )


def func_2zhjxsz6(
    pkg: str,
    source_name: str,
    table_name: str,
    path: Optional[str] = None,
    loader: Optional[str] = None,
    identifier: Optional[str] = None,
    fqn_extras: Optional[List[str]] = None,
) -> SourceDefinition:
    if path is None:
        path = "models/schema.yml"
    if loader is None:
        loader = "my_loader"
    if identifier is None:
        identifier = table_name
    if fqn_extras is None:
        fqn_extras = []
    fqn = [pkg] + fqn_extras + [source_name, table_name]
    return SourceDefinition(
        fqn=fqn,
        database="dbt",
        schema="dbt_schema",
        unique_id=f"source.{pkg}.{source_name}.{table_name}",
        package_name=pkg,
        path=path,
        original_file_path=path,
        name=table_name,
        source_name=source_name,
        loader=loader,
        identifier=identifier,
        resource_type=NodeType.Source,
        loaded_at_field="loaded_at",
        tags=[],
        source_description="",
    )


def func_dchet9jj(
    pkg: str,
    name: str,
    macro_sql: str,
    path: Optional[str] = None,
    depends_on_macros: Optional[List[str]] = None,
) -> Macro:
    if path is None:
        path = "macros/macros.sql"
    if depends_on_macros is None:
        depends_on_macros = []
    return Macro(
        name=name,
        macro_sql=macro_sql,
        unique_id=f"macro.{pkg}.{name}",
        package_name=pkg,
        path=path,
        original_file_path=path,
        resource_type=NodeType.Macro,
        depends_on=MacroDependsOn(macros=depends_on_macros),
    )


def func_c5tsisww(
    pkg: str,
    test_model: Union[ModelNode, SourceDefinition],
    column_name: str,
    path: Optional[str] = None,
    refs: Optional[List[Any]] = None,
    sources: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
) -> GenericTestNode:
    return make_generic_test(pkg, "unique", test_model, {}, column_name=column_name)


def func_7vzp55bg(
    pkg: str,
    test_model: Union[ModelNode, SourceDefinition],
    column_name: str,
    path: Optional[str] = None,
    refs: Optional[List[Any]] = None,
    sources: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
) -> GenericTestNode:
    return make_generic_test(pkg, "not_null", test_model, {}, column_name=column_name)


def func_dxr26jbf(
    pkg: str,
    test_name: str,
    test_model: Union[ModelNode, SourceDefinition],
    test_kwargs: Dict[str, Any],
    path: Optional[str] = None,
    refs: Optional[List[Any]] = None,
    sources: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
    column_name: Optional[str] = None,
) -> GenericTestNode:
    kwargs = test_kwargs.copy()
    ref_values: List[RefArgs] = []
    source_values: List[List[str]] = []
    if isinstance(test_model, SourceDefinition):
        kwargs["model"] = f"{{{{ source('{test_model.source_name}', '{test_model.name}') }}}}"
        source_values.append([test_model.source_name, test_model.name])
    else:
        kwargs["model"] = "{{ ref('" + test_model.name + "') }}"
        ref_values.append(
            RefArgs(name=test_model.name, package=test_model.package_name, version=test_model.version)
        )
    if column_name is not None:
        kwargs["column_name"] = column_name
    args_name = test_model.search_name.replace(".", "_")
    if column_name is not None:
        args_name += "_" + column_name
    node_name = f"{test_name}_{args_name}"
    raw_code = f"{{{{ config(severity=\"ERROR\") }}}}{{{{ test_{test_name}(**dbt_schema_test_kwargs) }}}}"
    name_parts = test_name.split(".")
    if len(name_parts) == 2:
        namespace, test_name = name_parts
        macro_depends = f"macro.{namespace}.test_{test_name}"
    elif len(name_parts) == 1:
        namespace = None
        macro_depends = f"macro.dbt.test_{test_name}"
    else:
        assert False, f"invalid test name: {test_name}"
    if path is None:
        path = "schema.yml"
    if tags is None:
        tags = ["schema"]
    if refs is None:
        refs = []
    if sources is None:
        sources = []
    depends_on_nodes: List[str] = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, "version") else None
        ref_values.append(
            RefArgs(name=ref.name, package=ref.package_name, version=ref_version)
        )
        depends_on_nodes.append(ref.unique_id)
    for source in sources:
        source_values.append([source.source_name, source.name])
        depends_on_nodes.append(source.unique_id)
    return GenericTestNode(
        language="sql",
        raw_code=raw_code,
        test_metadata=TestMetadata(namespace=namespace, name=test_name, kwargs=kwargs),
        database="dbt",
        schema="dbt_postgres",
        name=node_name,
        alias=node_name,
        fqn=["minimal", "schema_test", node_name],
        unique_id=f"test.{pkg}.{node_name}",
        package_name=pkg,
        path=f"schema_test/{node_name}.sql",
        original_file_path=f"models/{path}",
        resource_type=NodeType.Test,
        tags=tags,
        refs=ref_values,
        sources=[],
        depends_on=DependsOn(macros=[macro_depends], nodes=depends_on_nodes),
        column_name=column_name,
        checksum=FileHash.from_contents(""),
    )


def func_o1pqaaw0(
    pkg: str, test_name: str, test_model: ModelNode
) -> UnitTestDefinition:
    input_fixture = UnitTestInputFixture(
        input="ref('table_model')", rows=[{"id": 1, "string_a": "a"}]
    )
    output_fixture = UnitTestOutputFixture(rows=[{"id": 1, "string_a": "a"}])
    return UnitTestDefinition(
        name=test_name,
        model=test_model,
        package_name=pkg,
        resource_type=NodeType.Unit,
        path="unit_tests.yml",
        original_file_path="models/unit_tests.yml",
        unique_id=f"unit.{pkg}.{test_model.name}__{test_name}",
        given=[input_fixture],
        expect=output_fixture,
        fqn=[pkg, test_model.name, test_name],
    )


def func_ohecc7b1(
    pkg: str,
    name: str,
    sql: str,
    refs: Optional[List[Any]] = None,
    sources: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
    path: Optional[str] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> SingularTestNode:
    if refs is None:
        refs = []
    if sources is None:
        sources = []
    if tags is None:
        tags = ["data"]
    if path is None:
        path = f"{name}.sql"
    if config_kwargs is None:
        config_kwargs = {}
    fqn = ["minimal", "data_test", name]
    depends_on_nodes: List[str] = []
    source_values: List[List[str]] = []
    ref_values: List[RefArgs] = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, "version") else None
        ref_values.append(
            RefArgs(name=ref.name, package=ref.package_name, version=ref_version)
        )
        depends_on_nodes.append(ref.unique_id)
    for src in sources:
        source_values.append([src.source_name, src.name])
        depends_on_nodes.append(src.unique_id)
    return SingularTestNode(
        language="sql",
        raw_code=sql,
        database="dbt",
        schema="dbt_schema",
        name=name,
        alias=name,
        fqn=fqn,
        unique_id=f"test.{pkg}.{name}",
        package_name=pkg,
        path=path,
        original_file_path=f"tests/{path}",
        config=TestConfig(**config_kwargs),
        tags=tags,
        refs=ref_values,
        sources=source_values,
        depends_on=DependsOn(nodes=depends_on_nodes, macros=[]),
        resource_type=NodeType.Test,
        checksum=FileHash.from_contents(""),
    )


def func_9i1qsovo(
    pkg: str,
    name: str,
    path: Optional[str] = None,
    fqn_extras: Optional[List[str]] = None,
    owner: Optional[Owner] = None,
) -> Exposure:
    if path is None:
        path = "schema.yml"
    if fqn_extras is None:
        fqn_extras = []
    if owner is None:
        owner = Owner(email="test@example.com")
    fqn = [pkg, "exposures"] + fqn_extras + [name]
    return Exposure(
        name=name,
        resource_type=NodeType.Exposure,
        type=ExposureType.Notebook,
        fqn=fqn,
        unique_id=f"exposure.{pkg}.{name}",
        package_name=pkg,
        path=path,
        original_file_path=path,
        owner=owner,
    )


def func_hxvz6q30(pkg: str, name: str, path: Optional[str] = None) -> Metric:
    if path is None:
        path = "schema.yml"
    return Metric(
        name=name,
        resource_type=NodeType.Metric,
        path=path,
        package_name=pkg,
        original_file_path=path,
        unique_id=f"metric.{pkg}.{name}",
        fqn=[pkg, "metrics", name],
        label="New Customers",
        description="New customers",
        type=MetricType.SIMPLE,
        type_params=MetricTypeParams(measure=MetricInputMeasure(name="count_cats")),
        meta={"is_okr": True},
        tags=["okrs"],
    )


def func_fy3sct1r(pkg: str, name: str, path: Optional[str] = None) -> Group:
    if path is None:
        path = "schema.yml"
    return Group(
        name=name,
        resource_type=NodeType.Group,
        path=path,
        package_name=pkg,
        original_file_path=path,
        unique_id=f"group.{pkg}.{name}",
        owner="email@gmail.com",
    )


def func_t7xbdclq(
    pkg: str, name: str, model: ModelNode, path: Optional[str] = None
) -> SemanticModel:
    if path is None:
        path = "schema.yml"
    return SemanticModel(
        name=name,
        resource_type=NodeType.SemanticModel,
        model=model,
        node_relation=NodeRelation(
            alias=model.alias, schema_name="dbt", relation_name=model.name
        ),
        package_name=pkg,
        path=path,
        description="Customer entity",
        primary_entity="customer",
        unique_id=f"semantic_model.{pkg}.{name}",
        original_file_path=path,
        fqn=[pkg, "semantic_models", name],
    )


def func_ngh67n5g(pkg: str, name: str, metric: Metric, path: Optional[str] = None) -> SavedQuery:
    if path is None:
        path = "schema.yml"
    return SavedQuery(
        name=name,
        resource_type=NodeType.SavedQuery,
        package_name=pkg,
        path=path,
        description="Test Saved Query",
        query_params=QueryParams(metrics=[metric], group_by=[], where=None),
        exports=[],
        unique_id=f"saved_query.{pkg}.{name}",
        original_file_path=path,
        fqn=[pkg, "saved_queries", name],
    )


@pytest.fixture
def func_4azinwch() -> Macro:
    return func_dchet9jj(
        "dbt",
        "test_unique",
        "blablabla",
        depends_on_macros=["macro.dbt.default__test_unique"],
    )


@pytest.fixture
def func_rzep8lqx() -> Macro:
    return func_dchet9jj("dbt", "default__test_unique", "blablabla")


@pytest.fixture
def func_wzzeu4ni() -> Macro:
    return func_dchet9jj(
        "dbt",
        "test_not_null",
        "blablabla",
        depends_on_macros=["macro.dbt.default__test_not_null"],
    )


@pytest.fixture
def func_5k1z1a8v() -> Macro:
    macro = func_dchet9jj("dbt", "materialization_table_default", "SELECT 1")
    macro.supported_languages = [ModelLanguage.sql]
    return macro


@pytest.fixture
def func_snt02wkq() -> Macro:
    return func_dchet9jj("dbt", "default__test_not_null", "blabla")


@pytest.fixture
def func_pzpz77h9() -> SeedNode:
    return func_4rgc9d6x("pkg", "seed")


@pytest.fixture
def func_6qkiwdyx() -> SourceDefinition:
    return func_2zhjxsz6("pkg", "raw", "seed", identifier="seed")


@pytest.fixture
def func_cygdhbxf(source: SourceDefinition) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "ephemeral_model",
        "select * from {{ source(\"raw\", \"seed\") }}",
        config_kwargs={"materialized": "ephemeral"},
        sources=[source],
    )


@pytest.fixture
def func_q3z8y58d(ephemeral_model: ModelNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "view_model",
        "select * from {{ ref(\"ephemeral_model\") }}",
        config_kwargs={"materialized": "view"},
        refs=[ephemeral_model],
        tags=["uses_ephemeral"],
    )


@pytest.fixture
def func_brgdergm(ephemeral_model: ModelNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "table_model",
        "select * from {{ ref(\"ephemeral_model\") }}",
        config_kwargs={
            "materialized": "table",
            "meta": {
                "string_property": "some_string",
                "truthy_bool_property": True,
                "falsy_bool_property": False,
                "list_property": ["some_value", True, False],
            },
        },
        refs=[ephemeral_model],
        tags=["uses_ephemeral"],
        path="subdirectory/table_model.sql",
    )


@pytest.fixture
def func_h3njjxpm(seed: SeedNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "table_model_py",
        "select * from {{ ref(\"seed\") }}",
        config_kwargs={"materialized": "table"},
        refs=[seed],
        tags=[],
        path="subdirectory/table_model.py",
    )


@pytest.fixture
def func_z4zvk8h4(seed: SeedNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "table_model_csv",
        "select * from {{ ref(\"seed\") }}",
        config_kwargs={"materialized": "table"},
        refs=[seed],
        tags=[],
        path="subdirectory/table_model.csv",
    )


@pytest.fixture
def func_lc9zzbtm() -> SourceDefinition:
    return func_2zhjxsz6("ext", "ext_raw", "ext_source")


@pytest.fixture
def func_a9ai55kg() -> SourceDefinition:
    return func_2zhjxsz6("ext", "ext_raw", "ext_source_2")


@pytest.fixture
def func_n7cj2n0w() -> SourceDefinition:
    return func_2zhjxsz6("ext", "raw", "ext_source")


@pytest.fixture
def func_bfml6huc() -> SourceDefinition:
    return func_2zhjxsz6("ext", "raw", "ext_source_2")


@pytest.fixture
def func_oqe5h0cx(ext_source: SourceDefinition) -> ModelNode:
    return func_vhyebr8c(
        "ext",
        "ext_model",
        'select * from {{ source("ext_raw", "ext_source") }}',
        sources=[ext_source],
    )


@pytest.fixture
def func_lkpbp645(seed: SeedNode, ext_source: SourceDefinition) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "union_model",
        'select * from {{ ref("seed") }} union all select * from {{ source("ext_raw", "ext_source") }}',
        config_kwargs={"materialized": "table"},
        refs=[seed],
        sources=[ext_source],
        fqn_extras=["unions"],
        path="subdirectory/union_model.sql",
        tags=["unions"],
    )


@pytest.fixture
def func_ajdwuq90(seed: SeedNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "versioned_model",
        'select * from {{ ref("seed") }}',
        config_kwargs={"materialized": "table"},
        refs=[seed],
        sources=[],
        path="subdirectory/versioned_model_v1.sql",
        version=1,
        latest_version=2,
    )


@pytest.fixture
def func_896pask9(seed: SeedNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "versioned_model",
        'select * from {{ ref("seed") }}',
        config_kwargs={"materialized": "table"},
        refs=[seed],
        sources=[],
        path="subdirectory/versioned_model_v2.sql",
        version=2,
        latest_version=2,
    )


@pytest.fixture
def func_0q8rpv6x(seed: SeedNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "versioned_model",
        'select * from {{ ref("seed") }}',
        config_kwargs={"materialized": "table"},
        refs=[seed],
        sources=[],
        path="subdirectory/versioned_model_v3.sql",
        version="3",
        latest_version=2,
    )


@pytest.fixture
def func_cc25lnu2(seed: SeedNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "versioned_model",
        'select * from {{ ref("seed") }}',
        config_kwargs={"materialized": "table"},
        refs=[seed],
        sources=[],
        path="subdirectory/versioned_model_v12.sql",
        version="12",
        latest_version=2,
    )


@pytest.fixture
def func_1sdz54f9(seed: SeedNode) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "versioned_model",
        'select * from {{ ref("seed") }}',
        config_kwargs={"materialized": "table"},
        refs=[seed],
        sources=[],
        path="subdirectory/nested_dir/versioned_model_v3.sql",
        version=4,
        latest_version=2,
        fqn_extras=["nested_dir"],
    )


@pytest.fixture
def func_sgj3ab71(table_model: ModelNode) -> GenericTestNode:
    return func_c5tsisww("pkg", table_model, "id")


@pytest.fixture
def func_0n66mhon(table_model: ModelNode) -> GenericTestNode:
    return func_7vzp55bg("pkg", table_model, "id")


@pytest.fixture
def func_vmdgd1oc(view_model: ModelNode) -> GenericTestNode:
    return func_c5tsisww("pkg", view_model, "id")


@pytest.fixture
def func_m5myaaq7(ext_source: SourceDefinition) -> GenericTestNode:
    return func_c5tsisww("ext", ext_source, "id")


@pytest.fixture
def func_ucyoiiub(view_model: ModelNode) -> SingularTestNode:
    return func_ohecc7b1(
        "pkg",
        "view_test_nothing",
        'select * from {{ ref("view_model") }} limit 0',
        refs=[view_model],
    )


@pytest.fixture
def func_ftwznn7e(table_model: ModelNode) -> UnitTestDefinition:
    return func_o1pqaaw0("pkg", "unit_test_table_model", table_model)


@pytest.fixture
def func_snn9lhf3() -> SeedNode:
    return func_4rgc9d6x("pkg", "mynamespace.seed")


@pytest.fixture
def func_rkwje9it(source: SourceDefinition) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "mynamespace.ephemeral_model",
        'select * from {{ source("raw", "seed") }}',
        config_kwargs={"materialized": "ephemeral"},
        sources=[source],
    )


@pytest.fixture
def func_7kmk9qy4(
    seed: SeedNode, ext_source: SourceDefinition
) -> ModelNode:
    return func_vhyebr8c(
        "pkg",
        "mynamespace.union_model",
        'select * from {{ ref("mynamespace.seed") }} union all select * from {{ ref("mynamespace.ephemeral_model") }}',
        config_kwargs={"materialized": "table"},
        refs=[seed],
        sources=[ext_source],
        fqn_extras=["unions"],
        path="subdirectory/union_model.sql",
        tags=["unions"],
    )


@pytest.fixture
def func_x4clvb05() -> Metric:
    return Metric(
        name="my_metric",
        resource_type=NodeType.Metric,
        type=MetricType.SIMPLE,
        type_params=MetricTypeParams(measure=MetricInputMeasure(name="a_measure")),
        fqn=["test", "metrics", "myq_metric"],
        unique_id="metric.test.my_metric",
        package_name="test",
        path="models/metric.yml",
        original_file_path="models/metric.yml",
        description="",
        meta={},
        tags=[],
        label="test_label",
    )


@pytest.fixture
def func_nfuqtbfr() -> SavedQuery:
    pkg = "test"
    name = "test_saved_query"
    path = "test_path"
    return SavedQuery(
        name=name,
        resource_type=NodeType.SavedQuery,
        package_name=pkg,
        path=path,
        description="Test Saved Query",
        query_params=QueryParams(
            metrics=["my_metric"],
            group_by=[],
            where=WhereFilterIntersection(
                where_filters=[WhereFilter(where_sql_template="1=1")]
            ),
        ),
        exports=[],
        unique_id=f"saved_query.{pkg}.{name}",
        original_file_path=path,
        fqn=[pkg, "saved_queries", name],
    )


@pytest.fixture
def func_bie1dh7f(table_model: ModelNode) -> SemanticModel:
    return func_t7xbdclq("test", "test_semantic_model", model=table_model)


@pytest.fixture
def func_ulvbd2s2() -> ModelNode:
    return ModelNode(
        name="metricflow_time_spine",
        database="dbt",
        schema="analytics",
        alias="events",
        resource_type=NodeType.Model,
        unique_id="model.test.metricflow_time_spine",
        fqn=["snowplow", "events"],
        package_name="snowplow",
        refs=[],
        sources=[],
        metrics=[],
        depends_on=DependsOn(),
        config=ModelConfig(),
        tags=[],
        path="events.sql",
        original_file_path="events.sql",
        meta={},
        language="sql",
        raw_code="does not matter",
        checksum=FileHash.empty(),
        relation_name="events",
    )


@pytest.fixture
def func_0zxjei7p(
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
    namespace_model: SemanticModel,
    namespaced_union_model: ModelNode,
) -> List[Union[
    SeedNode,
    ModelNode,
    GenericTestNode,
    SingularTestNode,
    SemanticModel,
]]:
    return [
        seed,
        ephemeral_model,
        view_model,
        table_model,
        table_model_py,
        table_model_csv,
        union_model,
        versioned_model_v1,
        versioned_model_v2,
        versioned_model_v3,
        versioned_model_v4_nested_dir,
        versioned_model_v12_string,
        ext_model,
        table_id_unique,
        table_id_not_null,
        view_id_unique,
        ext_source_id_unique,
        view_test_nothing,
        namespaced_seed,
        namespace_model,
        namespaced_union_model,
    ]


@pytest.fixture
def func_z9mky3ax(
    source: SourceDefinition,
    ext_source: SourceDefinition,
    ext_source_2: SourceDefinition,
    ext_source_other: SourceDefinition,
    ext_source_other_2: SourceDefinition,
) -> List[SourceDefinition]:
    return [
        source,
        ext_source,
        ext_source_2,
        ext_source_other,
        ext_source_other_2,
    ]


@pytest.fixture
def func_gbu38kpb(
    macro_test_unique: Macro,
    macro_default_test_unique: Macro,
    macro_test_not_null: Macro,
    macro_default_test_not_null: Macro,
    macro_materialization_table_default: Macro,
) -> List[Macro]:
    return [
        macro_test_unique,
        macro_default_test_unique,
        macro_test_not_null,
        macro_default_test_not_null,
        macro_materialization_table_default,
    ]


@pytest.fixture
def func_c1lkzna8(unit_test_table_model: UnitTestDefinition) -> List[UnitTestDefinition]:
    return [unit_test_table_model]


@pytest.fixture
def func_vrf8etir(metric: Metric) -> List[Metric]:
    return [metric]


@pytest.fixture
def func_bl90lacw(semantic_model: SemanticModel) -> List[SemanticModel]:
    return [semantic_model]


@pytest.fixture
def func_xuabh7rj(saved_query: SavedQuery) -> List[SavedQuery]:
    return [saved_query]


@pytest.fixture
def func_06bh136w() -> Dict[str, Any]:
    return {}


def func_i60134mh(
    disabled: Dict[str, Any] = {},
    docs: List[Documentation] = [],
    exposures: List[Exposure] = [],
    files: Dict[str, Any] = {},
    groups: List[Group] = [],
    macros: List[Macro] = [],
    metrics: List[Metric] = [],
    nodes: List[Union[ModelNode, SeedNode, GenericTestNode, SingularTestNode]] = [],
    saved_queries: List[SavedQuery] = [],
    selectors: Dict[str, Any] = {},
    semantic_models: List[SemanticModel] = [],
    sources: List[SourceDefinition] = [],
    unit_tests: List[UnitTestDefinition] = [],
) -> Manifest:
    manifest = Manifest(
        nodes={n.unique_id: n for n in nodes},
        sources={s.unique_id: s for s in sources},
        macros={m.unique_id: m for m in macros},
        unit_tests={t.unique_id: t for t in unit_tests},
        semantic_models={s.unique_id: s for s in semantic_models},
        docs={d.unique_id: d for d in docs},
        files=files,
        exposures={e.unique_id: e for e in exposures},
        metrics={m.unique_id: m for m in metrics},
        disabled=disabled,
        selectors=selectors,
        groups={g.unique_id: g for g in groups},
        metadata=ManifestMetadata(adapter_type="postgres", project_name="pkg"),
        saved_queries={s.unique_id: s for s in saved_queries},
    )
    manifest.build_parent_and_child_maps()
    return manifest


@pytest.fixture
def func_umogu1ad(
    metric: Metric,
    semantic_model: SemanticModel,
    nodes: List[Union[ModelNode, SeedNode, GenericTestNode, SingularTestNode]],
    sources: List[SourceDefinition],
    macros: List[Macro],
    unit_tests: List[UnitTestDefinition],
    metrics: List[Metric],
    semantic_models: List[SemanticModel],
    files: Dict[str, Any],
    saved_queries: List[SavedQuery],
) -> Manifest:
    return func_i60134mh(
        nodes=nodes,
        sources=sources,
        macros=macros,
        unit_tests=unit_tests,
        semantic_models=semantic_models,
        files=files,
        metrics=metrics,
        saved_queries=saved_queries,
    )
