from typing import Dict, Any, Union
from pathlib import Path
import pytest
from dbt.tests.fixtures.project import write_project_files

snapshots__snapshot_sql: str = (
    "\n{% snapshot my_snapshot %}\n    {{\n        config(\n            target_database=var('target_database', database),\n            target_schema=schema,\n            unique_key='id',\n            strategy='timestamp',\n            updated_at='updated_at',\n        )\n    }}\n    select * from {{database}}.{{schema}}.seed\n{% endsnapshot %}\n\n"
)
tests__t_sql: str = "\nselect 1 as id limit 0\n\n"
models__schema_yml: str = (
    "\nversion: 2\nmodels:\n  - name: outer\n    description: The outer table\n    columns:\n      - name: id\n        description: The id value\n        data_tests:\n          - unique\n          - not_null\n\nsources:\n  - name: my_source\n    tables:\n      - name: my_table\n\n"
)
models__ephemeral_sql: str = (
    "\n\n{{ config(materialized='ephemeral') }}\n\nselect\n  1 as id,\n  {{ dbt.date_trunc('day', dbt.current_timestamp()) }} as created_at\n\n"
)
models__metric_flow: str = (
    "\n\nselect\n  {{ dbt.date_trunc('day', dbt.current_timestamp()) }} as date_day\n\n"
)
models__incremental_sql: str = (
    "\n{{\n  config(\n    materialized = \"incremental\",\n    incremental_strategy = \"delete+insert\",\n  )\n}}\n\nselect * from {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where a > (select max(a) from {{this}})\n{% endif %}\n\n"
)
models__docs_md: str = "\n{% docs my_docs %}\n  some docs\n{% enddocs %}\n\n"
models__outer_sql: str = "\nselect * from {{ ref('ephemeral') }}\n\n"
models__sub__inner_sql: str = "\nselect * from {{ ref('outer') }}\n\n"
macros__macro_stuff_sql: str = (
    "\n{% macro cool_macro() %}\n  wow!\n{% endmacro %}\n\n{% macro other_cool_macro(a, b) %}\n  cool!\n{% endmacro %}\n\n"
)
seeds__seed_csv: str = "a,b\n1,2\n"
analyses__a_sql: str = "\nselect 4 as id\n\n"
semantic_models__sm_yml: str = (
    "\nsemantic_models:\n  - name: my_sm\n    model: ref('outer')\n    defaults:\n      agg_time_dimension: created_at\n    entities:\n      - name: my_entity\n        type: primary\n        expr: id\n    dimensions:\n      - name: created_at\n        type: time\n        type_params:\n          time_granularity: day\n    measures:\n      - name: total_outer_count\n        agg: count\n        expr: 1\n\n"
)
metrics__m_yml: str = (
    "\nmetrics:\n  - name: total_outer\n    type: simple\n    description: The total count of outer\n    label: Total Outer\n    type_params:\n      measure: total_outer_count\n"
)
saved_queries__sq_yml: str = (
    "\nsaved_queries:\n  - name: my_saved_query\n    label: My Saved Query\n    query_params:\n        metrics:\n            - total_outer\n        group_by:\n            - \"Dimension('my_entity__created_at')\"\n    exports:\n        - name: my_export\n          config:\n            alias: my_export_alias\n            export_as: table\n            schema: my_export_schema_name\n"
)

@pytest.fixture(scope="class")
def snapshots() -> Dict[str, str]:
    return {"snapshot.sql": snapshots__snapshot_sql}

@pytest.fixture(scope="class")
def tests() -> Dict[str, str]:
    return {"t.sql": tests__t_sql}

@pytest.fixture(scope="class")
def models() -> Dict[str, Union[str, Dict[str, str]]]:
    return {
        "schema.yml": models__schema_yml,
        "ephemeral.sql": models__ephemeral_sql,
        "incremental.sql": models__incremental_sql,
        "docs.md": models__docs_md,
        "outer.sql": models__outer_sql,
        "metricflow_time_spine.sql": models__metric_flow,
        "sq.yml": saved_queries__sq_yml,
        "sm.yml": semantic_models__sm_yml,
        "m.yml": metrics__m_yml,
        "sub": {"inner.sql": models__sub__inner_sql},
    }

@pytest.fixture(scope="class")
def macros() -> Dict[str, str]:
    return {"macro_stuff.sql": macros__macro_stuff_sql}

@pytest.fixture(scope="class")
def seeds() -> Dict[str, str]:
    return {"seed.csv": seeds__seed_csv}

@pytest.fixture(scope="class")
def analyses() -> Dict[str, str]:
    return {"a.sql": analyses__a_sql}

@pytest.fixture(scope="class")
def semantic_models() -> Dict[str, str]:
    return {"sm.yml": semantic_models__sm_yml}

@pytest.fixture(scope="class")
def metrics() -> Dict[str, str]:
    return {"m.yml": metrics__m_yml}

@pytest.fixture(scope="class")
def saved_queries() -> Dict[str, str]:
    return {"sq.yml": saved_queries__sq_yml}

@pytest.fixture(scope="class")
def project_files(
    project_root: Path,
    snapshots: Dict[str, str],
    tests: Dict[str, str],
    models: Dict[str, Union[str, Dict[str, str]]],
    macros: Dict[str, str],
    seeds: Dict[str, str],
    analyses: Dict[str, str],
) -> None:
    write_project_files(project_root, "snapshots", snapshots)
    write_project_files(project_root, "tests", tests)
    write_project_files(project_root, "models", models)
    write_project_files(project_root, "macros", macros)
    write_project_files(project_root, "seeds", seeds)
    write_project_files(project_root, "analyses", analyses)