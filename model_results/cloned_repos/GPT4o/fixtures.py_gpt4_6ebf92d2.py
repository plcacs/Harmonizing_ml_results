```python
import pytest
from typing import Dict, Any

from dbt.tests.fixtures.project import write_project_files

snapshots__snapshot_sql: str = """
{% snapshot my_snapshot %}
    {{
        config(
            target_database=var('target_database', database),
            target_schema=schema,
            unique_key='id',
            strategy='timestamp',
            updated_at='updated_at',
        )
    }}
    select * from {{database}}.{{schema}}.seed
{% endsnapshot %}

"""

tests__t_sql: str = """
select 1 as id limit 0

"""

models__schema_yml: str = """
version: 2
models:
  - name: outer
    description: The outer table
    columns:
      - name: id
        description: The id value
        data_tests:
          - unique
          - not_null

sources:
  - name: my_source
    tables:
      - name: my_table

"""

models__ephemeral_sql: str = """

{{ config(materialized='ephemeral') }}

select
  1 as id,
  {{ dbt.date_trunc('day', dbt.current_timestamp()) }} as created_at

"""

models__metric_flow: str = """

select
  {{ dbt.date_trunc('day', dbt.current_timestamp()) }} as date_day

"""

models__incremental_sql: str = """
{{
  config(
    materialized = "incremental",
    incremental_strategy = "delete+insert",
  )
}}

select * from {{ ref('seed') }}

{% if is_incremental() %}
    where a > (select max(a) from {{this}})
{% endif %}

"""

models__docs_md: str = """
{% docs my_docs %}
  some docs
{% enddocs %}

"""

models__outer_sql: str = """
select * from {{ ref('ephemeral') }}

"""

models__sub__inner_sql: str = """
select * from {{ ref('outer') }}

"""

macros__macro_stuff_sql: str = """
{% macro cool_macro() %}
  wow!
{% endmacro %}

{% macro other_cool_macro(a, b) %}
  cool!
{% endmacro %}

"""

seeds__seed_csv: str = """a,b
1,2
"""

analyses__a_sql: str = """
select 4 as id

"""

semantic_models__sm_yml: str = """
semantic_models:
  - name: my_sm
    model: ref('outer')
    defaults:
      agg_time_dimension: created_at
    entities:
      - name: my_entity
        type: primary
        expr: id
    dimensions:
      - name: created_at
        type: time
        type_params:
          time_granularity: day
    measures:
      - name: total_outer_count
        agg: count
        expr: 1

"""

metrics__m_yml: str = """
metrics:
  - name: total_outer
    type: simple
    description: The total count of outer
    label: Total Outer
    type_params:
      measure: total_outer_count
"""


saved_queries__sq_yml: str = """
saved_queries:
  - name: my_saved_query
    label: My Saved Query
    query_params:
        metrics:
            - total_outer
        group_by:
            - "Dimension('my_entity__created_at')"
    exports:
        - name: my_export
          config:
            alias: my_export_alias
            export_as: table
            schema: my_export_schema_name
"""


@pytest.fixture(scope="class")
def snapshots() -> Dict[str, str]:
    return {"snapshot.sql": snapshots__snapshot_sql}


@pytest.fixture(scope="class")
def tests() -> Dict[str, str]:
    return {"t.sql": tests__t_sql}


@pytest.fixture(scope="class")
def models() -> Dict[str, Any]:
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
    project_root: str,
    snapshots: Dict[str, str],
    tests: Dict[str, str],
    models: Dict[str, Any],
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
```