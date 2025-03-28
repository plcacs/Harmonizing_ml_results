import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pytest

from dbt.artifacts.exceptions import IncompatibleSchemaError
from dbt.artifacts.schemas.base import get_artifact_schema_version
from dbt.artifacts.schemas.run import RunResultsArtifact
from dbt.contracts.graph.manifest import WritableManifest
from dbt.tests.util import get_manifest, run_dbt

models__my_model_sql: str = """
select 1 as id
"""

models__disabled_model_sql: str = """
{{ config(enabled=False) }}
select 2 as id
"""

seeds__my_seed_csv: str = """
id,value
4,2
"""

seeds__disabled_seed_csv: str = """
id,value
6,4
"""

docs__somedoc_md: str = """
{% docs somedoc %}
Testing, testing
{% enddocs %}
"""

macros__do_nothing_sql: str = """
{% macro do_nothing(foo2, bar2) %}
    select
        '{{ foo2 }}' as foo2,
        '{{ bar2 }}' as bar2
{% endmacro %}
"""

macros__dummy_test_sql: str = """
{% test check_nothing(model) %}
-- a silly test to make sure that table-level tests show up in the manifest
-- without a column_name field

select 0

{% endtest %}
"""

macros__disabled_dummy_test_sql: str = """
{% test disabled_check_nothing(model) %}
-- a silly test to make sure that table-level tests show up in the manifest
-- without a column_name field

{{ config(enabled=False) }}
select 0

{% endtest %}
"""

snapshot__snapshot_seed_sql: str = """
{% snapshot snapshot_seed %}
{{
    config(
      unique_key='id',
      strategy='check',
      check_cols='all',
      target_schema=schema,
    )
}}
select * from {{ ref('my_seed') }}
{% endsnapshot %}
"""

snapshot__disabled_snapshot_seed_sql: str = """
{% snapshot disabled_snapshot_seed %}
{{
    config(
      unique_key='id',
      strategy='check',
      check_cols='all',
      target_schema=schema,
      enabled=False,
    )
}}
select * from {{ ref('my_seed') }}
{% endsnapshot %}
"""

tests__just_my_sql: str = """
{{ config(tags = ['data_test_tag']) }}

select * from {{ ref('my_model') }}
where false
"""

tests__disabled_just_my_sql: str = """
{{ config(enabled=False) }}

select * from {{ ref('my_model') }}
where false
"""

analyses__a_sql: str = """
select 4 as id
"""

analyses__disabled_a_sql: str = """
{{ config(enabled=False) }}
select 9 as id
"""

metricflow_time_spine_sql: str = """
SELECT to_date('02/20/2023', 'mm/dd/yyyy') as date_day
"""

models__schema_yml: str = """
version: 2
models:
  - name: my_model
    description: "Example model"
    data_tests:
      - check_nothing
      - disabled_check_nothing
    columns:
     - name: id
       data_tests:
       - not_null

semantic_models:
  - name: semantic_people
    model: ref('my_model')
    dimensions:
      - name: favorite_color
        type: categorical
      - name: created_at
        type: TIME
        type_params:
          time_granularity: day
    measures:
      - name: years_tenure
        agg: SUM
        expr: tenure
      - name: people
        agg: count
        expr: id
      - name: customers
        agg: count
        expr: id
    entities:
      - name: id
        type: primary
    defaults:
      agg_time_dimension: created_at

metrics:
  - name: blue_customers_post_2010
    label: Blue Customers since 2010
    type: simple
    filter: "{{ TimeDimension('id__created_at', 'day') }} > '2010-01-01'"
    type_params:
      measure:
        name: customers
        filter: "{{ Dimension('id__favorite_color') }} = 'blue'"
  - name: customers
    label: Customers Metric
    type: simple
    type_params:
      measure: customers
  - name: disabled_metric
    label: Count records
    config:
        enabled: False
    filter: "{{ Dimension('id__favorite_color') }} = 'blue'"
    type: simple
    type_params:
      measure: customers
  - name: ratio_of_blue_customers_to_red_customers
    label: Very Important Customer Color Ratio
    type: ratio
    type_params:
      numerator:
        name: customers
        filter: "{{ Dimension('id__favorite_color')}} = 'blue'"
      denominator:
        name: customers
        filter: "{{ Dimension('id__favorite_color')}} = 'red'"
  - name: doubled_blue_customers
    type: derived
    label: Inflated blue customer numbers
    type_params:
      expr: 'customers * 2'
      metrics:
        - name: customers
          filter: "{{ Dimension('id__favorite_color')}} = 'blue'"


sources:
  - name: my_source
    description: "My source"
    loader: a_loader
    tables:
      - name: my_table
        description: "My table"
        identifier: my_seed
      - name: disabled_table
        description: "Disabled table"
        config:
           enabled: False

exposures:
  - name: simple_exposure
    type: dashboard
    depends_on:
      - ref('my_model')
      - source('my_source', 'my_table')
    owner:
      email: something@example.com
  - name: disabled_exposure
    type: dashboard
    config:
      enabled: False
    depends_on:
      - ref('my_model')
    owner:
      email: something@example.com

seeds:
  - name: disabled_seed
    config:
      enabled: False
"""

class TestPreviousVersionState:
    CURRENT_EXPECTED_MANIFEST_VERSION: int = 12
    CURRENT_EXPECTED_RUN_RESULTS_VERSION: int = 6

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {
            "my_model.sql": models__my_model_sql,
            "schema.yml": models__schema_yml,
            "somedoc.md": docs__somedoc_md,
            "disabled_model.sql": models__disabled_model_sql,
            "metricflow_time_spine.sql": metricflow_time_spine_sql,
        }

    @pytest.fixture(scope="class")
    def seeds(self) -> Dict[str, str]:
        return {
            "my_seed.csv": seeds__my_seed_csv,
            "disabled_seed.csv": seeds__disabled_seed_csv,
        }

    @pytest.fixture(scope="class")
    def snapshots(self) -> Dict[str, str]:
        return {
            "snapshot_seed.sql": snapshot__snapshot_seed_sql,
            "disabled_snapshot_seed.sql": snapshot__disabled_snapshot_seed_sql,
        }

    @pytest.fixture(scope="class")
    def tests(self) -> Dict[str, str]:
        return {
            "just_my.sql": tests__just_my_sql,
            "disabled_just_my.sql": tests__disabled_just_my_sql,
        }

    @pytest.fixture(scope="class")
    def macros(self) -> Dict[str, str]:
        return {
            "do_nothing.sql": macros__do_nothing_sql,
            "dummy_test.sql": macros__dummy_test_sql,
            "disabled_dummy_test.sql": macros__disabled_dummy_test_sql,
        }

    @pytest.fixture(scope="class")
    def analyses(self) -> Dict[str, str]:
        return {
            "a.sql": analyses__a_sql,
            "disabled_al.sql": analyses__disabled_a_sql,
        }

    def test_project(self, project: Any) -> None:
        results: List[Any] = run_dbt(["run"])
        assert len(results) == 2
        manifest: Any = get_manifest(project.project_root)
        assert len(manifest.nodes) == 8
        assert len(manifest.sources) == 1
        assert len(manifest.exposures) == 1
        assert len(manifest.metrics) == 4
        assert len(manifest.disabled) == 9
        assert "macro.test.do_nothing" in manifest.macros

    def generate_latest_manifest(
        self,
        project: Any,
        current_manifest_version: int,
    ) -> None:
        run_dbt(["parse"])
        source_path: str = os.path.join(project.project_root, "target/manifest.json")
        state_path: str = os.path.join(project.test_data_dir, f"state/v{current_manifest_version}")
        target_path: str = os.path.join(state_path, "manifest.json")
        os.makedirs(state_path, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    def generate_latest_run_results(
        self,
        project: Any,
        current_run_results_version: int,
    ) -> None:
        run_dbt(["run"])
        source_path: str = os.path.join(project.project_root, "target/run_results.json")
        state_path: str = os.path.join(project.test_data_dir, f"results/v{current_run_results_version}")
        target_path: str = os.path.join(state_path, "run_results.json")
        os.makedirs(state_path, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    def compare_previous_state(
        self,
        project: Any,
        compare_manifest_version: int,
        expect_pass: bool,
        num_results: int,
    ) -> None:
        state_path: str = os.path.join(project.test_data_dir, f"state/v{compare_manifest_version}")
        cli_args: List[str] = [
            "list",
            "--resource-types",
            "model",
            "--select",
            "state:modified",
            "--state",
            state_path,
        ]
        if expect_pass:
            results: List[Any] = run_dbt(cli_args, expect_pass=expect_pass)
            assert len(results) == num_results
        else:
            with pytest.raises(IncompatibleSchemaError):
                run_dbt(cli_args, expect_pass=expect_pass)

    def compare_previous_results(
        self,
        project: Any,
        compare_run_results_version: int,
        expect_pass: bool,
        num_results: int,
    ) -> None:
        state_path: str = os.path.join(project.test_data_dir, f"results/v{compare_run_results_version}")
        cli_args: List[str] = [
            "retry",
            "--state",
            state_path,
        ]
        if expect_pass:
            results: List[Any] = run_dbt(cli_args, expect_pass=expect_pass)
            assert len(results) == num_results
        else:
            with pytest.raises(IncompatibleSchemaError):
                run_dbt(cli_args, expect_pass=expect_pass)

    def test_compare_state_current(self, project: Any) -> None:
        current_manifest_schema_version: int = WritableManifest.dbt_schema_version.version
        assert (
            current_manifest_schema_version == self.CURRENT_EXPECTED_MANIFEST_VERSION
        ), "Sounds like you've bumped the manifest version and need to update this test!"
        self.compare_previous_state(project, current_manifest_schema_version, True, 0)

    def test_backwards_compatible_versions(self, project: Any) -> None:
        for schema_version in range(4, 10):
            self.compare_previous_state(project, schema_version, True, 1)
        for schema_version in range(10, self.CURRENT_EXPECTED_MANIFEST_VERSION):
            self.compare_previous_state(project, schema_version, True, 0)

    def test_nonbackwards_compatible_versions(self, project: Any) -> None:
        for schema_version in range(1, 4):
            self.compare_previous_state(project, schema_version, False, 0)

    def test_get_manifest_schema_version(self, project: Any) -> None:
        for schema_version in range(1, self.CURRENT_EXPECTED_MANIFEST_VERSION):
            manifest_path: str = os.path.join(
                project.test_data_dir, f"state/v{schema_version}/manifest.json"
            )
            manifest: Dict[str, Any] = json.load(open(manifest_path))

            manifest_version: int = get_artifact_schema_version(manifest)
            assert manifest_version == schema_version

    def test_compare_results_current(self, project: Any) -> None:
        current_run_results_schema_version: int = RunResultsArtifact.dbt_schema_version.version
        assert (
            current_run_results_schema_version == self.CURRENT_EXPECTED_RUN_RESULTS_VERSION
        ), "Sounds like you've bumped the run_results version and need to update this test!"
        self.compare_previous_results(project, current_run_results_schema_version, True, 0)

    def test_backwards_compatible_run_results_versions(self, project: Any) -> None:
        for schema_version in range(4, self.CURRENT_EXPECTED_RUN_RESULTS_VERSION):
            self.compare_previous_results(project, schema_version, True, 0)
