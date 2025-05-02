import json
import os
import shutil
import pytest
from dbt.artifacts.exceptions import IncompatibleSchemaError
from dbt.artifacts.schemas.base import get_artifact_schema_version
from dbt.artifacts.schemas.run import RunResultsArtifact
from dbt.contracts.graph.manifest import WritableManifest
from dbt.tests.util import get_manifest, run_dbt
from typing import Dict, Any, List

models__my_model_sql: str = '\nselect 1 as id\n'
models__disabled_model_sql: str = '\n{{ config(enabled=False) }}\nselect 2 as id\n'
seeds__my_seed_csv: str = '\nid,value\n4,2\n'
seeds__disabled_seed_csv: str = '\nid,value\n6,4\n'
docs__somedoc_md: str = '\n{% docs somedoc %}\nTesting, testing\n{% enddocs %}\n'
macros__do_nothing_sql: str = "\n{% macro do_nothing(foo2, bar2) %}\n    select\n        '{{ foo2 }}' as foo2,\n        '{{ bar2 }}' as bar2\n{% endmacro %}\n"
macros__dummy_test_sql: str = '\n{% test check_nothing(model) %}\n-- a silly test to make sure that table-level tests show up in the manifest\n-- without a column_name field\n\nselect 0\n\n{% endtest %}\n'
macros__disabled_dummy_test_sql: str = '\n{% test disabled_check_nothing(model) %}\n-- a silly test to make sure that table-level tests show up in the manifest\n-- without a column_name field\n\n{{ config(enabled=False) }}\nselect 0\n\n{% endtest %}\n'
snapshot__snapshot_seed_sql: str = "\n{% snapshot snapshot_seed %}\n{{\n    config(\n      unique_key='id',\n      strategy='check',\n      check_cols='all',\n      target_schema=schema,\n    )\n}}\nselect * from {{ ref('my_seed') }}\n{% endsnapshot %}\n"
snapshot__disabled_snapshot_seed_sql: str = "\n{% snapshot disabled_snapshot_seed %}\n{{\n    config(\n      unique_key='id',\n      strategy='check',\n      check_cols='all',\n      target_schema=schema,\n      enabled=False,\n    )\n}}\nselect * from {{ ref('my_seed') }}\n{% endsnapshot %}\n"
tests__just_my_sql: str = "\n{{ config(tags = ['data_test_tag']) }}\n\nselect * from {{ ref('my_model') }}\nwhere false\n"
tests__disabled_just_my_sql: str = "\n{{ config(enabled=False) }}\n\nselect * from {{ ref('my_model') }}\nwhere false\n"
analyses__a_sql: str = '\nselect 4 as id\n'
analyses__disabled_a_sql: str = '\n{{ config(enabled=False) }}\nselect 9 as id\n'
metricflow_time_spine_sql: str = "\nSELECT to_date('02/20/2023', 'mm/dd/yyyy') as date_day\n"
models__schema_yml: str = '\nversion: 2\nmodels:\n  - name: my_model\n    description: "Example model"\n    data_tests:\n      - check_nothing\n      - disabled_check_nothing\n    columns:\n     - name: id\n       data_tests:\n       - not_null\n\nsemantic_models:\n  - name: semantic_people\n    model: ref(\'my_model\')\n    dimensions:\n      - name: favorite_color\n        type: categorical\n      - name: created_at\n        type: TIME\n        type_params:\n          time_granularity: day\n    measures:\n      - name: years_tenure\n        agg: SUM\n        expr: tenure\n      - name: people\n        agg: count\n        expr: id\n      - name: customers\n        agg: count\n        expr: id\n    entities:\n      - name: id\n        type: primary\n    defaults:\n      agg_time_dimension: created_at\n\nmetrics:\n  - name: blue_customers_post_2010\n    label: Blue Customers since 2010\n    type: simple\n    filter: "{{ TimeDimension(\'id__created_at\', \'day\') }} > \'2010-01-01\'"\n    type_params:\n      measure:\n        name: customers\n        filter: "{{ Dimension(\'id__favorite_color\') }} = \'blue\'"\n  - name: customers\n    label: Customers Metric\n    type: simple\n    type_params:\n      measure: customers\n  - name: disabled_metric\n    label: Count records\n    config:\n        enabled: False\n    filter: "{{ Dimension(\'id__favorite_color\') }} = \'blue\'"\n    type: simple\n    type_params:\n      measure: customers\n  - name: ratio_of_blue_customers_to_red_customers\n    label: Very Important Customer Color Ratio\n    type: ratio\n    type_params:\n      numerator:\n        name: customers\n        filter: "{{ Dimension(\'id__favorite_color\')}} = \'blue\'"\n      denominator:\n        name: customers\n        filter: "{{ Dimension(\'id__favorite_color\')}} = \'red\'"\n  - name: doubled_blue_customers\n    type: derived\n    label: Inflated blue customer numbers\n    type_params:\n      expr: \'customers * 2\'\n      metrics:\n        - name: customers\n          filter: "{{ Dimension(\'id__favorite_color\')}} = \'blue\'"\n\n\nsources:\n  - name: my_source\n    description: "My source"\n    loader: a_loader\n    tables:\n      - name: my_table\n        description: "My table"\n        identifier: my_seed\n      - name: disabled_table\n        description: "Disabled table"\n        config:\n           enabled: False\n\nexposures:\n  - name: simple_exposure\n    type: dashboard\n    depends_on:\n      - ref(\'my_model\')\n      - source(\'my_source\', \'my_table\')\n    owner:\n      email: something@example.com\n  - name: disabled_exposure\n    type: dashboard\n    config:\n      enabled: False\n    depends_on:\n      - ref(\'my_model\')\n    owner:\n      email: something@example.com\n\nseeds:\n  - name: disabled_seed\n    config:\n      enabled: False\n'

class TestPreviousVersionState:
    CURRENT_EXPECTED_MANIFEST_VERSION: int = 12
    CURRENT_EXPECTED_RUN_RESULTS_VERSION: int = 6

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': models__my_model_sql, 'schema.yml': models__schema_yml, 'somedoc.md': docs__somedoc_md, 'disabled_model.sql': models__disabled_model_sql, 'metricflow_time_spine.sql': metricflow_time_spine_sql}

    @pytest.fixture(scope='class')
    def seeds(self) -> Dict[str, str]:
        return {'my_seed.csv': seeds__my_seed_csv, 'disabled_seed.csv': seeds__disabled_seed_csv}

    @pytest.fixture(scope='class')
    def snapshots(self) -> Dict[str, str]:
        return {'snapshot_seed.sql': snapshot__snapshot_seed_sql, 'disabled_snapshot_seed.sql': snapshot__disabled_snapshot_seed_sql}

    @pytest.fixture(scope='class')
    def tests(self) -> Dict[str, str]:
        return {'just_my.sql': tests__just_my_sql, 'disabled_just_my.sql': tests__disabled_just_my_sql}

    @pytest.fixture(scope='class')
    def macros(self) -> Dict[str, str]:
        return {'do_nothing.sql': macros__do_nothing_sql, 'dummy_test.sql': macros__dummy_test_sql, 'disabled_dummy_test.sql': macros__disabled_dummy_test_sql}

    @pytest.fixture(scope='class')
    def analyses(self) -> Dict[str, str]:
        return {'a.sql': analyses__a_sql, 'disabled_al.sql': analyses__disabled_a_sql}

    def test_project(self, project: Any) -> None:
        results: List[Any] = run_dbt(['run'])
        assert len(results) == 2
        manifest: Any = get_manifest(project.project_root)
        assert len(manifest.nodes) == 8
        assert len(manifest.sources) == 1
        assert len(manifest.exposures) == 1
        assert len(manifest.metrics) == 4
        assert len(manifest.disabled) == 9
        assert 'macro.test.do_nothing' in manifest.macros

    def generate_latest_manifest(self, project: Any, current_manifest_version: int) -> None:
        run_dbt(['parse'])
        source_path: str = os.path.join(project.project_root, 'target/manifest.json')
        state_path: str = os.path.join(project.test_data_dir, f'state/v{current_manifest_version}')
        target_path: str = os.path.join(state_path, 'manifest.json')
        os.makedirs(state_path, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    def generate_latest_run_results(self, project: Any, current_run_results_version: int) -> None:
        run_dbt(['run'])
        source_path: str = os.path.join(project.project_root, 'target/run_results.json')
        state_path: str = os.path.join(project.test_data_dir, f'results/v{current_run_results_version}')
        target_path: str = os.path.join(state_path, 'run_results.json')
        os.makedirs(state_path, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    def compare_previous_state(self, project: Any, compare_manifest_version: int, expect_pass: bool, num_results: int) -> None:
        state_path: str = os.path.join(project.test_data_dir, f'state/v{compare_manifest_version}')
        cli_args: List[str] = ['list', '--resource-types', 'model', '--select', 'state:modified', '--state', state_path]
        if expect_pass:
            results: List[Any] = run_dbt(cli_args, expect_pass=expect_pass)
            assert len(results) == num_results
        else:
            with pytest.raises(IncompatibleSchemaError):
                run_dbt(cli_args, expect_pass=expect_pass)

    def compare_previous_results(self, project: Any, compare_run_results_version: int, expect_pass: bool, num_results: int) -> None:
        state_path: str = os.path.join(project.test_data_dir, f'results/v{compare_run_results_version}')
        cli_args: List[str] = ['retry', '--state', state_path]
        if expect_pass:
            results: List[Any] = run_dbt(cli_args, expect_pass=expect_pass)
            assert len(results) == num_results
        else:
            with pytest.raises(IncompatibleSchemaError):
                run_dbt(cli_args, expect_pass=expect_pass)

    def test_compare_state_current(self, project: Any) -> None:
        current_manifest_schema_version: int = WritableManifest.dbt_schema_version.version
        assert current_manifest_schema_version == self.CURRENT_EXPECTED_MANIFEST_VERSION, "Sounds like you've bumped the manifest version and need to update this test!"
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
            manifest_path: str = os.path.join(project.test_data_dir, f'state/v{schema_version}/manifest.json')
            manifest: Dict[str, Any] = json.load(open(manifest_path))
            manifest_version: int = get_artifact_schema_version(manifest)
            assert manifest_version == schema_version

    def test_compare_results_current(self, project: Any) -> None:
        current_run_results_schema_version: int = RunResultsArtifact.dbt_schema_version.version
        assert current_run_results_schema_version == self.CURRENT_EXPECTED_RUN_RESULTS_VERSION, "Sounds like you've bumped the run_results version and need to update this test!"
        self.compare_previous_results(project, current_run_results_schema_version, True, 0)

    def test_backwards_compatible_run_results_versions(self, project: Any) -> None:
        for schema_version in range(4, self.CURRENT_EXPECTED_RUN_RESULTS_VERSION):
            self.compare_previous_results(project, schema_version, True, 0)
