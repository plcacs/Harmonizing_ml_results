from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from unittest import mock
import pytest
from pytest_mock import MockerFixture
from dbt.events.types import (
    ArtifactWritten,
    EndOfRunSummary,
    GenericExceptionOnRun,
    InvalidConcurrentBatchesConfig,
    JinjaLogDebug,
    LogBatchResult,
    LogModelResult,
    MicrobatchExecutionDebug,
    MicrobatchMacroOutsideOfBatchesDeprecation,
    MicrobatchModelNoEventTimeInputs,
)
from dbt.tests.fixtures.project import TestProjInfo
from dbt.tests.util import (
    get_artifact,
    patch_microbatch_end_time,
    read_file,
    relation_from_name,
    run_dbt,
    run_dbt_and_capture,
    write_file,
)
from tests.utils import EventCatcher

input_model_sql: str = "\n{{ config(materialized='table', event_time='event_time') }}\n\nselect 1 as id, TIMESTAMP '2020-01-01 00:00:00-0' as event_time\nunion all\nselect 2 as id, TIMESTAMP '2020-01-02 00:00:00-0' as event_time\nunion all\nselect 3 as id, TIMESTAMP '2020-01-03 00:00:00-0' as event_time\n"
input_model_invalid_sql: str = "\n{{ config(materialized='table', event_time='event_time') }}\n\nselect invalid as event_time\n"
input_model_without_event_time_sql: str = "\n{{ config(materialized='table') }}\n\nselect 1 as id, TIMESTAMP '2020-01-01 00:00:00-0' as event_time\nunion all\nselect 2 as id, TIMESTAMP '2020-01-02 00:00:00-0' as event_time\nunion all\nselect 3 as id, TIMESTAMP '2020-01-03 00:00:00-0' as event_time\n"
microbatch_model_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\nselect * from {{ ref('input_model') }}\n"
microbatch_model_with_pre_and_post_sql: str = '\n{{ config(\n        materialized=\'incremental\',\n        incremental_strategy=\'microbatch\',\n        unique_key=\'id\',\n        event_time=\'event_time\',\n        batch_size=\'day\',\n        begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0),\n        pre_hook=\'{{log("execute: " ~ execute ~ ", pre-hook run by batch " ~ model.batch.id)}}\',\n        post_hook=\'{{log("execute: " ~ execute ~ ", post-hook run by batch " ~ model.batch.id)}}\',\n    )\n}}\nselect * from {{ ref(\'input_model\') }}\n'
microbatch_model_force_concurrent_batches_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0), concurrent_batches=true) }}\nselect * from {{ ref('input_model') }}\n"
microbatch_yearly_model_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='year', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\nselect * from {{ ref('input_model') }}\n"
microbatch_yearly_model_downstream_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='year', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\nselect * from {{ ref('microbatch_model') }}\n"
invalid_batch_jinja_context_macro_sql: str = '\n{% macro check_invalid_batch_jinja_context() %}\n\n{% if model is not mapping %}\n    {{ exceptions.raise_compiler_error("`model` is invalid: expected mapping type") }}\n{% elif compiled_code and compiled_code is not string %}\n    {{ exceptions.raise_compiler_error("`compiled_code` is invalid: expected string type") }}\n{% elif sql and sql is not string %}\n    {{ exceptions.raise_compiler_error("`sql` is invalid: expected string type") }}\n{% elif is_incremental is not callable %}\n    {{ exceptions.raise_compiler_error("`is_incremental()` is invalid: expected callable type") }}\n{% elif should_full_refresh is not callable %}\n    {{ exceptions.raise_compiler_error("`should_full_refresh()` is invalid: expected callable type") }}\n{% endif %}\n\n{% endmacro %}\n'
microbatch_model_with_context_checks_sql: str = '\n{{ config(pre_hook="{{ check_invalid_batch_jinja_context() }}", materialized=\'incremental\', incremental_strategy=\'microbatch\', unique_key=\'id\', event_time=\'event_time\', batch_size=\'day\', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\n\n{{ check_invalid_batch_jinja_context() }}\nselect * from {{ ref(\'input_model\') }}\n'
microbatch_model_downstream_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\nselect * from {{ ref('microbatch_model') }}\n"
microbatch_model_ref_render_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\nselect * from {{ ref('input_model').render() }}\n"
seed_csv: str = "id,event_time\n1,'2020-01-01 00:00:00-0'\n2,'2020-01-02 00:00:00-0'\n3,'2020-01-03 00:00:00-0'\n"
seeds_yaml: str = '\nseeds:\n  - name: raw_source\n    config:\n      column_types:\n        event_time: TIMESTAMP\n'
sources_yaml: str = '\nsources:\n  - name: seed_sources\n    schema: "{{ target.schema }}"\n    tables:\n      - name: raw_source\n        config:\n          event_time: event_time\n'
microbatch_model_calling_source_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\nselect * from {{ source('seed_sources', 'raw_source') }}\n"
custom_microbatch_strategy: str = '\n{% macro get_incremental_microbatch_sql(arg_dict) %}\n    {% do log(\'custom microbatch strategy\', info=True) %}\n\n     {%- set dest_cols_csv = get_quoted_csv(arg_dict["dest_columns"] | map(attribute="name")) -%}\n\n    insert into {{ arg_dict["target_relation"] }} ({{ dest_cols_csv }})\n    (\n        select {{ dest_cols_csv }}\n        from {{ arg_dict["temp_relation"] }}\n    )\n\n{% endmacro %}\n'
downstream_model_of_microbatch_sql: str = "\nSELECT * FROM {{ ref('microbatch_model') }}\n"
microbatch_model_full_refresh_false_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0), full_refresh=False) }}\nselect * from {{ ref('input_model') }}\n"


class BaseMicrobatchCustomUserStrategy:
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_sql}

    @pytest.fixture(scope='class')
    def macros(self) -> Dict[str, str]:
        return {'microbatch.sql': custom_microbatch_strategy}

    @pytest.fixture(scope='class')
    def project_config_update(self) -> Dict[str, Dict[str, bool]]:
        return {'flags': {'require_batched_execution_for_custom_microbatch_strategy': True}}

    @pytest.fixture(scope='class')
    def deprecation_catcher(self) -> EventCatcher:
        return EventCatcher(MicrobatchMacroOutsideOfBatchesDeprecation)


class TestMicrobatchCustomUserStrategyDefault(BaseMicrobatchCustomUserStrategy):
    @pytest.fixture(scope='class')
    def project_config_update(self) -> Dict[str, Dict[str, bool]]:
        return {'flags': {'require_batched_execution_for_custom_microbatch_strategy': False}}

    def test_use_custom_microbatch_strategy_by_default(self, project: TestProjInfo, deprecation_catcher: EventCatcher) -> None:
        run_dbt(['run'], callbacks=[deprecation_catcher.catch])
        assert len(deprecation_catcher.caught_events) == 1
        _, logs = run_dbt_and_capture(['run'])
        assert 'custom microbatch strategy' in logs
        assert 'START batch' not in logs


class TestMicrobatchCustomUserStrategyProjectFlagTrueValid(BaseMicrobatchCustomUserStrategy):
    def test_use_custom_microbatch_strategy_project_flag_true_invalid_incremental_strategy(self, project: TestProjInfo, deprecation_catcher: EventCatcher) -> None:
        with mock.patch.object(type(project.adapter), 'valid_incremental_strategies', lambda _: ['microbatch']):
            with patch_microbatch_end_time('2020-01-03 13:57:00'):
                run_dbt(['run'], callbacks=[deprecation_catcher.catch])
            assert len(deprecation_catcher.caught_events) == 0
            with patch_microbatch_end_time('2020-01-03 13:57:00'):
                _, logs = run_dbt_and_capture(['run'])
            assert 'custom microbatch strategy' in logs
            assert 'START batch' in logs


class TestMicrobatchCustomUserStrategyProjectFlagTrueNoValidBuiltin(BaseMicrobatchCustomUserStrategy):
    def test_use_custom_microbatch_strategy_project_flag_true_invalid_incremental_strategy(self, project: TestProjInfo) -> None:
        with mock.patch.object(type(project.adapter), 'valid_incremental_strategies', lambda _: []):
            with patch_microbatch_end_time('2020-01-03 13:57:00'):
                _, logs = run_dbt_and_capture(['run'])
            assert "'microbatch' is not valid" not in logs
            assert 'The use of a custom microbatch macro outside of batched execution is deprecated' not in logs


class BaseMicrobatchTest:
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_sql}

    def assert_row_count(self, project: TestProjInfo, relation_name: str, expected_row_count: int) -> None:
        relation = relation_from_name(project.adapter, relation_name)
        result = project.run_sql(f'select count(*) as num_rows from {relation}', fetch='one')
        if result[0] != expected_row_count:
            run_dbt(['show', '--inline', f'select * from {relation}'])
            assert result[0] == expected_row_count


class TestMicrobatchCLI(BaseMicrobatchTest):
    CLI_COMMAND_NAME: str = 'run'

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        model_catcher: EventCatcher = EventCatcher(event_to_catch=LogModelResult)
        batch_catcher: EventCatcher = EventCatcher(event_to_catch=LogBatchResult)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt([self.CLI_COMMAND_NAME], callbacks=[model_catcher.catch, batch_catcher.catch])
        self.assert_row_count(project, 'microbatch_model', 3)
        assert len(model_catcher.caught_events) == 2
        assert len(batch_catcher.caught_events) == 3
        batch_creation_events = 0
        for caught_event in batch_catcher.caught_events:
            if 'batch 2020' in caught_event.data.description:
                batch_creation_events += 1
                assert caught_event.data.execution_time > 0
        assert batch_creation_events == 3
        run_dbt([self.CLI_COMMAND_NAME, '--event-time-start', '2020-01-02', '--event-time-end', '2020-01-03', '--full-refresh'])
        self.assert_row_count(project, 'microbatch_model', 1)


class TestMicrobatchCLIBuild(TestMicrobatchCLI):
    CLI_COMMAND_NAME: str = 'build'


class TestMicrobatchCLIRunOutputJSON(BaseMicrobatchTest):
    def test_list_output_json(self, project: TestProjInfo) -> None:
        """Test whether the command `dbt list --output json` works"""
        model_catcher: EventCatcher = EventCatcher(event_to_catch=LogModelResult)
        batch_catcher: EventCatcher = EventCatcher(event_to_catch=LogBatchResult)
        _, microbatch_json = run_dbt_and_capture(['list', '--output', 'json'], callbacks=[model_catcher.catch, batch_catcher.catch])
        microbatch_dict: Dict[str, Any] = json.loads(microbatch_json)
        assert microbatch_dict['config']['begin'] == '2020-01-01T00:00:00'


class TestMicroBatchBoundsDefault(BaseMicrobatchTest):
    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 3)
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 3)
        test_schema_relation = project.adapter.Relation.create(database=project.database, schema=project.test_schema)
        project.run_sql(f"insert into {test_schema_relation}.input_model(id, event_time) values (4, TIMESTAMP '2020-01-04 00:00:00-0'), (5, TIMESTAMP '2020-01-05 00:00:00-0')")
        self.assert_row_count(project, 'input_model', 5)
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 3)
        with patch_microbatch_end_time('2020-01-04 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 4)
        with patch_microbatch_end_time('2020-01-05 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 5)


class TestMicrobatchWithSource(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def seeds(self) -> Dict[str, str]:
        return {'raw_source.csv': seed_csv}

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'microbatch_model.sql': microbatch_model_calling_source_sql, 'sources.yml': sources_yaml, 'seeds.yml': seeds_yaml}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        run_dbt(['seed'])
        catcher: EventCatcher = EventCatcher(event_to_catch=MicrobatchModelNoEventTimeInputs)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'], callbacks=[catcher.catch])
        self.assert_row_count(project, 'microbatch_model', 3)
        assert len(catcher.caught_events) == 0
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 3)
        test_schema_relation = project.adapter.Relation.create(database=project.database, schema=project.test_schema)
        project.run_sql(f"insert into {test_schema_relation}.raw_source(id, event_time) values (4, TIMESTAMP '2020-01-04 00:00:00-0'), (5, TIMESTAMP '2020-01-05 00:00:00-0')")
        self.assert_row_count(project, 'raw_source', 5)
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 3)
        with patch_microbatch_end_time('2020-01-04 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 4)
        with patch_microbatch_end_time('2020-01-05 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 5)


class TestMicrobatchJinjaContext(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def macros(self) -> Dict[str, str]:
        return {'check_batch_jinja_context.sql': invalid_batch_jinja_context_macro_sql}

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_with_context_checks_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 3)


class TestMicrobatchWithInputWithoutEventTime(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_without_event_time_sql, 'microbatch_model.sql': microbatch_model_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        catcher: EventCatcher = EventCatcher(event_to_catch=MicrobatchModelNoEventTimeInputs)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'], callbacks=[catcher.catch])
        self.assert_row_count(project, 'microbatch_model', 3)
        assert len(catcher.caught_events) == 1
        catcher.caught_events = []
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run'], callbacks=[catcher.catch])
        self.assert_row_count(project, 'microbatch_model', 3)
        assert len(catcher.caught_events) == 1
        test_schema_relation = project.adapter.Relation.create(database=project.database, schema=project.test_schema)
        project.run_sql(f"insert into {test_schema_relation}.input_model(id, event_time) values (4, TIMESTAMP '2020-01-04 00:00:00-0'), (5, TIMESTAMP '2020-01-05 00:00:00-0')")
        self.assert_row_count(project, 'input_model', 5)
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 5)


class TestMicrobatchUsingRefRenderSkipsFilter(BaseMicrobatchTest):
    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 3)
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 3)
        test_schema_relation = project.adapter.Relation.create(database=project.database, schema=project.test_schema)
        project.run_sql(f"insert into {test_schema_relation}.input_model(id, event_time) values (4, TIMESTAMP '2020-01-04 00:00:00-0'), (5, TIMESTAMP '2020-01-05 00:00:00-0')")
        self.assert_row_count(project, 'input_model', 5)
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 3)
        write_file(microbatch_model_ref_render_sql, project.project_root, 'models', 'microbatch_model.sql')
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run', '--select', 'microbatch_model'])
        self.assert_row_count(project, 'microbatch_model', 5)


microbatch_model_context_vars: str = '\n{{ config(materialized=\'incremental\', incremental_strategy=\'microbatch\', unique_key=\'id\', event_time=\'event_time\', batch_size=\'day\', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\n{{ log("start: "~ model.config.__dbt_internal_microbatch_event_time_start, info=True)}}\n{{ log("end: "~ model.config.__dbt_internal_microbatch_event_time_end, info=True)}}\n{% if model.batch %}\n{{ log("batch.event_time_start: "~ model.batch.event_time_start, info=True)}}\n{{ log("batch.event_time_end: "~ model.batch.event_time_end, info=True)}}\n{{ log("batch.id: "~ model.batch.id, info=True)}}\n{{ log("start timezone: "~ model.batch.event_time_start.tzinfo, info=True)}}\n{{ log("end timezone: "~ model.batch.event_time_end.tzinfo, info=True)}}\n{% endif %}\nselect * from {{ ref(\'input_model\') }}\n'


class TestMicrobatchJinjaContextVarsAvailable(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_context_vars}

    def test_run_with_event_time_logs(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _, logs = run_dbt_and_capture(['run'])
        assert 'start: 2020-01-01 00:00:00+00:00' in logs
        assert 'end: 2020-01-02 00:00:00+00:00' in logs
        assert 'batch.event_time_start: 2020-01-01 00:00:00+00:00' in logs
        assert 'batch.event_time_end: 2020-01-02 00:00:00+00:00' in logs
        assert 'batch.id: 20200101' in logs
        assert 'start timezone: UTC' in logs
        assert 'end timezone: UTC' in logs
        assert 'start: 2020-01-02 00:00:00+00:00' in logs
        assert 'end: 2020-01-03 00:00:00+00:00' in logs
        assert 'batch.event_time_start: 2020-01-02 00:00:00+00:00' in logs
        assert 'batch.event_time_end: 2020-01-03 00:00:00+00:00' in logs
        assert 'batch.id: 20200102' in logs
        assert 'start: 2020-01-03 00:00:00+00:00' in logs
        assert 'end: 2020-01-03 13:57:00+00:00' in logs
        assert 'batch.event_time_start: 2020-01-03 00:00:00+00:00' in logs
        assert 'batch.event_time_end: 2020-01-03 13:57:00+00:00' in logs
        assert 'batch.id: 20200103' in logs


microbatch_model_failing_incremental_partition_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\n{% if '2020-01-02' in (model.config.__dbt_internal_microbatch_event_time_start | string) %}\n invalid_sql\n{% endif %}\nselect * from {{ ref('input_model') }}\n"


class TestMicrobatchIncrementalBatchFailure(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_failing_incremental_partition_sql, 'downstream_model.sql': downstream_model_of_microbatch_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        event_catcher: EventCatcher = EventCatcher(GenericExceptionOnRun, predicate=lambda event: event.data.node_info is not None)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'], callbacks=[event_catcher.catch], expect_pass=False)
        assert len(event_catcher.caught_events) == 1
        self.assert_row_count(project, 'microbatch_model', 2)
        run_results: Dict[str, Any] = get_artifact(project.project_root, 'target', 'run_results.json')
        microbatch_run_result: Dict[str, Any] = run_results['results'][1]
        assert microbatch_run_result['status'] == 'partial success'
        batch_results: Optional[Dict[str, Any]] = microbatch_run_result['batch_results']
        assert batch_results is not None
        assert len(batch_results['successful']) == 2
        assert len(batch_results['failed']) == 1
        assert run_results['results'][2]['status'] == 'skipped'


class TestMicrobatchRetriesPartialSuccesses(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_failing_incremental_partition_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _, console_output = run_dbt_and_capture(['run'], expect_pass=False)
        assert 'PARTIAL SUCCESS (2/3)' in console_output
        assert 'Completed with 1 partial success' in console_output
        self.assert_row_count(project, 'microbatch_model', 2)
        run_results: Dict[str, Any] = get_artifact(project.project_root, 'target', 'run_results.json')
        microbatch_run_result: Dict[str, Any] = run_results['results'][1]
        assert microbatch_run_result['status'] == 'partial success'
        batch_results: Optional[Dict[str, Any]] = microbatch_run_result['batch_results']
        assert batch_results is not None
        assert len(batch_results['successful']) == 2
        assert len(batch_results['failed']) == 1
        write_file(microbatch_model_sql, project.project_root, 'models', 'microbatch_model.sql')
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _, console_output = run_dbt_and_capture(['retry'])
        assert 'PARTIAL SUCCESS' not in console_output
        assert 'Completed with 1 partial success' not in console_output
        assert 'Completed successfully' in console_output
        self.assert_row_count(project, 'microbatch_model', 3)


class TestMicrobatchMultipleRetries(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_failing_incremental_partition_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _, console_output = run_dbt_and_capture(['run'], expect_pass=False)
        assert 'PARTIAL SUCCESS (2/3)' in console_output
        assert 'Completed with 1 partial success' in console_output
        self.assert_row_count(project, 'microbatch_model', 2)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _, console_output = run_dbt_and_capture(['retry'], expect_pass=False)
        assert 'PARTIAL SUCCESS' not in console_output
        assert 'ERROR' in console_output
        assert 'Completed with 1 error, 0 partial successes, and 0 warnings' in console_output
        self.assert_row_count(project, 'microbatch_model', 2)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _, console_output = run_dbt_and_capture(['retry'], expect_pass=False)
        assert 'PARTIAL SUCCESS' not in console_output
        assert 'ERROR' in console_output
        assert 'Completed with 1 error, 0 partial successes, and 0 warnings' in console_output
        self.assert_row_count(project, 'microbatch_model', 2)


microbatch_model_first_partition_failing_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\n{% if '2020-01-01' in (model.config.__dbt_internal_microbatch_event_time_start | string) %}\n invalid_sql\n{% endif %}\nselect * from {{ ref('input_model') }}\n"
microbatch_model_second_batch_failing_sql: str = "\n{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}\n{% if '20200102' == model.batch.id %}\n invalid_sql\n{% endif %}\nselect * from {{ ref('input_model') }}\n"


class TestMicrobatchInitialBatchFailure(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_first_partition_failing_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        general_exc_catcher: EventCatcher = EventCatcher(GenericExceptionOnRun, predicate=lambda event: event.data.node_info is not None)
        batch_catcher: EventCatcher = EventCatcher(event_to_catch=LogBatchResult, predicate=lambda event: event.data.status == 'skipped')
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'], expect_pass=False, callbacks=[general_exc_catcher.catch, batch_catcher.catch])
        assert len(general_exc_catcher.caught_events) == 1
        assert len(batch_catcher.caught_events) == 2
        relation_info = relation_from_name(project.adapter, 'microbatch_model')
        relation = project.adapter.get_relation(relation_info.database, relation_info.schema, relation_info.name)
        assert relation is None


class TestMicrobatchSecondBatchFailure(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_second_batch_failing_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        event_catcher: EventCatcher = EventCatcher(GenericExceptionOnRun, predicate=lambda event: event.data.node_info is not None)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'], expect_pass=False, callbacks=[event_catcher.catch])
        assert len(event_catcher.caught_events) == 1
        self.assert_row_count(project, 'microbatch_model', 2)


class TestMicrobatchCompiledRunPaths(BaseMicrobatchTest):
    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'])
        assert read_file(project.project_root, 'target', 'compiled', 'test', 'models', 'microbatch_model', 'microbatch_model_2020-01-01.sql')
        assert read_file(project.project_root, 'target', 'compiled', 'test', 'models', 'microbatch_model', 'microbatch_model_2020-01-02.sql')
        assert read_file(project.project_root, 'target', 'compiled', 'test', 'models', 'microbatch_model', 'microbatch_model_2020-01-03.sql')
        assert read_file(project.project_root, 'target', 'run', 'test', 'models', 'microbatch_model', 'microbatch_model_2020-01-01.sql')
        assert read_file(project.project_root, 'target', 'run', 'test', 'models', 'microbatch_model', 'microbatch_model_2020-01-02.sql')
        assert read_file(project.project_root, 'target', 'run', 'test', 'models', 'microbatch_model', 'microbatch_model_2020-01-03.sql')


class TestMicrobatchFullRefreshConfigFalse(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_full_refresh_false_sql, 'downstream_model.sql': downstream_model_of_microbatch_sql}

    def test_run_with_event_time(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run', '--event-time-start', '2020-01-02', '--event-time-end', '2020-01-03 13:57:00'])
        self.assert_row_count(project, 'microbatch_model', 2)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 2)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run', '--full-refresh'])
        self.assert_row_count(project, 'microbatch_model', 2)
        write_file(microbatch_model_sql, project.project_root, 'models', 'microbatch_model.sql')
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run', '--full-refresh'])
        self.assert_row_count(project, 'microbatch_model', 3)


class TestMicrbobatchModelsRunWithSameCurrentTime(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_yearly_model_sql, 'second_microbatch_model.sql': microbatch_yearly_model_downstream_sql}

    def test_microbatch(self, project: TestProjInfo) -> None:
        run_dbt(['run'])
        run_results: Dict[str, Any] = get_artifact(project.project_root, 'target', 'run_results.json')
        microbatch_model_last_batch: Any = run_results['results'][1]['batch_results']['successful'][-1]
        second_microbatch_model_last_batch: Any = run_results['results'][2]['batch_results']['successful'][-1]
        assert microbatch_model_last_batch == second_microbatch_model_last_batch


class TestMicrobatchModelStoppedByKeyboardInterrupt(BaseMicrobatchTest):
    @pytest.fixture
    def catch_eors(self) -> EventCatcher:
        return EventCatcher(EndOfRunSummary)

    @pytest.fixture
    def catch_aw(self) -> EventCatcher:
        return EventCatcher(event_to_catch=ArtifactWritten, predicate=lambda event: event.data.artifact_type == 'RunExecutionResult')

    def test_microbatch(self, mocker: MockerFixture, project: TestProjInfo, catch_eors: EventCatcher, catch_aw: EventCatcher) -> None:
        mocked_fbs = mocker.patch('dbt.materializations.incremental.microbatch.MicrobatchBuilder.format_batch_start')
        mocked_fbs.side_effect = KeyboardInterrupt
        try:
            run_dbt(['run'], callbacks=[catch_eors.catch, catch_aw.catch])
            assert False, 'KeyboardInterrupt failed to stop batch execution'
        except KeyboardInterrupt:
            assert len(catch_eors.caught_events) == 1
            assert 'Exited because of keyboard interrupt' in catch_eors.caught_events[0].info.msg
            assert len(catch_aw.caught_events) == 1


class TestMicrobatchModelSkipped(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_invalid_sql, 'microbatch_model.sql': microbatch_model_sql}

    def test_microbatch_model_skipped(self, project: TestProjInfo) -> None:
        run_dbt(['run'], expect_pass=False)
        run_results: Dict[str, Any] = get_artifact(project.project_root, 'target', 'run_results.json')
        microbatch_result: Dict[str, Any] = run_results['results'][1]
        assert microbatch_result['status'] == 'skipped'
        assert microbatch_result['batch_results'] is None


class TestMicrobatchCanRunParallelOrSequential(BaseMicrobatchTest):
    @pytest.fixture
    def batch_exc_catcher(self) -> EventCatcher:
        return EventCatcher(MicrobatchExecutionDebug)

    def test_microbatch(self, mocker: MockerFixture, project: TestProjInfo, batch_exc_catcher: EventCatcher) -> None:
        mocked_srip = mocker.patch('dbt.task.run.MicrobatchModelRunner.should_run_in_parallel')
        mocked_srip.return_value = True
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _ = run_dbt(['run'], callbacks=[batch_exc_catcher.catch])
        assert len(batch_exc_catcher.caught_events) > 1
        some_batches_run_concurrently = False
        for caugh_event in batch_exc_catcher.caught_events:
            if 'is being run concurrently' in caugh_event.data.msg:
                some_batches_run_concurrently = True
                break
        assert some_batches_run_concurrently, 'Found no batches being run concurrently!'
        batch_exc_catcher.caught_events = []
        mocked_srip.return_value = False
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _ = run_dbt(['run'], callbacks=[batch_exc_catcher.catch])
        assert len(batch_exc_catcher.caught_events) > 1
        some_batches_run_concurrently = False
        for caugh_event in batch_exc_catcher.caught_events:
            if 'is being run concurrently' in caugh_event.data.msg:
                some_batches_run_concurrently = True
                break
        assert not some_batches_run_concurrently, 'Found a batch being run concurrently!'


class TestFirstAndLastBatchAlwaysSequential(BaseMicrobatchTest):
    @pytest.fixture
    def batch_exc_catcher(self) -> EventCatcher:
        return EventCatcher(MicrobatchExecutionDebug)

    def test_microbatch(self, mocker: MockerFixture, project: TestProjInfo, batch_exc_catcher: EventCatcher) -> None:
        mocked_srip = mocker.patch('dbt.task.run.MicrobatchModelRunner.should_run_in_parallel')
        mocked_srip.return_value = True
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            _ = run_dbt(['run'], callbacks=[batch_exc_catcher.catch])
        assert len(batch_exc_catcher.caught_events) > 1
        first_batch_event = batch_exc_catcher.caught_events[0]
        last_batch_event = batch_exc_catcher.caught_events[-1]
        for event in [first_batch_event, last_batch_event]:
            assert 'is being run sequentially' in event.data.msg
        for event in batch_exc_catcher.caught_events[1:-1]:
            assert 'is being run concurrently' in event.data.msg


class TestFirstBatchRunsPreHookLastBatchRunsPostHook(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_with_pre_and_post_sql}

    @pytest.fixture
    def batch_log_catcher(self) -> EventCatcher:
        def pre_or_post_hook(event: Any) -> bool:
            return 'execute: True' in event.data.msg and ('pre-hook' in event.data.msg or 'post-hook' in event.data.msg)

        return EventCatcher(event_to_catch=JinjaLogDebug, predicate=pre_or_post_hook)

    def test_microbatch(self, mocker: MockerFixture, project: TestProjInfo, batch_log_catcher: EventCatcher) -> None:
        with patch_microbatch_end_time('2020-01-04 13:57:00'):
            _ = run_dbt(['run'], callbacks=[batch_log_catcher.catch])
        assert len(batch_log_catcher.caught_events) == 2
        for event in batch_log_catcher.caught_events:
            if '20200101' in event.data.msg:
                assert 'pre-hook' in event.data.msg
            if '20200104' in event.data.msg:
                assert 'post-hook' in event.data.msg


class TestWhenOnlyOneBatchRunBothPostAndPreHooks(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_with_pre_and_post_sql}

    @pytest.fixture
    def batch_log_catcher(self) -> EventCatcher:
        def pre_or_post_hook(event: Any) -> bool:
            return 'execute: True' in event.data.msg and ('pre-hook' in event.data.msg or 'post-hook' in event.data.msg)

        return EventCatcher(event_to_catch=JinjaLogDebug, predicate=pre_or_post_hook)

    @pytest.fixture
    def generic_exception_catcher(self) -> EventCatcher:
        return EventCatcher(event_to_catch=GenericExceptionOnRun)

    def test_microbatch(self, project: TestProjInfo, batch_log_catcher: EventCatcher, generic_exception_catcher: EventCatcher) -> None:
        with patch_microbatch_end_time('2020-01-01 13:57:00'):
            _ = run_dbt(['run'], callbacks=[batch_log_catcher.catch, generic_exception_catcher.catch])
        assert len(batch_log_catcher.caught_events) == 2
        assert '20200101' in batch_log_catcher.caught_events[0].data.msg
        assert 'pre-hook' in batch_log_catcher.caught_events[0].data.msg
        assert '20200101' in batch_log_catcher.caught_events[1].data.msg
        assert 'post-hook' in batch_log_catcher.caught_events[1].data.msg
        assert len(generic_exception_catcher.caught_events) == 0


class TestCanSilenceInvalidConcurrentBatchesConfigWarning(BaseMicrobatchTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_force_concurrent_batches_sql}

    @pytest.fixture
    def event_catcher(self) -> EventCatcher:
        return EventCatcher(event_to_catch=InvalidConcurrentBatchesConfig)

    def test_microbatch(self, project: TestProjInfo, event_catcher: EventCatcher) -> None:
        with patch_microbatch_end_time('2020-01-01 13:57:00'):
            _ = run_dbt(['run'], callbacks=[event_catcher.catch])
        assert len(event_catcher.caught_events) == 1
        event_catcher.caught_events = []
        with patch_microbatch_end_time('2020-01-01 13:57:00'):
            _ = run_dbt(['run', '--warn-error-options', "{'silence': ['InvalidConcurrentBatchesConfig']}"], callbacks=[event_catcher.catch])
        assert len(event_catcher.caught_events) == 0