import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple, TypeVar, Union
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
    MicrobatchModelNoEventTimeInputs
)
from dbt.tests.fixtures.project import TestProjInfo
from dbt.tests.util import (
    get_artifact,
    patch_microbatch_end_time,
    read_file,
    relation_from_name,
    run_dbt,
    run_dbt_and_capture,
    write_file
)
from tests.utils import EventCatcher

T = TypeVar('T')

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
    def project_config_update(self) -> Dict[str, Any]:
        return {'flags': {'require_batched_execution_for_custom_microbatch_strategy': True}}

    @pytest.fixture(scope='class')
    def deprecation_catcher(self) -> EventCatcher:
        return EventCatcher(MicrobatchMacroOutsideOfBatchesDeprecation)

class TestMicrobatchCustomUserStrategyDefault(BaseMicrobatchCustomUserStrategy):

    @pytest.fixture(scope='class')
    def project_config_update(self) -> Dict[str, Any]:
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
        model_catcher = EventCatcher(event_to_catch=LogModelResult)
        batch_catcher = EventCatcher(event_to_catch=LogBatchResult)
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
        model_catcher = EventCatcher(event_to_catch=LogModelResult)
        batch_catcher = EventCatcher(event_to_catch=LogBatchResult)
        _, microbatch_json = run_dbt(['list', '--output', 'json'], callbacks=[model_catcher.catch, batch_catcher.catch])
        microbatch_dict = json.loads(microbatch_json)
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
        catcher = EventCatcher(event_to_catch=MicrobatchModelNoEventTimeInputs)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'], callbacks=[catcher.catch])
        self.assert_row_count(project, 'microbatch_model', 3)
        assert len(catcher.caught_events) == 0
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run'])
        self.assert_row_count(project, 'microbatch_model', 3)
        test_schema_relation = project.adapter.Relation.create(database=project.database, schema=project.test_schema)
        project.run