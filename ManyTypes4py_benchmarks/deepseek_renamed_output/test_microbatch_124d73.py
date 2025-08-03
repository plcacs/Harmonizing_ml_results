import json
from unittest import mock
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
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

input_model_sql: str = """
{{ config(materialized='table', event_time='event_time') }}

select 1 as id, TIMESTAMP '2020-01-01 00:00:00-0' as event_time
union all
select 2 as id, TIMESTAMP '2020-01-02 00:00:00-0' as event_time
union all
select 3 as id, TIMESTAMP '2020-01-03 00:00:00-0' as event_time
"""

input_model_invalid_sql: str = """
{{ config(materialized='table', event_time='event_time') }}

select invalid as event_time
"""

input_model_without_event_time_sql: str = """
{{ config(materialized='table') }}

select 1 as id, TIMESTAMP '2020-01-01 00:00:00-0' as event_time
union all
select 2 as id, TIMESTAMP '2020-01-02 00:00:00-0' as event_time
union all
select 3 as id, TIMESTAMP '2020-01-03 00:00:00-0' as event_time
"""

microbatch_model_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
select * from {{ ref('input_model') }}
"""

microbatch_model_with_pre_and_post_sql: str = """
{{ config(
        materialized='incremental',
        incremental_strategy='microbatch',
        unique_key='id',
        event_time='event_time',
        batch_size='day',
        begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0),
        pre_hook='{{log("execute: " ~ execute ~ ", pre-hook run by batch " ~ model.batch.id)}}',
        post_hook='{{log("execute: " ~ execute ~ ", post-hook run by batch " ~ model.batch.id)}}',
    )
}}
select * from {{ ref('input_model') }}
"""

microbatch_model_force_concurrent_batches_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0), concurrent_batches=true) }}
select * from {{ ref('input_model') }}
"""

microbatch_yearly_model_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='year', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
select * from {{ ref('input_model') }}
"""

microbatch_yearly_model_downstream_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='year', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
select * from {{ ref('microbatch_model') }}
"""

invalid_batch_jinja_context_macro_sql: str = """
{% macro check_invalid_batch_jinja_context() %}

{% if model is not mapping %}
    {{ exceptions.raise_compiler_error("`model` is invalid: expected mapping type") }}
{% elif compiled_code and compiled_code is not string %}
    {{ exceptions.raise_compiler_error("`compiled_code` is invalid: expected string type") }}
{% elif sql and sql is not string %}
    {{ exceptions.raise_compiler_error("`sql` is invalid: expected string type") }}
{% elif is_incremental is not callable %}
    {{ exceptions.raise_compiler_error("`is_incremental()` is invalid: expected callable type") }}
{% elif should_full_refresh is not callable %}
    {{ exceptions.raise_compiler_error("`should_full_refresh()` is invalid: expected callable type") }}
{% endif %}

{% endmacro %}
"""

microbatch_model_with_context_checks_sql: str = """
{{ config(pre_hook="{{ check_invalid_batch_jinja_context() }}", materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}

{{ check_invalid_batch_jinja_context() }}
select * from {{ ref('input_model') }}
"""

microbatch_model_downstream_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
select * from {{ ref('microbatch_model') }}
"""

microbatch_model_ref_render_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
select * from {{ ref('input_model').render() }}
"""

seed_csv: str = """id,event_time
1,'2020-01-01 00:00:00-0'
2,'2020-01-02 00:00:00-0'
3,'2020-01-03 00:00:00-0'
"""

seeds_yaml: str = """
seeds:
  - name: raw_source
    config:
      column_types:
        event_time: TIMESTAMP
"""

sources_yaml: str = """
sources:
  - name: seed_sources
    schema: "{{ target.schema }}"
    tables:
      - name: raw_source
        config:
          event_time: event_time
"""

microbatch_model_calling_source_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
select * from {{ source('seed_sources', 'raw_source') }}
"""

custom_microbatch_strategy: str = """
{% macro get_incremental_microbatch_sql(arg_dict) %}
    {% do log('custom microbatch strategy', info=True) %}

     {%- set dest_cols_csv = get_quoted_csv(arg_dict["dest_columns"] | map(attribute="name")) -%}

    insert into {{ arg_dict["target_relation"] }} ({{ dest_cols_csv }})
    (
        select {{ dest_cols_csv }}
        from {{ arg_dict["temp_relation"] }}
    )

{% endmacro %}
"""

downstream_model_of_microbatch_sql: str = """
SELECT * FROM {{ ref('microbatch_model') }}
"""

microbatch_model_full_refresh_false_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0), full_refresh=False) }}
select * from {{ ref('input_model') }}
"""

microbatch_model_context_vars: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
{{ log("start: "~ model.config.__dbt_internal_microbatch_event_time_start, info=True)}}
{{ log("end: "~ model.config.__dbt_internal_microbatch_event_time_end, info=True)}}
{% if model.batch %}
{{ log("batch.event_time_start: "~ model.batch.event_time_start, info=True)}}
{{ log("batch.event_time_end: "~ model.batch.event_time_end, info=True)}}
{{ log("batch.id: "~ model.batch.id, info=True)}}
{{ log("start timezone: "~ model.batch.event_time_start.tzinfo, info=True)}}
{{ log("end timezone: "~ model.batch.event_time_end.tzinfo, info=True)}}
{% endif %}
select * from {{ ref('input_model') }}
"""

microbatch_model_failing_incremental_partition_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
{% if '2020-01-02' in (model.config.__dbt_internal_microbatch_event_time_start | string) %}
 invalid_sql
{% endif %}
select * from {{ ref('input_model') }}
"""

microbatch_model_first_partition_failing_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
{% if '2020-01-01' in (model.config.__dbt_internal_microbatch_event_time_start | string) %}
 invalid_sql
{% endif %}
select * from {{ ref('input_model') }}
"""

microbatch_model_second_batch_failing_sql: str = """
{{ config(materialized='incremental', incremental_strategy='microbatch', unique_key='id', event_time='event_time', batch_size='day', begin=modules.datetime.datetime(2020, 1, 1, 0, 0, 0)) }}
{% if '20200102' == model.batch.id %}
 invalid_sql
{% endif %}
select * from {{ ref('input_model') }}
"""


class BaseMicrobatchCustomUserStrategy:
    @pytest.fixture(scope='class')
    def func_4tdsamcl(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_sql}

    @pytest.fixture(scope='class')
    def func_fc82lafq(self) -> Dict[str, str]:
        return {'microbatch.sql': custom_microbatch_strategy}

    @pytest.fixture(scope='class')
    def func_j8bzhtol(self) -> Dict[str, Dict[str, bool]]:
        return {'flags': {'require_batched_execution_for_custom_microbatch_strategy': True}}

    @pytest.fixture(scope='class')
    def func_ecqiazn1(self) -> EventCatcher:
        return EventCatcher(MicrobatchMacroOutsideOfBatchesDeprecation)


class TestMicrobatchCustomUserStrategyDefault(BaseMicrobatchCustomUserStrategy):
    @pytest.fixture(scope='class')
    def func_j8bzhtol(self) -> Dict[str, Dict[str, bool]]:
        return {'flags': {'require_batched_execution_for_custom_microbatch_strategy': False}}

    def func_k4gtz03u(self, project: TestProjInfo, deprecation_catcher: EventCatcher) -> None:
        run_dbt(['run'], callbacks=[deprecation_catcher.catch])
        assert len(deprecation_catcher.caught_events) == 1
        _, logs = run_dbt_and_capture(['run'])
        assert 'custom microbatch strategy' in logs
        assert 'START batch' not in logs


class TestMicrobatchCustomUserStrategyProjectFlagTrueValid(BaseMicrobatchCustomUserStrategy):
    def func_h0wlz00w(self, project: TestProjInfo, deprecation_catcher: EventCatcher) -> None:
        with mock.patch.object(type(project.adapter), 'valid_incremental_strategies', lambda _: ['microbatch']):
            with patch_microbatch_end_time('2020-01-03 13:57:00'):
                run_dbt(['run'], callbacks=[deprecation_catcher.catch])
            assert len(deprecation_catcher.caught_events) == 0
            with patch_microbatch_end_time('2020-01-03 13:57:00'):
                _, logs = run_dbt_and_capture(['run'])
            assert 'custom microbatch strategy' in logs
            assert 'START batch' in logs


class TestMicrobatchCustomUserStrategyProjectFlagTrueNoValidBuiltin(BaseMicrobatchCustomUserStrategy):
    def func_h0wlz00w(self, project: TestProjInfo) -> None:
        with mock.patch.object(type(project.adapter), 'valid_incremental_strategies', lambda _: []):
            with patch_microbatch_end_time('2020-01-03 13:57:00'):
                _, logs = run_dbt_and_capture(['run'])
            assert "'microbatch' is not valid" not in logs
            assert 'The use of a custom microbatch macro outside of batched execution is deprecated' not in logs


class BaseMicrobatchTest:
    @pytest.fixture(scope='class')
    def func_4tdsamcl(self) -> Dict[str, str]:
        return {'input_model.sql': input_model_sql, 'microbatch_model.sql': microbatch_model_sql}

    def func_a86bxq7n(self, project: TestProjInfo, relation_name: str, expected_row_count: int) -> None:
        relation = relation_from_name(project.adapter, relation_name)
        result = project.run_sql(f'select count(*) as num_rows from {relation}', fetch='one')
        if result[0] != expected_row_count:
            run_dbt(['show', '--inline', f'select * from {relation}'])
            assert result[0] == expected_row_count


class TestMicrobatchCLI(BaseMicrobatchTest):
    CLI_COMMAND_NAME: str = 'run'

    def func_yx4w9g3x(self, project: TestProjInfo) -> None:
        model_catcher = EventCatcher(event_to_catch=LogModelResult)
        batch_catcher = EventCatcher(event_to_catch=LogBatchResult)
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt([self.CLI_COMMAND_NAME], callbacks=[model_catcher.catch, batch_catcher.catch])
        self.func_a86bxq7n(project, 'microbatch_model', 3)
        assert len(model_catcher.caught_events) == 2
        assert len(batch_catcher.caught_events) == 3
        batch_creation_events = 0
        for caught_event in batch_catcher.caught_events:
            if 'batch 2020' in caught_event.data.description:
                batch_creation_events += 1
                assert caught_event.data.execution_time > 0
        assert batch_creation_events == 3
        run_dbt([self.CLI_COMMAND_NAME, '--event-time-start', '2020-01-02', '--event-time-end', '2020-01-03', '--full-refresh'])
        self.func_a86bxq7n(project, 'microbatch_model', 1)


class TestMicrobatchCLIBuild(TestMicrobatchCLI):
    CLI_COMMAND_NAME: str = 'build'


class TestMicrobatchCLIRunOutputJSON(BaseMicrobatchTest):
    def func_51fsnlsc(self, project: TestProjInfo) -> None:
        """Test whether the command `dbt list --output json` works"""
        model_catcher = EventCatcher(event_to_catch=LogModelResult)
        batch_catcher = EventCatcher(event_to_catch=LogBatchResult)
        _, microbatch_json = run_dbt(['list', '--output', 'json'], callbacks=[model_catcher.catch, batch_catcher.catch])
        microbatch_dict = json.loads(microbatch_json)
        assert microbatch_dict['config']['begin'] == '2020-01-01T00:00:00'


class TestMicroBatchBoundsDefault(BaseMicrobatchTest):
    def func_yx4w9g3x(self, project: TestProjInfo) -> None:
        with patch_microbatch_end_time('2020-01-03 13:57:00'):
            run_dbt(['run'])
        self.func_a86bxq7n(project, 'microbatch_model', 3)
        with patch_microbatch_end_time('2020-01-03 14:57:00'):
            run_dbt(['run'])
        self.func_a86bxq7n(project, 'microbatch_model', 3)
        test_schema_relation = project.adapter.Relation.create(database=project.database, schema=project.test_schema)
        project.run_sql(
            f"insert into {test_schema_relation}.input_model(id, event_time) values (4, TIMESTAMP '2020-01-04 00:00:00-0'), (5, TIMESTAMP '2020-01-05 00:00:00-0')"
        )
        self.func_a86bxq7