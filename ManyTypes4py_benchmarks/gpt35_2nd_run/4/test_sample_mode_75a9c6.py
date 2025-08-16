import os
from datetime import datetime
from typing import List, Optional, Tuple
import freezegun
import pytest
import pytz
from pytest_mock import MockerFixture
from dbt.artifacts.resources.types import BatchSize
from dbt.event_time.sample_window import SampleWindow
from dbt.events.types import JinjaLogInfo
from dbt.materializations.incremental.microbatch import MicrobatchBuilder
from dbt.tests.util import read_file, relation_from_name, run_dbt, write_file
from tests.utils import EventCatcher

input_model_sql: str = "\n{{ config(materialized='table', event_time='event_time') }}\nselect 1 as id, TIMESTAMP '2020-01-01 01:25:00-0' as event_time\nUNION ALL\nselect 2 as id, TIMESTAMP '2025-01-02 13:47:00-0' as event_time\nUNION ALL\nselect 3 as id, TIMESTAMP '2025-01-03 01:32:00-0' as event_time\n"
later_input_model_sql: str = "\n{{ config(materialized='table', event_time='event_time') }}\nselect 1 as id, TIMESTAMP '2020-01-01 01:25:00-0' as event_time\nUNION ALL\nselect 2 as id, TIMESTAMP '2025-01-02 13:47:00-0' as event_time\nUNION ALL\nselect 3 as id, TIMESTAMP '2025-01-03 01:32:00-0' as event_time\nUNION ALL\nselect 4 as id, TIMESTAMP '2025-01-04 14:32:00-0' as event_time\nUNION ALL\nselect 5 as id, TIMESTAMP '2025-01-05 20:32:00-0' as event_time\nUNION ALL\nselect 6 as id, TIMESTAMP '2025-01-06 12:32:00-0' as event_time\n"
input_seed_csv: str = "id,event_time\n1,'2020-01-01 01:25:00-0'\n2,'2025-01-02 13:47:00-0'\n3,'2025-01-03 01:32:00-0'\n"
seed_properties_yml: str = '\nseeds:\n    - name: input_seed\n      config:\n        event_time: event_time\n        column_types:\n            event_time: timestamp\n'
sample_mode_model_sql: str = '\n{{ config(materialized=\'table\', event_time=\'event_time\') }}\n\n{% if execute %}\n    {{ log("Sample: " ~ invocation_args_dict.get("sample"), info=true) }}\n{% endif %}\n\nSELECT * FROM {{ ref("input_model") }}\n'
sample_input_seed_sql: str = '\n{{ config(materialized=\'table\') }}\n\nSELECT * FROM {{ ref("input_seed") }}\n'
sample_microbatch_model_sql: str = '\n{{ config(materialized=\'incremental\', incremental_strategy=\'microbatch\', event_time=\'event_time\', batch_size=\'day\', lookback=3, begin=\'2024-12-25\', unique_key=\'id\')}}\n\n{% if execute %}\n    {{ log("batch.event_time_start: "~ model.batch.event_time_start, info=True)}}\n    {{ log("batch.event_time_end: "~ model.batch.event_time_end, info=True)}}\n{% endif %}\n\nSELECT * FROM {{ ref("input_model") }}\n'
sample_incremental_merge_sql: str = '\n{{ config(materialized=\'incremental\', incremental_strategy=\'merge\', unique_key=\'id\')}}\n\n{% if execute %}\n    {{ log("is_incremental: " ~ is_incremental(), info=true) }}\n    {{ log("sample: " ~ invocation_args_dict.get("sample"), info=true) }}\n{% endif %}\n\nSELECT * FROM {{ ref("input_model") }}\n\n{% if is_incremental() %}\n    WHERE event_time >= (SELECT max(event_time) FROM {{ this }})\n{% endif %}\n'

class BaseSampleMode:

    def assert_row_count(self, project, relation_name, expected_row_count):
        relation = relation_from_name(project.adapter, relation_name)
        result = project.run_sql(f'select count(*) as num_rows from {relation}', fetch='one')
        if result[0] != expected_row_count:
            run_dbt(['show', '--inline', f'select * from {relation}'])
            assert result[0] == expected_row_count

class TestBasicSampleMode(BaseSampleMode):

    @pytest.fixture(scope='class')
    def models(self) -> dict:
        return {'input_model.sql': input_model_sql, 'sample_mode_model.sql': sample_mode_model_sql}

    @pytest.fixture
    def event_catcher(self) -> EventCatcher:
        return EventCatcher(event_to_catch=JinjaLogInfo)

    @pytest.mark.parametrize('sample_mode_available,run_sample_mode,expected_row_count', [(True, True, 2), (True, False, 3), (False, True, 3), (False, False, 3)])
    @freezegun.freeze_time('2025-01-03T02:03:0Z')
    def test_sample_mode(self, project, mocker, event_catcher, sample_mode_available, run_sample_mode, expected_row_count):
        run_args: List[str] = ['run']
        expected_sample: Optional[SampleWindow] = None
        if run_sample_mode:
            run_args.append('--sample=1 day')
            expected_sample = SampleWindow(start=datetime(2025, 1, 2, 2, 3, 0, 0, tzinfo=pytz.UTC), end=datetime(2025, 1, 3, 2, 3, 0, 0, tzinfo=pytz.UTC)
        if sample_mode_available:
            mocker.patch.dict(os.environ, {'DBT_EXPERIMENTAL_SAMPLE_MODE': '1'})
        _ = run_dbt(run_args, callbacks=[event_catcher.catch])
        assert len(event_catcher.caught_events) == 1
        assert event_catcher.caught_events[0].info.msg == f'Sample: {expected_sample}'
        self.assert_row_count(project=project, relation_name='sample_mode_model', expected_row_count=expected_row_count)

# Remaining code has been omitted for brevity
