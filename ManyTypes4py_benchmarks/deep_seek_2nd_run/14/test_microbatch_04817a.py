from datetime import datetime
from unittest import mock
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import pytz
from freezegun import freeze_time
from dbt.artifacts.resources import NodeConfig
from dbt.artifacts.resources.types import BatchSize
from dbt.materializations.incremental.microbatch import MicrobatchBuilder

MODEL_CONFIG_BEGIN: datetime = datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC)

class TestMicrobatchBuilder:

    @pytest.fixture(scope='class')
    def microbatch_model(self) -> mock.Mock:
        model = mock.Mock()
        model.config = mock.MagicMock(NodeConfig)
        model.config.materialized = 'incremental'
        model.config.incremental_strategy = 'microbatch'
        model.config.begin = MODEL_CONFIG_BEGIN
        model.config.batch_size = BatchSize.day
        return model

    @freeze_time('2024-09-05 08:56:00')
    @pytest.mark.parametrize('is_incremental,event_time_end,expected_end_time', [
        (False, None, datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC)), 
        (True, None, datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC)), 
        (False, datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC), datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC)), 
        (True, datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC), datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC))
    ])
    def test_build_end_time(
        self, 
        microbatch_model: mock.Mock, 
        is_incremental: bool, 
        event_time_end: Optional[datetime], 
        expected_end_time: datetime
    ) -> None:
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model, 
            is_incremental=is_incremental, 
            event_time_start=None, 
            event_time_end=event_time_end
        )
        assert microbatch_builder.build_end_time() == expected_end_time

    @pytest.mark.parametrize(
        'is_incremental,event_time_start,checkpoint,batch_size,lookback,expected_start_time', 
        [
            (False, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.day, 0, MODEL_CONFIG_BEGIN), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.year, 0, datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.year, 1, datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
            (False, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.year, 0, MODEL_CONFIG_BEGIN), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.year, 0, datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.year, 1, datetime(2023, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.month, 0, datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC)), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.month, 1, datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC)), 
            (False, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.month, 0, MODEL_CONFIG_BEGIN), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.month, 0, datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.month, 1, datetime(2024, 8, 1, 0, 0, 0, 0, pytz.UTC)), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.day, 0, datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC)), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.day, 1, datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC)), 
            (False, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.day, 0, MODEL_CONFIG_BEGIN), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.day, 0, datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.day, 1, datetime(2024, 9, 4, 0, 0, 0, 0, pytz.UTC)), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.hour, 0, datetime(2024, 9, 5, 8, 0, 0, 0, pytz.UTC)), 
            (False, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.hour, 1, datetime(2024, 9, 5, 8, 0, 0, 0, pytz.UTC)), 
            (False, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.hour, 0, MODEL_CONFIG_BEGIN), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.hour, 0, datetime(2024, 9, 5, 8, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 8, 56, 0, 0, pytz.UTC), BatchSize.hour, 1, datetime(2024, 9, 5, 7, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC), BatchSize.hour, 0, datetime(2024, 9, 4, 23, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC), BatchSize.hour, 1, datetime(2024, 9, 4, 22, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC), BatchSize.day, 0, datetime(2024, 9, 4, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC), BatchSize.day, 1, datetime(2024, 9, 3, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC), BatchSize.month, 0, datetime(2024, 8, 1, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC), BatchSize.month, 1, datetime(2024, 7, 1, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC), BatchSize.year, 0, datetime(2023, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
            (True, None, datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC), BatchSize.year, 1, datetime(2022, 1, 1, 0, 0, 0, 0, pytz.UTC))
        ]
    )
    def test_build_start_time(
        self, 
        microbatch_model: mock.Mock, 
        is_incremental: bool, 
        event_time_start: Optional[datetime], 
        checkpoint: datetime, 
        batch_size: BatchSize, 
        lookback: int, 
        expected_start_time: datetime
    ) -> None:
        microbatch_model.config.batch_size = batch_size
        microbatch_model.config.lookback = lookback
        microbatch_builder = MicrobatchBuilder(
            model=microbatch_model, 
            is_incremental=is_incremental, 
            event_time_start=event_time_start, 
            event_time_end=None
        )
        assert microbatch_builder.build_start_time(checkpoint) == expected_start_time

    @pytest.mark.parametrize(
        'start,end,batch_size,expected_batches', 
        [
            (datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC), datetime(2026, 1, 7, 3, 56, 0, 0, pytz.UTC), BatchSize.year, [
                (datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC), datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
                (datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC), datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
                (datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC), datetime(2026, 1, 7, 3, 56, 0, 0, pytz.UTC))
            ]), 
            (datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC), datetime(2024, 11, 7, 3, 56, 0, 0, pytz.UTC), BatchSize.month, [
                (datetime(2024, 9, 1, 0, 0, 0, 0, pytz.UTC), datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC)), 
                (datetime(2024, 10, 1, 0, 0, 0, 0, pytz.UTC), datetime(2024, 11, 1, 0, 0, 0, 0, pytz.UTC)), 
                (datetime(2024, 11, 1, 0, 0, 0, 0, pytz.UTC), datetime(2024, 11, 7, 3, 56, 0, 0, pytz.UTC))
            ]), 
            (datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC), datetime(2024, 9, 7, 3, 56, 0, 0, pytz.UTC), BatchSize.day, [
                (datetime(2024, 9, 5, 0, 0, 0, 0, pytz.UTC), datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC)), 
                (datetime(2024, 9, 6, 0, 0, 0, 0, pytz.UTC), datetime(2024, 9, 7, 0, 0, 0, 0, pytz.UTC)), 
                (datetime(2024, 9, 7, 0, 0, 0, 0, pytz.UTC), datetime(2024, 9, 7, 3, 56, 0, 0, pytz.UTC))
            ]), 
            (datetime(2024, 9, 5, 1, 0, 0, 0, pytz.UTC), datetime(2024, 9, 5, 3, 56, 0, 0, pytz.UTC), BatchSize.hour, [
                (datetime(2024, 9, 5, 1, 0, 0, 0, pytz.UTC), datetime(2024, 9, 5, 2, 0, 0, 0, pytz.UTC)), 
                (datetime(2024, 9, 5, 2, 0, 0, 0, pytz.UTC), datetime(2024, 9, 5, 3, 0, 0, 0, pytz.UTC)), 
                (datetime(2024, 9, 5, 3, 0, 0, 0, pytz.UTC), datetime(2024, 9, 5, 3, 56, 0, 0, pytz.UTC))
            ]), 
            (datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC), datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC), BatchSize.year, [
                (datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC), datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC)), 
                (datetime(2025, 1, 1, 0, 0, 0, 0, pytz.UTC), datetime(2026, 1, 1, 0, 0, 0, 0, pytz.UTC))
            ]),