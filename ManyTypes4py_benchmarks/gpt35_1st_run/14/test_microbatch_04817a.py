from datetime import datetime
from typing import List, Tuple
import pytz

MODEL_CONFIG_BEGIN: datetime = datetime(2024, 1, 1, 0, 0, 0, 0, pytz.UTC)

class TestMicrobatchBuilder:

    def microbatch_model(self) -> mock.Mock:
    
    def test_build_end_time(self, microbatch_model: mock.Mock, is_incremental: bool, event_time_end: datetime, expected_end_time: datetime) -> None:
    
    def test_build_start_time(self, microbatch_model: mock.Mock, is_incremental: bool, event_time_start: datetime, checkpoint: datetime, batch_size: BatchSize, lookback: int, expected_start_time: datetime) -> None:
    
    def test_build_batches(self, microbatch_model: mock.Mock, start: datetime, end: datetime, batch_size: BatchSize, expected_batches: List[Tuple[datetime, datetime]]) -> None:
    
    def test_build_jinja_context_for_incremental_batch(self, microbatch_model: mock.Mock) -> None:
    
    def test_build_jinja_context_for_incremental_batch_false(self, microbatch_model: mock.Mock) -> None:
    
    def test_offset_timestamp(self, timestamp: datetime, batch_size: BatchSize, offset: int, expected_timestamp: datetime) -> None:
    
    def test_truncate_timestamp(self, timestamp: datetime, batch_size: BatchSize, expected_timestamp: datetime) -> None:
    
    def test_format_batch_start(self, batch_size: BatchSize, batch_start: datetime, expected_formatted_batch_start: str) -> None:
    
    def test_ceiling_timestamp(self, timestamp: datetime, batch_size: BatchSize, expected_datetime: datetime) -> None:
