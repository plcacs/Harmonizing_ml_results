from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pytz
from dbt.artifacts.resources.types import BatchSize
from dbt.artifacts.schemas.batch_results import BatchType
from dbt.contracts.graph.nodes import ModelNode, NodeConfig
from dbt.exceptions import DbtInternalError, DbtRuntimeError


class MicrobatchBuilder:
    """A utility class for building microbatch definitions associated with a specific model"""

    def __init__(
        self,
        model: ModelNode,
        is_incremental: bool,
        event_time_start: Optional[datetime],
        event_time_end: Optional[datetime],
        default_end_time: Optional[datetime] = None,
    ) -> None:
        if model.config.incremental_strategy != 'microbatch':
            raise DbtInternalError(f"Model '{model.name}' does not use 'microbatch' incremental_strategy.")
        self.model: ModelNode = model
        if self.model.config.batch_size is None:
            raise DbtRuntimeError(
                f"Microbatch model '{self.model.name}' does not have a 'batch_size' config (one of "
                f"{[batch_size.value for batch_size in BatchSize]}) specificed."
            )
        self.is_incremental: bool = is_incremental
        self.event_time_start: Optional[datetime] = (
            event_time_start.replace(tzinfo=pytz.UTC) if event_time_start else None
        )
        self.event_time_end: Optional[datetime] = (
            event_time_end.replace(tzinfo=pytz.UTC) if event_time_end else None
        )
        self.default_end_time: datetime = default_end_time or datetime.now(pytz.UTC)

    def build_end_time(self) -> datetime:
        """Defaults the end_time to the current time in UTC unless a non `None` event_time_end was provided"""
        end_time: datetime = self.event_time_end or self.default_end_time
        return MicrobatchBuilder.ceiling_timestamp(end_time, self.model.config.batch_size)

    def build_start_time(self, checkpoint: Optional[datetime]) -> datetime:
        """
        Create a start time based off the passed in checkpoint.
    
        If the checkpoint is `None`, or this is the first run of a microbatch model, then the
        model's configured `begin` value will be returned as a checkpoint is necessary
        to build a start time. This is because we build the start time relative to the checkpoint
        via the batchsize and offset, and we cannot offset a checkpoint if there is no checkpoint.
        """
        assert isinstance(self.model.config, NodeConfig)
        batch_size: BatchSize = self.model.config.batch_size
        if self.event_time_start:
            return MicrobatchBuilder.truncate_timestamp(self.event_time_start, batch_size)
        if not self.is_incremental or checkpoint is None:
            if not self.model.config.begin:
                raise DbtRuntimeError(f"Microbatch model '{self.model.name}' requires a 'begin' configuration.")
            return MicrobatchBuilder.truncate_timestamp(self.model.config.begin, batch_size)
        lookback: int = self.model.config.lookback
        if checkpoint == MicrobatchBuilder.truncate_timestamp(checkpoint, batch_size):
            lookback += 1
        return MicrobatchBuilder.offset_timestamp(checkpoint, batch_size, -1 * lookback)

    def build_batches(
        self, start: datetime, end: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        Given a start and end datetime, builds a list of batches where each batch is
        the size of the model's batch_size.
        """
        batch_size: BatchSize = self.model.config.batch_size
        curr_batch_start: datetime = start
        curr_batch_end: datetime = MicrobatchBuilder.offset_timestamp(curr_batch_start, batch_size, 1)
        batches: List[Tuple[datetime, datetime]] = [(curr_batch_start, curr_batch_end)]
        while curr_batch_end < end:
            curr_batch_start = curr_batch_end
            curr_batch_end = MicrobatchBuilder.offset_timestamp(curr_batch_start, batch_size, 1)
            batches.append((curr_batch_start, curr_batch_end))
        batches[-1] = (batches[-1][0], end)
        return batches

    def build_jinja_context_for_batch(self, incremental_batch: bool) -> Dict[str, Any]:
        """
        Create context with entries that reflect microbatch model + incremental execution state
    
        Assumes self.model has been (re)-compiled with necessary batch filters applied.
        """
        jinja_context: Dict[str, Any] = {}
        jinja_context['model'] = self.model.to_dict()
        jinja_context['sql'] = self.model.compiled_code
        jinja_context['compiled_code'] = self.model.compiled_code
        if incremental_batch:
            jinja_context['is_incremental'] = lambda: True
            jinja_context['should_full_refresh'] = lambda: False
        return jinja_context

    @staticmethod
    def offset_timestamp(timestamp: datetime, batch_size: BatchSize, offset: int) -> datetime:
        """Truncates the passed in timestamp based on the batch_size and then applies the offset by the batch_size.

        Note: It's important to understand that the offset applies to the truncated timestamp, not
        the origin timestamp. Thus being offset by a day isn't relative to the any given hour that day,
        but relative to the start of the day. So if the timestamp is the very end of a day, 2024-09-17 23:59:59,
        you have a batch size of a day, and an offset of +1, then the returned value ends up being only one
        second later, 2024-09-18 00:00:00.

        2024-09-17 16:06:00 + Batchsize.hour -1 -> 2024-09-17 15:00:00
        2024-09-17 16:06:00 + Batchsize.hour +1 -> 2024-09-17 17:00:00
        2024-09-17 16:06:00 + Batchsize.day -1 -> 2024-09-16 00:00:00
        2024-09-17 16:06:00 + Batchsize.day +1 -> 2024-09-18 00:00:00
        2024-09-17 16:06:00 + Batchsize.month -1 -> 2024-08-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.month +1 -> 2024-10-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.year -1 -> 2023-01-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.year +1 -> 2025-01-01 00:00:00
        """
        truncated: datetime = MicrobatchBuilder.truncate_timestamp(timestamp, batch_size)
        if batch_size == BatchSize.hour:
            offset_timestamp: datetime = truncated + timedelta(hours=offset)
        elif batch_size == BatchSize.day:
            offset_timestamp = truncated + timedelta(days=offset)
        elif batch_size == BatchSize.month:
            offset_timestamp = truncated
            for _ in range(abs(offset)):
                if offset < 0:
                    offset_timestamp = offset_timestamp - timedelta(days=1)
                else:
                    offset_timestamp = offset_timestamp + timedelta(days=31)
                offset_timestamp = MicrobatchBuilder.truncate_timestamp(offset_timestamp, batch_size)
        elif batch_size == BatchSize.year:
            offset_timestamp = truncated.replace(year=truncated.year + offset)
        else:
            offset_timestamp = truncated  # Fallback, though ideally all cases are covered.
        return offset_timestamp

    @staticmethod
    def truncate_timestamp(timestamp: datetime, batch_size: BatchSize) -> datetime:
        """Truncates the passed in timestamp based on the batch_size.
    
        2024-09-17 16:06:00 + Batchsize.hour -> 2024-09-17 16:00:00
        2024-09-17 16:06:00 + Batchsize.day -> 2024-09-17 00:00:00
        2024-09-17 16:06:00 + Batchsize.month -> 2024-09-01 00:00:00
        2024-09-17 16:06:00 + Batchsize.year -> 2024-01-01 00:00:00
        """
        if batch_size == BatchSize.hour:
            truncated: datetime = datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, 0, 0, 0, pytz.utc)
        elif batch_size == BatchSize.day:
            truncated = datetime(timestamp.year, timestamp.month, timestamp.day, 0, 0, 0, 0, pytz.utc)
        elif batch_size == BatchSize.month:
            truncated = datetime(timestamp.year, timestamp.month, 1, 0, 0, 0, 0, pytz.utc)
        elif batch_size == BatchSize.year:
            truncated = datetime(timestamp.year, 1, 1, 0, 0, 0, 0, pytz.utc)
        else:
            truncated = timestamp  # Fallback, though ideally all cases are covered.
        return truncated

    @staticmethod
    def batch_id(start_time: datetime, batch_size: BatchSize) -> str:
        return MicrobatchBuilder.format_batch_start(start_time, batch_size).replace('-', '')

    @staticmethod
    def format_batch_start(batch_start: datetime, batch_size: BatchSize) -> str:
        return str(batch_start.date() if batch_start and batch_size != BatchSize.hour else batch_start)

    @staticmethod
    def ceiling_timestamp(timestamp: datetime, batch_size: BatchSize) -> datetime:
        """Takes the given timestamp and moves it to the ceiling for the given batch size
    
        Note, if the timestamp is already the batch size ceiling, that is returned
        2024-09-17 16:06:00 + BatchSize.hour -> 2024-09-17 17:00:00
        2024-09-17 16:00:00 + BatchSize.hour -> 2024-09-17 16:00:00
        2024-09-17 16:06:00 + BatchSize.day -> 2024-09-18 00:00:00
        2024-09-17 00:00:00 + BatchSize.day -> 2024-09-17 00:00:00
        2024-09-17 16:06:00 + BatchSize.month -> 2024-10-01 00:00:00
        2024-09-01 00:00:00 + BatchSize.month -> 2024-09-01 00:00:00
        2024-09-17 16:06:00 + BatchSize.year -> 2025-01-01 00:00:00
        2024-01-01 00:00:00 + BatchSize.year -> 2024-01-01 00:00:00
    
        """
        ceiling: datetime = MicrobatchBuilder.truncate_timestamp(timestamp, batch_size)
        if ceiling != timestamp:
            ceiling = MicrobatchBuilder.offset_timestamp(ceiling, batch_size, 1)
        return ceiling