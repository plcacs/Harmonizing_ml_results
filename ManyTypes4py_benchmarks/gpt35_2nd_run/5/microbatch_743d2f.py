    def __init__(self, model: ModelNode, is_incremental: bool, event_time_start: Optional[datetime], event_time_end: Optional[datetime], default_end_time: Optional[datetime] = None) -> None:
    def build_end_time(self) -> datetime:
    def build_start_time(self, checkpoint: Optional[datetime]) -> datetime:
    def build_batches(self, start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
    def build_jinja_context_for_batch(self, incremental_batch: bool) -> Dict[str, Any]:
    @staticmethod
    def offset_timestamp(timestamp: datetime, batch_size: BatchSize, offset: int) -> datetime:
    @staticmethod
    def truncate_timestamp(timestamp: datetime, batch_size: BatchSize) -> datetime:
    @staticmethod
    def batch_id(start_time: datetime, batch_size: BatchSize) -> str:
    @staticmethod
    def format_batch_start(batch_start: datetime, batch_size: BatchSize) -> str:
    @staticmethod
    def ceiling_timestamp(timestamp: datetime, batch_size: BatchSize) -> datetime:
