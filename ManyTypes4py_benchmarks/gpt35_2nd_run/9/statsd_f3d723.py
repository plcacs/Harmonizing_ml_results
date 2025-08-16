    def on_message_in(self, tp: TP, offset: int, message: Message) -> None:
    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> Optional[Dict[str, Any]]:
    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Optional[Dict[str, Any]] = None) -> None:
    def on_message_out(self, tp: TP, offset: int, message: Message) -> None:
    def on_table_get(self, table: CollectionT, key: Any) -> None:
    def on_table_set(self, table: CollectionT, key: Any, value: Any) -> None:
    def on_table_del(self, table: CollectionT, key: Any) -> None:
    def on_commit_completed(self, consumer: ConsumerT, state: Any) -> None:
    def on_send_initiated(self, producer: ProducerT, topic: str, message: PendingMessage, keysize: int, valsize: int) -> None:
    def on_send_completed(self, producer: ProducerT, state: Any, metadata: RecordMetadata) -> None:
    def on_send_error(self, producer: ProducerT, exc: Exception, state: Any) -> None:
    def on_assignment_error(self, assignor: PartitionAssignorT, state: Dict[str, Any], exc: Exception) -> None:
    def on_assignment_completed(self, assignor: PartitionAssignorT, state: Dict[str, Any]) -> None:
    def on_rebalance_start(self, app: AppT) -> Optional[Dict[str, Any]]:
    def on_rebalance_return(self, app: AppT, state: Dict[str, Any]) -> None:
    def on_rebalance_end(self, app: AppT, state: Dict[str, Any]) -> None:
    def count(self, metric_name: str, count: int = 1) -> None:
    def on_tp_commit(self, tp_offsets: TPOffsetMapping) -> None:
    def track_tp_end_offset(self, tp: TP, offset: int) -> None:
    def on_web_request_end(self, app: AppT, request: web.Request, response: web.Response, state: Dict[str, Any], *, view: Optional[web.View] = None) -> None:
    @cached_property
    def client(self) -> StatsClient:
