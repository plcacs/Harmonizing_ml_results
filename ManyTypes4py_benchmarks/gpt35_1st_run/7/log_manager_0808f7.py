    def __init__(self, n_batches_per_epoch: int, log_writer: Optional[LogWriter] = None, checkpointer: Optional[Checkpointer] = None, **kwargs: Any) -> None:
    def update(self, batch_size: int) -> None:
    def trigger_evaluation(self) -> bool:
    def trigger_checkpointing(self) -> bool:
    def cleanup(self, model: MultitaskClassifier) -> MultitaskClassifier:
