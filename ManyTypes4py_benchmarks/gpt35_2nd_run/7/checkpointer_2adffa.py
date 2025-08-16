    def __init__(self, serialization_dir: str, save_completed_epochs: bool = True, save_every_num_seconds: Optional[int] = None, save_every_num_batches: Optional[int] = None, keep_most_recent_by_count: int = 2, keep_most_recent_by_age: Optional[int] = None) -> None:
    def _model_state_path(self, epochs_completed: int, batches_in_epoch_completed: int) -> str:
    def _training_state_path(self, epochs_completed: int, batches_in_epoch_completed: int) -> str:
    @classmethod
    def _parse_model_state_path(cls, path: Union[str, os.PathLike]) -> Optional[Tuple[int, int]]:
    @classmethod
    def _parse_training_state_path(cls, path: Union[str, os.PathLike]) -> Optional[Tuple[int, int]]:
    def _find_all_checkpoints(self) -> Set[Tuple[int, int]]:
    def _remove_checkpoint(self, epochs_completed: int, batches_in_epoch_completed: int) -> None:
    def maybe_save_checkpoint(self, trainer: Trainer, num_epochs_completed: int, num_batches_in_epoch_completed: int) -> bool:
    def save_checkpoint(self, trainer: Trainer) -> None:
    def find_latest_checkpoint(self) -> Optional[Tuple[str, str]]:
    def load_checkpoint(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
