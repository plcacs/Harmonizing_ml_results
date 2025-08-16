    def __init__(self, serialization_dir: str, summary_interval: int = 100, distribution_interval: Optional[int] = None, batch_size_interval: Optional[int] = None, should_log_parameter_statistics: bool = True, should_log_learning_rate: bool = False, project: Optional[str] = None, entity: Optional[str] = None, group: Optional[str] = None, name: Optional[str] = None, notes: Optional[str] = None, tags: Optional[List[str]] = None, watch_model: bool = True, files_to_save: Tuple[str, ...] = ('config.json', 'out.log'), wandb_kwargs: Optional[Dict[str, Any]] = None) -> None:

    def log_scalars(self, scalars: Dict[str, Union[int, float]], log_prefix: str = '', epoch: Optional[int] = None) -> None:

    def log_tensors(self, tensors: Dict[str, torch.Tensor], log_prefix: str = '', epoch: Optional[int] = None) -> None:

    def _log(self, dict_to_log: Dict[str, Any], log_prefix: str = '', epoch: Optional[int] = None) -> None:

    def on_start(self, trainer: 'GradientDescentTrainer', is_primary: bool = True, **kwargs: Any) -> None:

    def state_dict(self) -> Dict[str, Any]:

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
