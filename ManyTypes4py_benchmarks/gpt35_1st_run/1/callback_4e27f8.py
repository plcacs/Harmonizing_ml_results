class TrainerCallback(Registrable):
    def __init__(self, serialization_dir: str) -> None:
        self.serialization_dir: str = serialization_dir
        self.trainer: Optional['GradientDescentTrainer'] = None

    def on_start(self, trainer: 'GradientDescentTrainer', is_primary: bool = True, **kwargs: Any) -> None:
        pass

    def on_backward(self, trainer: 'GradientDescentTrainer', batch_outputs: TensorDict, backward_called: bool, **kwargs: Any) -> bool:
        return False

    def on_batch(self, trainer: 'GradientDescentTrainer', batch_inputs: TensorDict, batch_outputs: TensorDict, batch_metrics: Dict[str, Any], epoch: int, batch_number: int, is_training: bool, is_primary: bool = True, batch_grad_norm: Optional[float] = None, **kwargs: Any) -> None:
        pass

    def on_epoch(self, trainer: 'GradientDescentTrainer', metrics: Dict[str, Any], epoch: int, is_primary: bool = True, **kwargs: Any) -> None:
        pass

    def on_end(self, trainer: 'GradientDescentTrainer', metrics: Optional[Dict[str, Any]] = None, epoch: Optional[int] = None, is_primary: bool = True, **kwargs: Any) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
