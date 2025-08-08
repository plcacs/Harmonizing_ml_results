from typing import Dict, List, Set, Type, Optional, Union

class Model(Module, Registrable):
    def __init__(self, vocab: Vocabulary, regularizer: Optional[RegularizerApplicator] = None, serialization_dir: Optional[str] = None, ddp_accelerator: Optional[DdpAccelerator] = None) -> None:
    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
    def get_parameters_for_histogram_logging(self) -> List[str]:
    def get_parameters_for_histogram_tensorboard_logging(self) -> List[str]:
    def forward(self, *inputs: Any) -> Dict[str, torch.Tensor]:
    def forward_on_instance(self, instance: Instance) -> Dict[str, numpy.ndarray]:
    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    def _get_prediction_device(self) -> int:
    def _maybe_warn_for_unseparable_batches(self, output_key: str) -> None:
    @classmethod
    def _load(cls, config: Params, serialization_dir: str, weights_file: Optional[str] = None, cuda_device: int = -1) -> Model:
    @classmethod
    def load(cls, config: Params, serialization_dir: str, weights_file: Optional[str] = None, cuda_device: int = -1) -> Model:
    def extend_embedder_vocab(self, embedding_sources_mapping: Optional[Dict[str, str]] = None) -> None:
    @classmethod
    def from_archive(cls, archive_file: str, vocab: Optional[Vocabulary] = None) -> Model:

def remove_weights_related_keys_from_params(params: Params, keys: List[str] = ['pretrained_file', 'initializer']) -> None:
def remove_pretrained_embedding_params(params: Params) -> None:
