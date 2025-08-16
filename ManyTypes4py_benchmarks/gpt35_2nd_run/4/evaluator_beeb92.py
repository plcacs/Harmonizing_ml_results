from typing import Union, Dict, Any, Optional
from os import PathLike
from pathlib import Path
import torch
import logging
from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, int_to_device
from allennlp.nn import util as nn_util
from allennlp.common import Registrable
from allennlp.models import Model
from allennlp.data import DataLoader
from allennlp.evaluation.serializers.serializers import Serializer, SimpleSerializer
logger = logging.getLogger(__name__)

class Evaluator(Registrable):
    def __init__(self, batch_serializer: Optional[Serializer] = None, cuda_device: Union[int, torch.device] = -1, postprocessor_fn_name: str = 'make_output_human_readable') -> None:
    def __call__(self, model: Model, data_loader: DataLoader, batch_weight_key: Optional[str] = None, metrics_output_file: Optional[Union[str, PathLike]] = None, predictions_output_file: Optional[Union[str, PathLike]] = None) -> Dict[str, Any]:
    def _to_params(self) -> Dict[str, Any]:

@Evaluator.register('simple')
class SimpleEvaluator(Evaluator):
    def __init__(self, batch_serializer: Optional[Serializer] = None, cuda_device: Union[int, torch.device] = -1, postprocessor_fn_name: str = 'make_output_human_readable') -> None:
    def __call__(self, model: Model, data_loader: DataLoader, batch_weight_key: Optional[str] = None, metrics_output_file: Optional[Union[str, PathLike]] = None, predictions_output_file: Optional[Union[str, PathLike]] = None) -> Dict[str, Any]:
