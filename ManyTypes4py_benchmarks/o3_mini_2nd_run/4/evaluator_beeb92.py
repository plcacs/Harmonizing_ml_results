#!/usr/bin/env python3
"""
Evaluator class for evaluating a model with a given dataset
"""

from typing import Union, Dict, Any, Optional, TextIO
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
    """
    Evaluation Base class

    # Parameters

    batch_serializer: `Serializer`, optional (default=`SimpleSerializer`)
        The serializer to use for turning both the batches and the outputs
        of the model into human readable data.

    cuda_device : `Union[int, torch.device]`, optional (default=`-1`)
        The cuda device to use for this evaluation.  The model is assumed to
        already be using this device; this parameter is only used for moving
        the input data to the correct device.

    postprocessor_fn_name: `str`, optional (default=`"make_output_human_readable"`)
        Function name of the model's postprocessing function.
    """
    default_implementation = 'simple'

    def __init__(
        self,
        batch_serializer: Optional[Serializer] = None,
        cuda_device: Union[int, torch.device] = -1,
        postprocessor_fn_name: str = 'make_output_human_readable'
    ) -> None:
        self.batch_serializer: Serializer = batch_serializer or SimpleSerializer()
        self.cuda_device: Union[int, torch.device] = cuda_device
        self.postprocessor_fn_name: str = postprocessor_fn_name

    def __call__(
        self,
        model: Model,
        data_loader: DataLoader,
        batch_weight_key: Optional[str] = None,
        metrics_output_file: Optional[Union[str, PathLike]] = None,
        predictions_output_file: Optional[Union[str, PathLike]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single data source.

        # Parameters

        model : `Model`
            The model to evaluate
        data_loader : `DataLoader`
            The `DataLoader` that will iterate over the evaluation data (data loaders already contain
            their data).
        batch_weight_key : `str`, optional (default=`None`)
            If given, this is a key in the output dictionary for each batch that specifies how to weight
            the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
        metrics_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the final metrics to.

        predictions_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the predictions to. If passed the
            postprocessor will be called and its output will be written as lines.

        # Returns

        metrics: `Dict[str, Any]`
            The metrics from evaluating the file.
        """
        raise NotImplementedError('__call__')

@Evaluator.register('simple')
class SimpleEvaluator(Evaluator):
    """
    Simple evaluator implementation. Uses the vanilla evaluation code.

    # Parameters

    batch_serializer: `Serializer`, optional (default=`SimpleSerializer`)
        The serializer to use for turning both the batches and the outputs
        of the model into human readable data.

    cuda_device : `Union[int, torch.device]`, optional (default=`-1`)
        The cuda device to use for this evaluation.  The model is assumed to
        already be using this device; this parameter is only used for moving
        the input data to the correct device.

    postprocessor_fn_name: `str`, optional (default=`"make_output_human_readable"`)
        Function name of the model's postprocessing function.
    """

    def __init__(
        self,
        batch_serializer: Optional[Serializer] = None,
        cuda_device: Union[int, torch.device] = -1,
        postprocessor_fn_name: str = 'make_output_human_readable'
    ) -> None:
        super(SimpleEvaluator, self).__init__(batch_serializer, cuda_device, postprocessor_fn_name)

    def __call__(
        self,
        model: Model,
        data_loader: DataLoader,
        batch_weight_key: Optional[str] = None,
        metrics_output_file: Optional[Union[str, PathLike]] = None,
        predictions_output_file: Optional[Union[str, PathLike]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single data source.

        # Parameters

        model : `Model`
            The model to evaluate
        data_loader : `DataLoader`
            The `DataLoader` that will iterate over the evaluation data (data loaders already contain
            their data).
        batch_weight_key : `str`, optional (default=`None`)
            If given, this is a key in the output dictionary for each batch that specifies how to weight
            the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
        metrics_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the final metrics to.
        predictions_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the predictions to.

        # Returns

        metrics: `Dict[str, Any]`
            The metrics from evaluating the file.
        """
        check_for_gpu(self.cuda_device)
        data_loader.set_target_device(int_to_device(self.cuda_device))
        metrics_output_file_path: Optional[Path] = Path(metrics_output_file) if metrics_output_file is not None else None
        predictions_file: Optional[TextIO] = None
        if predictions_output_file is not None:
            predictions_file = Path(predictions_output_file).open('w', encoding='utf-8')
        model_postprocess_function = getattr(model, self.postprocessor_fn_name, None)
        with torch.no_grad():
            model.eval()
            iterator = iter(data_loader)
            logger.info('Iterating over dataset')
            generator_tqdm = Tqdm.tqdm(iterator)
            batch_count: int = 0
            loss_count: int = 0
            total_loss: float = 0.0
            total_weight: float = 0.0
            for batch in generator_tqdm:
                batch_count += 1
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict: Dict[str, Any] = model(**batch)
                loss = output_dict.get('loss')
                metrics: Dict[str, float] = model.get_metrics()
                if loss is not None:
                    loss_count += 1
                    if batch_weight_key:
                        weight: float = output_dict[batch_weight_key].item()
                    else:
                        weight = 1.0
                    total_weight += weight
                    total_loss += loss.item() * weight
                    metrics['loss'] = total_loss / total_weight
                description: str = ', '.join(['%s: %.2f' % (name, value) for name, value in metrics.items() if not name.startswith('_')]) + ' ||'
                generator_tqdm.set_description(description, refresh=False)
                if predictions_file is not None:
                    predictions_file.write(
                        self.batch_serializer(batch, output_dict, data_loader, output_postprocess_function=model_postprocess_function) + '\n'
                    )
            if predictions_file is not None:
                predictions_file.close()
            final_metrics: Dict[str, Any] = model.get_metrics(reset=True)
            if loss_count > 0:
                if loss_count != batch_count:
                    raise RuntimeError('The model you are trying to evaluate only sometimes produced a loss!')
                final_metrics['loss'] = total_loss / total_weight
            if metrics_output_file_path is not None:
                dump_metrics(str(metrics_output_file_path), final_metrics, log=True)
            return final_metrics

    def _to_params(self) -> Dict[str, Any]:
        return {
            'type': 'simple',
            'cuda_device': self.cuda_device,
            'batch_serializer': self.batch_serializer.to_params()
        }