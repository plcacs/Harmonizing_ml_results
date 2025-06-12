from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional, ContextManager
import logging
from os import PathLike
import json
import re
from contextlib import contextmanager
import numpy
import torch
from torch.utils.hooks import RemovableHandle
from torch import Tensor
from torch import backends
from allennlp.common import Registrable, plugins
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util

logger = logging.getLogger(__name__)

class Predictor(Registrable):
    """
    a `Predictor` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        if frozen:
            model.eval()
        self._model = model
        self._dataset_reader = dataset_reader
        self.cuda_device = next(self._model.named_parameters())[1].get_device()
        self._token_offsets: List[Tensor] = []

    def load_line(self, line: str) -> JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return json.loads(line)

    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs) + '\n'

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        """
        Converts incoming json to a [`Instance`](../data/instance.md),
        runs the model on the newly created instance, and adds labels to the
        `Instance`s given by the model's output.

        # Returns

        `List[instance]`
            A list of `Instance`'s.
        """
        instance = self._json_to_instance(inputs)
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        # Parameters

        instances : `List[Instance]`

        # Returns

        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a Dict of gradient entries for each input.
            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.

        # Notes

        Takes a `JsonDict` representing the inputs of the model and converts
        them to [`Instances`](../data/instance.md)), sends these through
        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding
        layer of the model. Calls `backward` on the loss and then removes the
        hooks.
        """
        original_param_name_to_requires_grad_dict: Dict[str, bool] = {}
        for param_name, param in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True
        embedding_gradients: List[Tensor] = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        dataset_tensor_dict = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        with backends.cudnn.flags(enabled=False):
            outputs = self._model.make_output_human_readable(self._model.forward(**dataset_tensor_dict))
            loss = outputs['loss']
            for p in self._model.parameters():
                p.grad = None
            loss.backward()
        for hook in hooks:
            hook.remove()
        grad_dict: Dict[str, numpy.ndarray] = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = 'grad_input_' + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()
        for param_name, param in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
        return (grad_dict, outputs)

    def get_interpretable_layer(self) -> torch.nn.Module:
        """
        Returns the input/embedding layer of the model.
        If the predictor wraps around a non-AllenNLP model,
        this function should be overridden to specify the correct input/embedding layer.
        For the cases where the input layer _is_ an embedding layer, this should be the
        layer 0 of the embedder.
        """
        try:
            return util.find_embedding_layer(self._model)
        except RuntimeError:
            raise RuntimeError('If the model does not use `TextFieldEmbedder`, please override `get_interpretable_layer` in your predictor to specify the embedding layer.')

    def get_interpretable_text_field_embedder(self) -> torch.nn.Module:
        """
        Returns the first `TextFieldEmbedder` of the model.
        If the predictor wraps around a non-AllenNLP model,
        this function should be overridden to specify the correct embedder.
        """
        try:
            return util.find_text_field_embedder(self._model)
        except RuntimeError:
            raise RuntimeError('If the model does not use `TextFieldEmbedder`, please override `get_interpretable_text_field_embedder` in your predictor to specify the embedding layer.')

    def _register_embedding_gradient_hooks(self, embedding_gradients: List[Tensor]) -> List[RemovableHandle]:
        """
        Registers a backward hook on the embedding layer of the model.  Used to save the gradients
        of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.

        We additionally add a hook on the _forward_ pass of the model's `TextFieldEmbedder` to save
        token offsets, if there are any.  Having token offsets means that you're using a mismatched
        token indexer, so we need to aggregate the gradients across wordpieces in a token.  We do
        that with a simple sum.
        """

        def hook_layers(module: torch.nn.Module, grad_in: Tuple[Tensor, ...], grad_out: Tuple[Tensor, ...]) -> None:
            grads = grad_out[0]
            if self._token_offsets:
                offsets = self._token_offsets.pop(0)
                span_grads, span_mask = util.batched_span_select(grads.contiguous(), offsets)
                span_mask = span_mask.unsqueeze(-1)
                span_grads *= span_mask
                span_grads_sum = span_grads.sum(2)
                span_grads_len = span_mask.sum(2)
                grads = span_grads_sum / torch.clamp_min(span_grads_len, 1)
                grads[(span_grads_len == 0).expand(grads.shape)] = 0
            embedding_gradients.append(grads)

        def get_token_offsets(module: torch.nn.Module, inputs: Tuple[Tensor, ...], outputs: Tensor) -> None:
            offsets = util.get_token_offsets_from_text_field_inputs(inputs)
            if offsets is not None:
                self._token_offsets.append(offsets)
        hooks: List[RemovableHandle] = []
        text_field_embedder = self.get_interpretable_text_field_embedder()
        hooks.append(text_field_embedder.register_forward_hook(get_token_offsets))
        embedding_layer = self.get_interpretable_layer()
        hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return hooks

    @contextmanager
    def capture_model_internals(self, module_regex: str = '.*') -> ContextManager[Dict[int, Dict[str, Any]]]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

        