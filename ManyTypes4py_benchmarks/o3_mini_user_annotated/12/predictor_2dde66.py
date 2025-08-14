from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional, Callable
import logging
from os import PathLike
import json
import re
from contextlib import contextmanager

import numpy
import torch
from torch.utils.hooks import RemovableHandle
from torch import Tensor, nn
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
    A `Predictor` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        if frozen:
            model.eval()
        self._model: Model = model
        self._dataset_reader: DatasetReader = dataset_reader
        self.cuda_device: int = next(self._model.named_parameters())[1].get_device()
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
        return json.dumps(outputs) + "\n"

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance: Instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        """
        Converts incoming json to an [`Instance`](../data/instance.md),
        runs the model on the newly created instance, and adds labels to the
        `Instance`s given by the model's output.

        # Returns

        `List[Instance]`
            A list of `Instance`s.
        """
        instance: Instance = self._json_to_instance(inputs)
        self._dataset_reader.apply_token_indexers(instance)
        outputs: Dict[str, Any] = self._model.forward_on_instance(instance)  # type: ignore
        new_instances: List[Instance] = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        # Parameters
        instances : `List[Instance]`

        # Returns
        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a dict of gradient entries for each input.
            The keys have the form `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.
        """
        original_param_name_to_requires_grad_dict: Dict[str, bool] = {}
        for param_name, param in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)

        dataset: Batch = Batch(instances)
        dataset.index_instances(self._model.vocab)
        dataset_tensor_dict: Dict[str, Any] = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        with backends.cudnn.flags(enabled=False):
            outputs: Dict[str, Any] = self._model.make_output_human_readable(
                self._model.forward(**dataset_tensor_dict)  # type: ignore
            )
            loss: Tensor = outputs["loss"]
            for p in self._model.parameters():
                p.grad = None
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict: Dict[str, Any] = {}
        for idx, grad in enumerate(embedding_gradients):
            key: str = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        for param_name, param in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return grad_dict, outputs

    def get_interpretable_layer(self) -> nn.Module:
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
            raise RuntimeError(
                "If the model does not use `TextFieldEmbedder`, please override "
                "`get_interpretable_layer` in your predictor to specify the embedding layer."
            )

    def get_interpretable_text_field_embedder(self) -> nn.Module:
        """
        Returns the first `TextFieldEmbedder` of the model.
        If the predictor wraps around a non-AllenNLP model,
        this function should be overridden to specify the correct embedder.
        """
        try:
            return util.find_text_field_embedder(self._model)
        except RuntimeError:
            raise RuntimeError(
                "If the model does not use `TextFieldEmbedder`, please override "
                "`get_interpretable_text_field_embedder` in your predictor to specify "
                "the embedding layer."
            )

    def _register_embedding_gradient_hooks(self, embedding_gradients: List[Tensor]) -> List[RemovableHandle]:
        """
        Registers a backward hook on the embedding layer of the model. Used to save the gradients
        of the embeddings for use in get_gradients().

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.

        We additionally add a hook on the _forward_ pass of the model's `TextFieldEmbedder` to save
        token offsets, if there are any.
        """
        def hook_layers(module: nn.Module, grad_in: Tuple[Optional[Tensor], ...], grad_out: Tuple[Optional[Tensor], ...]) -> None:
            grads: Tensor = grad_out[0]  # type: ignore
            if self._token_offsets:
                offsets: Tensor = self._token_offsets.pop(0)
                span_grads, span_mask = util.batched_span_select(grads.contiguous(), offsets)
                span_mask = span_mask.unsqueeze(-1)
                span_grads *= span_mask
                span_grads_sum: Tensor = span_grads.sum(2)
                span_grads_len: Tensor = span_mask.sum(2)
                grads = span_grads_sum / torch.clamp_min(span_grads_len, 1)
                grads[(span_grads_len == 0).expand(grads.shape)] = 0
            embedding_gradients.append(grads)

        def get_token_offsets(module: nn.Module, inputs: Any, outputs: Any) -> None:
            offsets: Optional[Tensor] = util.get_token_offsets_from_text_field_inputs(inputs)
            if offsets is not None:
                self._token_offsets.append(offsets)

        hooks: List[RemovableHandle] = []
        text_field_embedder: nn.Module = self.get_interpretable_text_field_embedder()
        hooks.append(text_field_embedder.register_forward_hook(get_token_offsets))
        embedding_layer: nn.Module = self.get_interpretable_layer()
        hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return hooks

    @contextmanager
    def capture_model_internals(self, module_regex: str = ".*") -> Iterator[Dict[str, Any]]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

            with predictor.capture_model_internals() as internals:
                outputs = predictor.predict_json(inputs)

            return {**outputs, "model_internals": internals}
        """
        results: Dict[str, Any] = {}
        hooks: List[RemovableHandle] = []

        def add_output(idx: int) -> Callable[[nn.Module, Any, Any], None]:
            def _add_output(mod: nn.Module, _inputs: Any, outputs: Any) -> None:
                results[str(idx)] = {"name": str(mod), "output": sanitize(outputs)}
            return _add_output

        regex = re.compile(module_regex)
        for idx, (name, module) in enumerate(self._model.named_modules()):
            if regex.fullmatch(name) and module != self._model:
                hook = module.register_forward_hook(add_output(idx))
                hooks.append(hook)

        yield results

        for hook in hooks:
            hook.remove()

    def predict_instance(self, instance: Instance) -> JsonDict:
        self._dataset_reader.apply_token_indexers(instance)
        outputs: Dict[str, Any] = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        """
        This function takes a model's outputs for an Instance, and labels that instance according
        to the `outputs`. This function is used to (1) compute gradients of what the model predicted;
        (2) label the instance for the attack. For example, (a) for the untargeted attack for classification
        this function labels the instance according to the class with the highest probability; (b) for
        targeted attack, it directly constructs fields from the given target.
        The return type is a list because in some tasks there are multiple predictions in the output
        (e.g., in NER a model predicts multiple spans). In this case, each instance in the returned list of
        Instances contains an individual entity prediction as the label.
        """
        raise RuntimeError("implement this method for model interpretations or attacks")

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an [`Instance`](../data/instance.md)
        and a `JsonDict` of information which the `Predictor` should pass through,
        such as tokenized inputs.
        """
        raise NotImplementedError

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances: List[Instance] = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
        outputs: List[Dict[str, Any]] = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        """
        Converts a list of JSON objects into a list of `Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by `predict_json`. In order to use this method for
        batch prediction, `_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """
        instances: List[Instance] = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    @classmethod
    def from_path(
        cls,
        archive_path: Union[str, PathLike],
        predictor_name: Optional[str] = None,
        cuda_device: int = -1,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
        import_plugins: bool = True,
        overrides: Union[str, Dict[str, Any]] = "",
        **kwargs: Any,
    ) -> "Predictor":
        """
        Instantiate a `Predictor` from an archive path.
        """
        if import_plugins:
            plugins.import_plugins()
        return Predictor.from_archive(
            load_archive(archive_path, cuda_device=cuda_device, overrides=overrides),
            predictor_name,
            dataset_reader_to_load=dataset_reader_to_load,
            frozen=frozen,
            extra_args=kwargs,
        )

    @classmethod
    def from_archive(
        cls,
        archive: Archive,
        predictor_name: Optional[str] = None,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> "Predictor":
        """
        Instantiate a `Predictor` from an [`Archive`](../models/archival.md).
        """
        config = archive.config.duplicate()

        if not predictor_name:
            model_type: str = config.get("model").get("type")
            model_class, _ = Model.resolve_class_name(model_type)
            predictor_name = model_class.default_predictor
        predictor_class: Type[Predictor] = (
            Predictor.by_name(predictor_name) if predictor_name is not None else cls  # type: ignore
        )

        if dataset_reader_to_load == "validation":
            dataset_reader: DatasetReader = archive.validation_dataset_reader  # type: ignore
        else:
            dataset_reader = archive.dataset_reader  # type: ignore

        model: Model = archive.model
        if frozen:
            model.eval()

        if extra_args is None:
            extra_args = {}

        return predictor_class(model, dataset_reader, **extra_args)
