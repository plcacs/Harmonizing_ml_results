from collections import defaultdict
import inspect
from typing import Any, Dict, List, Set, Union, Mapping, Optional
import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import Backbone
from allennlp.models.model import Model
from allennlp.models.heads import Head
from allennlp.nn import InitializerApplicator

def get_forward_arguments(module: torch.nn.Module) -> Set[str]:
    signature = inspect.signature(module.forward)
    return set([arg for arg in signature.parameters if arg != 'self'])

@Model.register('multitask')
class MultiTaskModel(Model):
    """
    A `MultiTaskModel` consists of a `Backbone` that encodes its inputs in some way, then a
    collection of `Heads` that make predictions from the backbone-encoded inputs. The predictions
    of each `Head` are combined to compute a joint loss, which is then used for training.

    This model works by taking `**kwargs` in `forward`, and passing the right arguments from that to
    the backbone and to each head. By default, we use `inspect` to try to figure out getting the
    right arguments to the right modules, but we allow you to specify these arguments yourself in
    case our inference code gets it wrong.

    It is the caller's responsibility to make sure that the backbone and all heads are compatible with
    each other, and with the input data that comes from a `MultiTaskDatasetReader`. We give some
    arguments in this class and in `MultiTaskDatasetReader` to help with plumbing the arguments in
    complex cases (e.g., you can change argument names so that they match what the backbone and
    heads expect).

    # Parameters

    vocab: `Vocab`
    backbone: `Backbone`
    heads: `Dict[str, Head]`
    loss_weights: `Dict[str, float]`, optional (default = `equal weighting`)
        If you want, you can specify a weight for each head, which we will multiply the loss by when
        aggregating across heads. This is equivalent in many cases to specifying a separate
        learning rate per head, and just putting a weighting on the loss is much easier than
        figuring out the right way to specify that in the optimizer.
    arg_name_mapping: `Dict[str, Dict[str, str]]`, optional (default = `identity mapping`)
        The mapping changes the names in the `**kwargs` dictionary passed to `forward` before
        passing on the arguments to the backbone and heads. This is keyed by component, and the
        top-level keys must match the keys passed in the `heads` parameter, plus a "backbone" key
        for the backbone. If you are using dataset readers that use dataset-specific names for
        their keys, this lets you change them to be consistent. For example, this dictionary might
        end up looking like this: `{"backbone": {"question": "text", "review": "text"},
        "classifier1": {"sentiment": "label"}, "classifier2": {"topic": "label"}}`.
        Though in this particular example, we have two different inputs mapping to the same key in
        the backbone; this will work, as long are you are careful that you don't give both of those
        inputs in the same batch. If we see overlapping keys, we will crash. If you want to be able
        to do this kind of mixed training in the same batch, you need to handle that in your data
        code, not here; we won't handle complex batching inside this model.
    allowed_arguments: `Dict[str, Set[str]]`, optional (default = `inferred`)
        The list of arguments that should be passed from `**kwargs` to the `forward` method for the
        backbone and each head. If you provide this, the keys in here should match the keys given
        in the `heads` parameter, plus a "backbone" key for the backbone arguments. If not given,
        we will use the `inspect` module to figure this out. The only time that this inference
        might fail is if you have optional arguments that you want to be ignored, or
        something. You very likely don't need to worry about this argument.
    initializer: `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """
    default_predictor: str = 'multitask'

    def __init__(
        self,
        vocab: Vocabulary,
        backbone: Backbone,
        heads: Dict[str, Head],
        *,
        loss_weights: Optional[Dict[str, float]] = None,
        arg_name_mapping: Optional[Dict[str, Dict[str, str]]] = None,
        allowed_arguments: Optional[Dict[str, Set[str]]] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs: Any
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._backbone: Backbone = backbone
        self._heads: torch.nn.ModuleDict = torch.nn.ModuleDict(heads)
        self._heads_called: Set[str] = set()
        self._arg_name_mapping: Dict[str, Dict[str, str]] = arg_name_mapping or defaultdict(dict)
        inferred_allowed_arguments: Dict[str, Set[str]] = {'backbone': get_forward_arguments(backbone)}
        inferred_allowed_arguments.update({key: get_forward_arguments(heads[key]) for key in heads})
        self._allowed_arguments: Dict[str, Set[str]] = allowed_arguments or inferred_allowed_arguments
        self._loss_weights: Dict[str, float] = loss_weights or defaultdict(lambda: 1.0)
        initializer(self)

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        if 'task' not in kwargs:
            raise ValueError("Instances for multitask training need to contain a MetadataField with the name 'task' to indicate which task they belong to. Usually the MultitaskDataLoader provides this field and you don't have to do anything.")
        task_indices_just_for_mypy: defaultdict = defaultdict(list)  # type: ignore
        for i, task in enumerate(kwargs['task']):
            task_indices_just_for_mypy[task].append(i)
        task_indices: Dict[str, torch.LongTensor] = {task: torch.LongTensor(indices) for task, indices in task_indices_just_for_mypy.items()}

        def make_inputs_for_task(task: str, whole_batch_input: Any) -> Any:
            if isinstance(whole_batch_input, dict):
                for k1, v1 in whole_batch_input.items():
                    for k2, v2 in v1.items():
                        whole_batch_input[k1][k2] = make_inputs_for_task(task, v2)
                return whole_batch_input
            if isinstance(whole_batch_input, torch.Tensor):
                task_indices[task] = task_indices[task].to(whole_batch_input.device)
                return torch.index_select(whole_batch_input, 0, task_indices[task])
            else:
                return [whole_batch_input[i] for i in task_indices[task]]

        backbone_arguments: Dict[str, Any] = self._get_arguments(kwargs, 'backbone')
        backbone_outputs: Dict[str, Any] = self._backbone(**backbone_arguments)
        combined_arguments: Dict[str, Any] = {**backbone_outputs, **kwargs}
        outputs: Dict[str, Any] = {**backbone_outputs}
        loss: Optional[torch.Tensor] = None
        for head_name in self._heads:
            if head_name not in task_indices:
                continue
            head_arguments: Dict[str, Any] = self._get_arguments(combined_arguments, head_name)
            head_arguments = {key: make_inputs_for_task(head_name, value) for key, value in head_arguments.items()}
            head_outputs: Dict[str, Any] = self._heads[head_name](**head_arguments)
            for key, value in head_outputs.items():
                outputs[f'{head_name}_{key}'] = value
            if 'loss' in head_outputs:
                self._heads_called.add(head_name)
                head_loss: torch.Tensor = self._loss_weights[head_name] * head_outputs['loss']
                if loss is None:
                    loss = head_loss
                else:
                    loss += head_loss
        if loss is not None:
            outputs['loss'] = loss
        return outputs

    def _get_arguments(self, available_args: Dict[str, Any], component: str) -> Dict[str, Any]:
        """
        Given a list of things we might want to pass to a component (where "component" is either the
        backbone or a head), this method figures out which things we should actually pass, by
        mapping names and looking at allowed arguments.
        """
        allowed_args: Set[str] = self._allowed_arguments[component]
        name_mapping: Dict[str, str] = self._arg_name_mapping.get(component, {})
        kept_arguments: Dict[str, Any] = {}
        for key, value in available_args.items():
            new_key: str = name_mapping.get(key, key)
            if new_key in allowed_args:
                if new_key in kept_arguments:
                    raise ValueError(f'Got duplicate argument {new_key} for {component}. This likely means that you mapped multiple inputs to the same name. This is generally ok for the backbone, but you have to be sure each batch only gets one of those inputs. This is typically not ok for heads, and means something is not set up right.')
                kept_arguments[new_key] = value
        return kept_arguments

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for head_name in self._heads_called:
            head_metrics: Dict[str, float] = self._heads[head_name].get_metrics(reset)
            for key, value in head_metrics.items():
                metrics[f'{head_name}_{key}'] = value
        if reset:
            self._heads_called.clear()
        return metrics

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        output_dict = self._backbone.make_output_human_readable(output_dict)
        for head_name, head in self._heads.items():
            head_outputs: Dict[str, Any] = {}
            for key, value in output_dict.items():
                if key.startswith(f'{head_name}_'):
                    head_outputs[key[len(f'{head_name}_'):]] = value
            readable_head_outputs: Dict[str, Any] = head.make_output_human_readable(head_outputs)
            for key, value in readable_head_outputs.items():
                output_dict[f'{head_name}_{key}'] = value
        return output_dict
