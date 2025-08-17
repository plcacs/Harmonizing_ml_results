from collections import defaultdict
import inspect
from typing import Any, Dict, List, Set, Union, Mapping, Optional, cast

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import Backbone
from allennlp.models.model import Model
from allennlp.models.heads import Head
from allennlp.nn import InitializerApplicator


def get_forward_arguments(module: torch.nn.Module) -> Set[str]:
    signature = inspect.signature(module.forward)
    return set([arg for arg in signature.parameters if arg != "self"])


@Model.register("multitask")
class MultiTaskModel(Model):
    default_predictor: str = "multitask"

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
        **kwargs: Any,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._backbone: Backbone = backbone
        self._heads: torch.nn.ModuleDict = torch.nn.ModuleDict(heads)
        self._heads_called: Set[str] = set()
        self._arg_name_mapping: Dict[str, Dict[str, str]] = arg_name_mapping or defaultdict(dict)

        self._allowed_arguments: Dict[str, Set[str]] = allowed_arguments or {
            "backbone": get_forward_arguments(backbone),
            **{key: get_forward_arguments(heads[key]) for key in heads},
        }
        self._loss_weights: Dict[str, float] = loss_weights or defaultdict(lambda: 1.0)
        initializer(self)

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:  # type: ignore
        if "task" not in kwargs:
            raise ValueError(
                "Instances for multitask training need to contain a MetadataField with "
                "the name 'task' to indicate which task they belong to. Usually the "
                "MultitaskDataLoader provides this field and you don't have to do anything."
            )

        task_indices_just_for_mypy: Mapping[str, List[int]] = defaultdict(lambda: [])
        for i, task in enumerate(kwargs["task"]):
            task_indices_just_for_mypy[task].append(i)
        task_indices: Dict[str, torch.LongTensor] = {
            task: torch.LongTensor(indices) for task, indices in task_indices_just_for_mypy.items()
        }

        def make_inputs_for_task(
            task: str, whole_batch_input: Union[torch.Tensor, TextFieldTensors, List[Any]]
        ) -> Any:
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

        backbone_arguments: Dict[str, Any] = self._get_arguments(kwargs, "backbone")
        backbone_outputs: Dict[str, torch.Tensor] = self._backbone(**backbone_arguments)
        combined_arguments: Dict[str, Any] = {**backbone_outputs, **kwargs}

        outputs: Dict[str, torch.Tensor] = {**backbone_outputs}
        loss: Optional[torch.Tensor] = None
        for head_name in self._heads:
            if head_name not in task_indices:
                continue

            head_arguments: Dict[str, Any] = self._get_arguments(combined_arguments, head_name)
            head_arguments = {
                key: make_inputs_for_task(head_name, value) for key, value in head_arguments.items()
            }

            head_outputs: Dict[str, torch.Tensor] = self._heads[head_name](**head_arguments)
            for key in head_outputs:
                outputs[f"{head_name}_{key}"] = head_outputs[key]

            if "loss" in head_outputs:
                self._heads_called.add(head_name)
                head_loss: torch.Tensor = self._loss_weights[head_name] * head_outputs["loss"]
                if loss is None:
                    loss = head_loss
                else:
                    loss += head_loss

        if loss is not None:
            outputs["loss"] = loss

        return outputs

    def _get_arguments(self, available_args: Dict[str, Any], component: str) -> Dict[str, Any]:
        allowed_args: Set[str] = self._allowed_arguments[component]
        name_mapping: Dict[str, str] = self._arg_name_mapping.get(component, {})
        kept_arguments: Dict[str, Any] = {}
        for key, value in available_args.items():
            new_key: str = name_mapping.get(key, key)
            if new_key in allowed_args:
                if new_key in kept_arguments:
                    raise ValueError(
                        f"Got duplicate argument {new_key} for {component}. This likely means that"
                        " you mapped multiple inputs to the same name. This is generally ok for"
                        " the backbone, but you have to be sure each batch only gets one of those"
                        " inputs. This is typically not ok for heads, and means something is not"
                        " set up right."
                    )
                kept_arguments[new_key] = value
        return kept_arguments

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for head_name in self._heads_called:
            for key, value in self._heads[head_name].get_metrics(reset).items():
                metrics[f"{head_name}_{key}"] = value
        if reset:
            self._heads_called.clear()
        return metrics

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        output_dict = self._backbone.make_output_human_readable(output_dict)
        for head_name, head in self._heads.items():
            head_outputs: Dict[str, torch.Tensor] = {}
            for key, value in output_dict.items():
                if key.startswith(head_name):
                    head_outputs[key.replace(f"{head_name}_", "")] = value
            readable_head_outputs: Dict[str, torch.Tensor] = head.make_output_human_readable(head_outputs)
            for key, value in readable_head_outputs.items():
                output_dict[f"{head_name}_{key}"] = value
        return output_dict
