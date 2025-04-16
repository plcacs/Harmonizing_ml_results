from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any
import heapq

import numpy as np
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.predictors import Predictor


@Attacker.register("input-reduction")
class InputReduction(Attacker):
    def __init__(self, predictor: Predictor, beam_size: int = 3) -> None:
        super().__init__(predictor)
        self.beam_size = beam_size

    def attack_from_json(
        self,
        inputs: JsonDict,
        input_field_to_attack: str = "tokens",
        grad_input_field: str = "grad_input_1",
        ignore_tokens: Optional[List[str]] = None,
        target: Optional[JsonDict] = None,
    ) -> JsonDict:
        if target is not None:
            raise ValueError("Input reduction does not implement targeted attacks")
        ignore_tokens = ["@@NULL@@"] if ignore_tokens is None else ignore_tokens
        original_instances = self.predictor.json_to_labeled_instances(inputs)
        original_text_field: TextField = original_instances[0][input_field_to_attack]  # type: ignore
        original_tokens = deepcopy(original_text_field.tokens)
        final_tokens = []
        for instance in original_instances:
            final_tokens.append(
                self._attack_instance(
                    inputs, instance, input_field_to_attack, grad_input_field, ignore_tokens
                )
            )
        return sanitize({"final": final_tokens, "original": original_tokens})

    def _attack_instance(
        self,
        inputs: JsonDict,
        instance: Instance,
        input_field_to_attack: str,
        grad_input_field: str,
        ignore_tokens: List[str],
    ) -> List:
        fields_to_compare = utils.get_fields_to_compare(inputs, instance, input_field_to_attack)

        if "tags" not in instance:
            num_ignore_tokens = 1
            tag_mask = None
        else:
            num_ignore_tokens, tag_mask, original_tags = _get_ner_tags_and_mask(
                instance, input_field_to_attack, ignore_tokens
            )

        text_field: TextField = instance[input_field_to_attack]  # type: ignore
        current_tokens = deepcopy(text_field.tokens)
        candidates: List[Tuple[Instance, int, Optional[List[int]]]] = [(instance, -1, tag_mask)]
        while len(current_tokens) > num_ignore_tokens and candidates:
            def get_length(input_instance: Instance) -> int:
                input_text_field: TextField = input_instance[input_field_to_attack]  # type: ignore
                return len(input_text_field.tokens)

            candidates = heapq.nsmallest(self.beam_size, candidates, key=lambda x: get_length(x[0]))

            beam_candidates = deepcopy(candidates)
            candidates = []
            for beam_instance, smallest_idx, tag_mask in beam_candidates:
                beam_tag_mask = deepcopy(tag_mask)
                grads, outputs = self.predictor.get_gradients([beam_instance])

                for output in outputs:
                    if isinstance(outputs[output], torch.Tensor):
                        outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()
                    elif isinstance(outputs[output], list):
                        outputs[output] = outputs[output][0]

                if "tags" not in instance:
                    beam_instance = self.predictor.predictions_to_labeled_instances(
                        beam_instance, outputs
                    )[0]
                    if utils.instance_has_changed(beam_instance, fields_to_compare):
                        continue
                else:
                    if smallest_idx != -1:
                        del beam_tag_mask[smallest_idx]  # type: ignore
                    cur_tags = [
                        outputs["tags"][x] for x in range(len(outputs["tags"])) if beam_tag_mask[x]  # type: ignore
                    ]
                    if cur_tags != original_tags:
                        continue

                text_field: TextField = beam_instance[input_field_to_attack]  # type: ignore
                current_tokens = deepcopy(text_field.tokens)
                reduced_instances_and_smallest = _remove_one_token(
                    beam_instance,
                    input_field_to_attack,
                    grads[grad_input_field][0],
                    ignore_tokens,
                    self.beam_size,
                    beam_tag_mask,  # type: ignore
                )
                candidates.extend(reduced_instances_and_smallest)
        return current_tokens


def _remove_one_token(
    instance: Instance,
    input_field_to_attack: str,
    grads: np.ndarray,
    ignore_tokens: List[str],
    beam_size: int,
    tag_mask: Optional[List[int]],
) -> List[Tuple[Instance, int, Optional[List[int]]]]:
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads]

    text_field: TextField = instance[input_field_to_attack]  # type: ignore
    for token_idx, token in enumerate(text_field.tokens):
        if token in ignore_tokens:
            grads_mag[token_idx] = float("inf")

    if "tags" in instance:
        tag_field: SequenceLabelField = instance["tags"]  # type: ignore
        labels: List[str] = tag_field.labels  # type: ignore
        for idx, label in enumerate(labels):
            if label != "O":
                grads_mag[idx] = float("inf")
    reduced_instances_and_smallest: List[Tuple[Instance, int, Optional[List[int]]]] = []
    for _ in range(beam_size):
        copied_instance = deepcopy(instance)
        copied_text_field: TextField = copied_instance[input_field_to_attack]  # type: ignore

        smallest = np.argmin(grads_mag)
        if grads_mag[smallest] == float("inf"):
            break
        grads_mag[smallest] = float("inf")

        inputs_before_smallest = copied_text_field.tokens[0:smallest]
        inputs_after_smallest = copied_text_field.tokens[smallest + 1 :]
        copied_text_field.tokens = inputs_before_smallest + inputs_after_smallest

        if "tags" in instance:
            tag_field: SequenceLabelField = copied_instance["tags"]  # type: ignore
            tag_field_before_smallest = tag_field.labels[0:smallest]
            tag_field_after_smallest = tag_field.labels[smallest + 1 :]
            tag_field.labels = tag_field_before_smallest + tag_field_after_smallest  # type: ignore
            tag_field.sequence_field = copied_text_field

        copied_instance.indexed = False
        reduced_instances_and_smallest.append((copied_instance, smallest, tag_mask))

    return reduced_instances_and_smallest


def _get_ner_tags_and_mask(
    instance: Instance, input_field_to_attack: str, ignore_tokens: List[str]
) -> Tuple[int, List[int], List[str]]:
    num_ignore_tokens = 0
    input_field: TextField = instance[input_field_to_attack]  # type: ignore
    for token in input_field.tokens:
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1

    tag_mask = []
    original_tags = []
    tag_field: SequenceLabelField = instance["tags"]  # type: ignore
    for label in tag_field.labels:
        if label != "O":
            tag_mask.append(1)
            original_tags.append(label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
