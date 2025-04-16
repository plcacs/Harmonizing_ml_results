from copy import deepcopy
from typing import List, Tuple, Optional, Any
import heapq
import numpy as np
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.predictors import Predictor


@Attacker.register('input-reduction')
class InputReduction(Attacker):
    """
    Runs the input reduction method from [Pathologies of Neural Models Make Interpretations
    Difficult](https://arxiv.org/abs/1804.07781), which removes as many words as possible from
    the input without changing the model's prediction.

    The functions on this class handle a special case for NER by looking for a field called "tags"
    This check is brittle, i.e., the code could break if the name of this field has changed, or if
    a non-NER model has a field called "tags".

    Registered as an `Attacker` with name "input-reduction".
    """

    def __init__(self, predictor, beam_size=3):
        super().__init__(predictor)
        self.beam_size: int = beam_size

    def attack_from_json(self, inputs, input_field_to_attack='tokens',
        grad_input_field='grad_input_1', ignore_tokens=None, target=None):
        if target is not None:
            raise ValueError(
                'Input reduction does not implement targeted attacks')
        ignore_tokens = ['@@NULL@@'
            ] if ignore_tokens is None else ignore_tokens
        original_instances: List[Instance
            ] = self.predictor.json_to_labeled_instances(inputs)
        original_text_field: TextField = original_instances[0][
            input_field_to_attack]
        original_tokens: List[Any] = deepcopy(original_text_field.tokens)
        final_tokens: List[List[Any]] = []
        for instance in original_instances:
            final_tokens.append(self._attack_instance(inputs, instance,
                input_field_to_attack, grad_input_field, ignore_tokens))
        return sanitize({'final': final_tokens, 'original': original_tokens})

    def _attack_instance(self, inputs, instance, input_field_to_attack,
        grad_input_field, ignore_tokens):
        fields_to_compare: Any = utils.get_fields_to_compare(inputs,
            instance, input_field_to_attack)
        if 'tags' not in instance:
            num_ignore_tokens: int = 1
            tag_mask: Optional[List[int]] = None
        else:
            num_ignore_tokens, tag_mask, original_tags = (
                _get_ner_tags_and_mask(instance, input_field_to_attack,
                ignore_tokens))
        text_field: TextField = instance[input_field_to_attack]
        current_tokens: List[Any] = deepcopy(text_field.tokens)
        candidates: List[Tuple[Instance, int, Optional[List[int]]]] = [(
            instance, -1, tag_mask)]
        while len(current_tokens) > num_ignore_tokens and candidates:

            def get_length(input_instance):
                input_text_field: TextField = input_instance[
                    input_field_to_attack]
                return len(input_text_field.tokens)
            candidates = heapq.nsmallest(self.beam_size, candidates, key=lambda
                x: get_length(x[0]))
            beam_candidates: List[Tuple[Instance, int, Optional[List[int]]]
                ] = deepcopy(candidates)
            candidates = []
            for beam_instance, smallest_idx, beam_tag_mask in beam_candidates:
                beam_tag_mask_copy: Optional[List[int]] = deepcopy(
                    beam_tag_mask)
                grads, outputs = self.predictor.get_gradients([beam_instance])
                for output_key in outputs:
                    if isinstance(outputs[output_key], torch.Tensor):
                        outputs[output_key] = outputs[output_key].detach().cpu(
                            ).numpy().squeeze().squeeze()
                    elif isinstance(outputs[output_key], list):
                        outputs[output_key] = outputs[output_key][0]
                if 'tags' not in instance:
                    beam_instance = (self.predictor.
                        predictions_to_labeled_instances(beam_instance,
                        outputs)[0])
                    if utils.instance_has_changed(beam_instance,
                        fields_to_compare):
                        continue
                else:
                    if smallest_idx != -1:
                        assert beam_tag_mask_copy is not None
                        del beam_tag_mask_copy[smallest_idx]
                    cur_tags: List[str] = [outputs['tags'][x] for x in
                        range(len(outputs['tags'])) if beam_tag_mask_copy[x]]
                    if cur_tags != original_tags:
                        continue
                text_field_copy: TextField = beam_instance[
                    input_field_to_attack]
                current_tokens = deepcopy(text_field_copy.tokens)
                reduced_instances_and_smallest: List[Tuple[Instance, int,
                    Optional[List[int]]]] = _remove_one_token(beam_instance,
                    input_field_to_attack, grads[grad_input_field][0],
                    ignore_tokens, self.beam_size, beam_tag_mask_copy)
                candidates.extend(reduced_instances_and_smallest)
        return current_tokens


def _remove_one_token(instance, input_field_to_attack, grads, ignore_tokens,
    beam_size, tag_mask):
    """
    Finds the token with the smallest gradient and removes it.
    """
    grads_mag: List[float] = [np.sqrt(grad.dot(grad)) for grad in grads]
    text_field: TextField = instance[input_field_to_attack]
    for token_idx, token in enumerate(text_field.tokens):
        if token in ignore_tokens:
            grads_mag[token_idx] = float('inf')
    if 'tags' in instance:
        tag_field: SequenceLabelField = instance['tags']
        labels: List[str] = tag_field.labels
        for idx, label in enumerate(labels):
            if label != 'O':
                grads_mag[idx] = float('inf')
    reduced_instances_and_smallest: List[Tuple[Instance, int, Optional[List
        [int]]]] = []
    for _ in range(beam_size):
        copied_instance: Instance = deepcopy(instance)
        copied_text_field: TextField = copied_instance[input_field_to_attack]
        smallest: int = int(np.argmin(grads_mag))
        if grads_mag[smallest] == float('inf'):
            break
        grads_mag[smallest] = float('inf')
        inputs_before_smallest: List[Any] = copied_text_field.tokens[0:smallest
            ]
        inputs_after_smallest: List[Any] = copied_text_field.tokens[
            smallest + 1:]
        copied_text_field.tokens = (inputs_before_smallest +
            inputs_after_smallest)
        if 'tags' in instance:
            tag_field_copy: SequenceLabelField = copied_instance['tags']
            tag_field_before_smallest: List[str] = tag_field_copy.labels[0:
                smallest]
            tag_field_after_smallest: List[str] = tag_field_copy.labels[
                smallest + 1:]
            tag_field_copy.labels = (tag_field_before_smallest +
                tag_field_after_smallest)
            tag_field_copy.sequence_field = copied_text_field
        copied_instance.indexed = False
        reduced_instances_and_smallest.append((copied_instance, smallest,
            tag_mask))
    return reduced_instances_and_smallest


def _get_ner_tags_and_mask(instance, input_field_to_attack, ignore_tokens):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original predicted tag and a 0/1
    mask in the position of the tags
    """
    num_ignore_tokens: int = 0
    input_field: TextField = instance[input_field_to_attack]
    for token in input_field.tokens:
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1
    tag_mask: List[int] = []
    original_tags: List[str] = []
    tag_field: SequenceLabelField = instance['tags']
    for label in tag_field.labels:
        if label != 'O':
            tag_mask.append(1)
            original_tags.append(label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
