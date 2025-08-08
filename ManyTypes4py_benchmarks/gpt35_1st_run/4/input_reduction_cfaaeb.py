from copy import deepcopy
from typing import List, Tuple
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
    def __init__(self, predictor: Predictor, beam_size: int = 3) -> None:
    def attack_from_json(self, inputs: JsonDict, input_field_to_attack: str = 'tokens', grad_input_field: str = 'grad_input_1', ignore_tokens: List[str] = None, target: None = None) -> JsonDict:
    def _attack_instance(self, inputs: JsonDict, instance: Instance, input_field_to_attack: str, grad_input_field: str, ignore_tokens: List[str]) -> List[str]:
    def _remove_one_token(instance: Instance, input_field_to_attack: str, grads: np.ndarray, ignore_tokens: List[str], beam_size: int, tag_mask: List[int]) -> List[Tuple[Instance, int, List[int]]]:
    def _get_ner_tags_and_mask(instance: Instance, input_field_to_attack: str, ignore_tokens: List[str]) -> Tuple[int, List[int], List[str]]:
