from copy import deepcopy
from typing import Dict, List, Tuple
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor
DEFAULT_IGNORE_TOKENS: List[str] = ['@@NULL@@', '.', ',', ';', '!', '?', '[MASK]', '[SEP]', '[CLS]']

@Attacker.register('hotflip')
class Hotflip(Attacker):
    def __init__(self, predictor: Predictor, vocab_namespace: str = 'tokens', max_tokens: int = 5000) -> None:
    def initialize(self) -> None:
    def _construct_embedding_matrix(self) -> torch.Tensor:
    def _make_embedder_input(self, all_tokens: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    def attack_from_json(self, inputs: JsonDict, input_field_to_attack: str = 'tokens', grad_input_field: str = 'grad_input_1', ignore_tokens: List[str] = None, target: JsonDict = None) -> JsonDict:
    def attack_instance(self, instance: Instance, inputs: JsonDict, input_field_to_attack: str = 'tokens', grad_input_field: str = 'grad_input_1', ignore_tokens: List[str] = None, target: JsonDict = None) -> Tuple[List[Token], Dict[str, numpy.ndarray]]:
    def _first_order_taylor(self, grad: numpy.ndarray, token_idx: torch.Tensor, sign: int) -> int:
