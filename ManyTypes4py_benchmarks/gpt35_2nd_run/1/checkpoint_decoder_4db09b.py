import logging
import os
import random
import time
from contextlib import ExitStack
from itertools import chain
from typing import Any, Dict, Optional, List
import torch
from . import constants as C
from . import evaluate
from . import inference
from . import model
from . import utils
from . import vocab
logger: logging.Logger = logging.getLogger(__name__)

class CheckpointDecoder:
    def __init__(self, model_folder: str, inputs: List[str], references: str, source_vocabs: List[vocab.Vocab], target_vocabs: List[vocab.Vocab], model: model.Transformer, device: torch.device, max_input_len: Optional[int] = None, batch_size: int = 16, beam_size: int = C.DEFAULT_BEAM_SIZE, nbest_size: int = C.DEFAULT_NBEST_SIZE, bucket_width_source: int = 10, length_penalty_alpha: float = 1.0, length_penalty_beta: float = 0.0, max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH, ensemble_mode: str = 'linear', sample_size: int = -1, random_seed: int = 42) -> None:
    
    def decode_and_evaluate(self, output_name: Optional[str] = None) -> Dict[str, Any]:
    
    def warmup(self) -> None:

def parallel_subsample(parallel_sequences: List[List[str]], sample_size: int, seed: int) -> List[List[str]]:

def write_to_file(data: List[str], fname: str) -> None:
