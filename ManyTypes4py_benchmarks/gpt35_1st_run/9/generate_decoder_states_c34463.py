import argparse
import logging
import os
from typing import List, Optional
import numpy as np
import torch as pt
from . import arguments
from . import constants as C
from . import data_io
from . import utils
from .log import setup_main_logger
from .model import SockeyeModel, load_model
from .vocab import Vocab
from .utils import check_condition
from .knn import KNNConfig, get_state_store_path, get_word_store_path, get_config_path
logger: logging.Logger = logging.getLogger(__name__)

class NumpyMemmapStorage:
    def __init__(self, file_name: str, num_dim: int, dtype: np.dtype):
        self.file_name: str = file_name
        self.num_dim: int = num_dim
        self.dtype: np.dtype = dtype
        self.block_size: int = -1
        self.mmap: Optional[np.memmap] = None
        self.tail_idx: int = 0
        self.size: int = 0

    def open(self, initial_size: int, block_size: int) -> None:
        ...

    def add(self, array: np.ndarray) -> None:
        ...

class DecoderStateGenerator:
    def __init__(self, model: SockeyeModel, source_vocabs: List[Vocab], target_vocabs: List[Vocab], output_dir: str, max_seq_len_source: int, max_seq_len_target: int, state_data_type: np.dtype, word_data_type: np.dtype, device: str):
        ...

    @staticmethod
    def probe_token_count(target_path: str, max_seq_len: int) -> int:
        ...

    def init_store_file(self, initial_size: int) -> None:
        ...

    def generate_states_and_store(self, sources: List[str], targets: List[str], batch_size: int, eop_id: int) -> None:
        ...

    def save_config(self) -> None:
        ...

def store(args: argparse.Namespace) -> None:
    ...

def main() -> None:
    ...
