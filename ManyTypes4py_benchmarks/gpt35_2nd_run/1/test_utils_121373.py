import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import sockeye.constants as C
import sockeye.prepare_data
import sockeye.train
import sockeye.translate
import sockeye.lexicon
import sockeye.utils

logger: logging.Logger = logging.getLogger(__name__)
_DIGITS: str = '0123456789'
_MID: int = 5

def generate_digits_file(source_path: str, target_path: str, line_count: int = 100, line_length: int = 9, sort_target: bool = False, line_count_empty: int = 0, source_text_prefix_token: str = '', seed: int = 13) -> None:
    ...

def generate_json_input_file_with_tgt_prefix(src_path: str, tgt_path: str, json_file_with_tgt_prefix_path: str, src_factors_path: Optional[List[str]] = None, tgt_factors_path: Optional[List[str]] = None, seed: int = 13) -> None:
    ...

def generate_low_high_factors(input_path: str, output_path: str) -> None:
    ...

def generate_odd_even_factors(input_path: str, output_path: str) -> None:
    ...

def generate_fast_align_lex(lex_path: str) -> None:
    ...

LEXICON_CREATE_PARAMS_COMMON: str = 'create -i {input} -m {model} -k {topk} -o {lexicon}'

@contextmanager
def tmp_digits_dataset(prefix: str, train_line_count: int, train_line_count_empty: int, train_max_length: int, dev_line_count: int, dev_max_length: int, test_line_count: int, test_line_count_empty: int, test_max_length: int, sort_target: bool = False, seed_train: int = 13, seed_dev: int = 13, source_text_prefix_token: str = '', with_n_source_factors: int = 0, with_n_target_factors: int = 0) -> Dict[str, Any]:
    ...

TRAIN_PARAMS_COMMON: str = '--use-cpu --max-seq-len {max_len} --source {train_source} --target {train_target} --validation-source {dev_source} --validation-target {dev_target} --output {model} --seed {seed}'
PREPARE_DATA_COMMON: str = ' --max-seq-len {max_len} --source {train_source} --target {train_target} --output {output} --pad-vocab-to-multiple-of 16'
TRAIN_WITH_SOURCE_FACTORS_COMMON: str = ' --source-factors {source_factors}'
DEV_WITH_SOURCE_FACTORS_COMMON: str = ' --validation-source-factors {dev_source_factors}'
TRAIN_WITH_TARGET_FACTORS_COMMON: str = ' --target-factors {target_factors}'
DEV_WITH_TARGET_FACTORS_COMMON: str = ' --validation-target-factors {dev_target_factors}'
TRAIN_PARAMS_PREPARED_DATA_COMMON: str = '--use-cpu --max-seq-len {max_len} --prepared-data {prepared_data} --validation-source {dev_source} --validation-target {dev_target} --output {model}'
TRANSLATE_PARAMS_COMMON: str = '--use-cpu --models {model} --input {input} --output {output} --output-type json'
TRANSLATE_WITH_FACTORS_COMMON: str = ' --input-factors {input_factors}'
TRANSLATE_WITH_JSON_FORMAT: str = ' --json-input'
TRANSLATE_PARAMS_RESTRICT: str = '--restrict-lexicon {lexicon} --restrict-lexicon-topk {topk}'
SCORE_PARAMS_COMMON: str = '--use-cpu --model {model} --source {source} --target {target} --output {output} '
SCORE_WITH_SOURCE_FACTORS_COMMON: str = ' --source-factors {source_factors}'
SCORE_WITH_TARGET_FACTORS_COMMON: str = ' --target-factors {target_factors}'

def run_train_translate(train_params: str, translate_params: str, data: Dict[str, Any], use_prepared_data: bool = False, max_seq_len: int = 10, seed: int = 13) -> Dict[str, Any]:
    ...

def run_translate_restrict(data: Dict[str, Any], translate_params: str) -> Dict[str, Any]:
    ...

def collect_translate_output_and_scores(out_path: str) -> List[Dict[str, Any]]:
    ...
