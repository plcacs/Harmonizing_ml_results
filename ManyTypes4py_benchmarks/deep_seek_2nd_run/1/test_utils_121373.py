import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Iterator, Tuple, Union, Sequence, Generator
from unittest.mock import patch
import sockeye.constants as C
import sockeye.prepare_data
import sockeye.train
import sockeye.translate
import sockeye.lexicon
import sockeye.utils

logger = logging.getLogger(__name__)
_DIGITS: str = '0123456789'
_MID: int = 5

def generate_digits_file(source_path: str, target_path: str, line_count: int = 100, line_length: int = 9, sort_target: bool = False, line_count_empty: int = 0, source_text_prefix_token: str = '', seed: int = 13) -> None:
    source_text_prefix: str = ''
    if source_text_prefix_token:
        line_length -= 1
        source_text_prefix = f'{source_text_prefix_token}{C.TOKEN_SEPARATOR}'
    assert line_count_empty <= line_count
    random_gen: random.Random = random.Random(seed)
    with open(source_path, 'w') as source_out, open(target_path, 'w') as target_out:
        all_digits: List[List[str]] = []
        for _ in range(line_count - line_count_empty):
            digits: List[str] = [random_gen.choice(_DIGITS) for _ in range(random_gen.randint(1, line_length))]
            all_digits.append(digits)
        for _ in range(line_count_empty):
            all_digits.append([])
        random_gen.shuffle(all_digits)
        for digits in all_digits:
            print(f'{source_text_prefix}{C.TOKEN_SEPARATOR.join(digits)}' if digits else '', file=source_out)
            if sort_target:
                digits.sort()
            print(C.TOKEN_SEPARATOR.join(digits), file=target_out)

def generate_json_input_file_with_tgt_prefix(src_path: str, tgt_path: str, json_file_with_tgt_prefix_path: str, src_factors_path: Optional[List[str]] = None, tgt_factors_path: Optional[List[str]] = None, seed: int = 13) -> None:
    random_gen: random.Random = random.Random(seed)
    with open(src_path, 'r') as src_reader, open(tgt_path, 'r') as tgt_reader:
        with open(json_file_with_tgt_prefix_path, 'w') as out:
            list_src_factors: Optional[List[List[str]]] = None
            list_tgt_factors: Optional[List[List[List[str]]]] = None
            if src_factors_path is not None:
                list_src_factors = [open(src_factors, 'r') for src_factors in src_factors_path]
                list_src_factors = [[sf.strip() for sf in src_factors] for src_factors in list_src_factors]
            if tgt_factors_path is not None:
                list_tgt_factors = [open(tgt_factors, 'r') for tgt_factors in tgt_factors_path]
                list_tgt_factors = [[tf.strip().split() for tf in tgt_factors] for tgt_factors in list_tgt_factors]
            for i, stdigits in enumerate(zip(src_reader, tgt_reader)):
                src_digits: str
                tgt_digits: str
                src_digits, tgt_digits = (stdigits[0].strip(), stdigits[1].strip())
                tgt_prefix: List[str] = tgt_digits.split()
                if len(tgt_digits) > 0:
                    random_pos: int = random_gen.choice([pos for pos in range(len(tgt_prefix))])
                    tgt_prefix = tgt_prefix[:random_pos]
                if tgt_factors_path is not None and len(list_tgt_factors[0][i]) > 0:
                    random_pos = random_gen.choice([pos for pos in range(len(list_tgt_factors[0][i]))])
                    for k in range(len(list_tgt_factors)):
                        list_tgt_factors[k][i] = list_tgt_factors[k][i][:random_pos]
                tgt_prefix_str: str = C.TOKEN_SEPARATOR.join(tgt_prefix)
                jsone_line: Dict[str, Any]
                if src_factors_path is None and tgt_factors_path is None:
                    jsone_line = {'text': src_digits, 'target_prefix': tgt_prefix_str}
                elif src_factors_path is not None and tgt_factors_path is None:
                    jsone_line = {'text': src_digits, 'factors': [src_factors[i] for src_factors in list_src_factors], 'target_prefix': tgt_prefix_str}
                elif tgt_factors_path is not None and src_factors_path is None:
                    jsone_line = {'text': src_digits, 'target_prefix_factors': [C.TOKEN_SEPARATOR.join(tgt_factors[i]) for tgt_factors in list_tgt_factors], 'target_prefix': tgt_prefix_str}
                else:
                    jsone_line = {'text': src_digits, 'factors': [src_factors[i] for src_factors in list_src_factors], 'target_prefix_factors': [C.TOKEN_SEPARATOR.join(tgt_factors[i]) for tgt_factors in list_tgt_factors], 'target_prefix': tgt_prefix_str}
                print(json.dumps(jsone_line), file=out)

def generate_low_high_factors(input_path: str, output_path: str) -> None:
    """
    Writes low/high factor file given a file of digit sequences.
    """
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            digits: Iterator[int] = map(int, line.rstrip().split())
            factors: Generator[str, None, None] = ('l' if digit < _MID else 'h' for digit in digits)
            print(C.TOKEN_SEPARATOR.join(factors), file=fout)

def generate_odd_even_factors(input_path: str, output_path: str) -> None:
    """
    Writes odd/even factor file given a file of digit sequences.
    """
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            digits: Iterator[int] = map(int, line.rstrip().split())
            factors: Generator[str, None, None] = ('e' if digit % 2 == 0 else 'o' for digit in digits)
            print(C.TOKEN_SEPARATOR.join(factors), file=fout)

def generate_fast_align_lex(lex_path: str) -> None:
    """
    Generate a fast_align format lex table for digits.

    :param lex_path: Path to write lex table.
    """
    with open(lex_path, 'w') as lex_out:
        for digit in _DIGITS:
            print('{0}\t{0}\t0'.format(digit), file=lex_out)

LEXICON_CREATE_PARAMS_COMMON: str = 'create -i {input} -m {model} -k {topk} -o {lexicon}'

@contextmanager
def tmp_digits_dataset(prefix: str, train_line_count: int, train_line_count_empty: int, train_max_length: int, dev_line_count: int, dev_max_length: int, test_line_count: int, test_line_count_empty: int, test_max_length: int, sort_target: bool = False, seed_train: int = 13, seed_dev: int = 13, source_text_prefix_token: str = '', with_n_source_factors: int = 0, with_n_target_factors: int = 0) -> Generator[Dict[str, Any], None, None]:
    """
    Creates a temporary dataset with train, dev, and test. Returns a dictionary with paths to the respective temporary
    files.
    """
    if source_text_prefix_token:
        sockeye.utils.check_condition(with_n_source_factors == 0, 'The digits dataset does not support source factors and source text prefix token at the same time.')
    with TemporaryDirectory(prefix=prefix) as work_dir:
        train_source_path: str = os.path.join(work_dir, 'train.src')
        train_target_path: str = os.path.join(work_dir, 'train.tgt')
        dev_source_path: str = os.path.join(work_dir, 'dev.src')
        dev_target_path: str = os.path.join(work_dir, 'dev.tgt')
        test_source_path: str = os.path.join(work_dir, 'test.src')
        test_target_path: str = os.path.join(work_dir, 'test.tgt')
        test_source_with_target_prefix_path: str = os.path.join(work_dir, 'test_source_with_target_prefix.json')
        generate_digits_file(train_source_path, train_target_path, train_line_count, train_max_length, line_count_empty=train_line_count_empty, sort_target=sort_target, seed=seed_train, source_text_prefix_token=source_text_prefix_token)
        generate_digits_file(dev_source_path, dev_target_path, dev_line_count, dev_max_length, sort_target=sort_target, seed=seed_dev, source_text_prefix_token=source_text_prefix_token)
        generate_digits_file(test_source_path, test_target_path, test_line_count, test_max_length, line_count_empty=test_line_count_empty, sort_target=sort_target, seed=seed_dev, source_text_prefix_token=source_text_prefix_token)
        data: Dict[str, Any] = {'work_dir': work_dir, 'train_source': train_source_path, 'train_target': train_target_path, 'dev_source': dev_source_path, 'dev_target': dev_target_path, 'test_source': test_source_path, 'test_target': test_target_path, 'test_source_with_target_prefix': test_source_with_target_prefix_path}
        if with_n_source_factors > 0:
            data['train_source_factors'] = []
            data['dev_source_factors'] = []
            data['test_source_factors'] = []
            for i in range(with_n_source_factors):
                train_factor_path: str = train_source_path + '.factors%d' % i
                dev_factor_path: str = dev_source_path + '.factors%d' % i
                test_factor_path: str = test_source_path + '.factors%d' % i
                generate_low_high_factors(train_source_path, train_factor_path)
                generate_low_high_factors(dev_source_path, dev_factor_path)
                generate_low_high_factors(test_source_path, test_factor_path)
                data['train_source_factors'].append(train_factor_path)
                data['dev_source_factors'].append(dev_factor_path)
                data['test_source_factors'].append(test_factor_path)
        if with_n_target_factors > 0:
            data['train_target_factors'] = []
            data['dev_target_factors'] = []
            data['test_target_factors'] = []
            for i in range(with_n_target_factors):
                train_factor_path = train_target_path + '.factors%d' % i
                dev_factor_path = dev_target_path + '.factors%d' % i
                test_factor_path = test_target_path + '.factors%d' % i
                generate_odd_even_factors(train_target_path, train_factor_path)
                generate_odd_even_factors(dev_target_path, dev_factor_path)
                generate_odd_even_factors(test_target_path, test_factor_path)
                data['train_target_factors'].append(train_factor_path)
                data['dev_target_factors'].append(dev_factor_path)
                data['test_target_factors'].append(test_factor_path)
        source_factors_path: Optional[List[str]] = None if 'test_source_factors' not in data else data['test_source_factors']
        target_factors_path: Optional[List[str]] = None if 'test_target_factors' not in data else data['test_target_factors']
        generate_json_input_file_with_tgt_prefix(test_source_path, test_target_path, test_source_with_target_prefix_path, source_factors_path, target_factors_path)
        yield data

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
    """
    Train a model and translate a test set. Returns the updated data dictionary containing paths to translation outputs
    and scores.

    :param train_params: Command line args for model training.
    :param translate_params: First command line args for translation.
    :param data: Dictionary containing test data
    :param use_prepared_data: Whether to use the prepared data functionality.
    :param max_seq_len: The maximum sequence length.
    :param seed: The seed used for training.
    :return: Data dictionary, updated with translation outputs and scores
    """
    work_dir: str = os.path.join(data['work_dir'], 'train_translate')
    data['model'] = os.path.join(work_dir, 'model')
    if use_prepared_data:
        data['train_prepared'] = os.path.join(work_dir, 'prepared_data')
        prepare_params: str = '{} {}'.format(sockeye.prepare_data.__file__, PREPARE_DATA_COMMON.format(train_source=data['train_source'], train_target=data['train_target'], output=data['train_prepared'], max_len=max_seq_len))
        if 'train_source_factors' in data:
            prepare_params += TRAIN_WITH_SOURCE_FACTORS_COMMON.format(source_factors=' '.join(data['train_source_factors']))
        if 'train_target_factors' in data:
            prepare_params += TRAIN_WITH_TARGET_FACTORS_COMMON.format(target_factors=' '.join(data['train_target_factors']))
        if '--weight-tying-type src_trg' in train_params:
            prepare_params += ' --shared-vocab'
        logger.info('Preparing data with parameters %s.', prepare_params)
        with patch.object(sys, 'argv', prepare_params.split()):
            sockeye.prepare_data.main()
        params: str = '{} {} {}'.format(sockeye.train.__file__, TRAIN_PARAMS_PREPARED_DATA_COMMON.format(prepared_data=data['train_prepared'], dev_source=data['dev_source'], dev_target=data['dev_target'], model=data['model'], max_len=max_seq_len), train_params)
        if 'dev_source_factors' in data:
            params += DEV_WITH_SOURCE_FACTORS_COMMON.format(dev_source_factors=' '.join(data['dev_source_factors']))
        if 'dev_target_factors' in data:
            params += DEV_WITH_TARGET_FACTORS_COMMON.format(dev_target_factors=' '.join(data['dev_target_factors']))
        logger.info('Starting training with parameters %s.', train_params)
        with patch.object(sys, 'argv', params.split()):
            sockeye.train.main()
    else:
        params = '{} {} {}'.format(sockeye.train.__file__, TRAIN_PARAMS_COMMON.format(train_source=data['train_source'], train_target=data['train_target'], dev_source=data['dev_source'], dev_target=data['dev_target'], model=data['model'], max_len=max_seq_len, seed=seed), train_params)
        if 'train_source_factors' in data:
            params += TRAIN_WITH_SOURCE_FACTORS_COMMON.format(source_factors=' '.join(data['train_source_factors']))
        if 'train_target_factors' in data:
            params += TRAIN_WITH_TARGET_FACTORS_COMMON.format(target_factors=' '.join(data['train_target_factors']))
        if 'dev_source_factors' in data:
            params += DEV_WITH_SOURCE_FACTORS_COMMON.format(dev_source_factors=' '.join(data['dev_source_factors']))
        if 'dev_target_factors' in data:
            params += DEV_WITH_TARGET_FACTORS_COMMON.format(dev_target_factors=' '.join(data['dev_target_factors']))
        logger.info('Starting training with parameters %s.', train_params)
        with patch.object(sys, 'argv', params.split()):
            sockeye.train.main()
    ttable_path: str = os.path.join(data['work_dir'], 'ttable')
    generate_fast_align_lex(ttable_path)
    lexicon_path: str = os.path.join(data['work_dir'], 'lexicon')
    params = '{} {}'.format(sockeye.lexicon.__file__,