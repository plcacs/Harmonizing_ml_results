"""
A set of utility methods.
"""
import argparse
import binascii
import gzip
import itertools
import logging
import math
import multiprocessing
import os
import pprint
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from itertools import starmap
from typing import Any, List, Iterator, Iterable, Tuple, Dict, Optional, Union, TypeVar, Callable, Sequence, Set, cast
import numpy as np
import torch as pt
import torch.distributed
try:
    import deepspeed
except ImportError:
    pass
from . import __version__, constants as C
from .log import log_sockeye_version, log_torch_version
logger = logging.getLogger(__name__)

class SockeyeError(Exception):
    pass

def check_version(version: str) -> None:
    """
    Checks given version against code version and determines compatibility.
    Throws if versions are incompatible.

    :param version: Given version.
    """
    code_version = parse_version(__version__)
    given_version = parse_version(version)
    if given_version[0] == '3' and given_version[1] == '0':
        logger.info(f'Code version: {__version__}')
        logger.warning(f'Given release version ({version}) does not match code version ({__version__}). Models with version {version} should be compatible though.')
        return
    check_condition(code_version[0] == given_version[0], 'Given release version (%s) does not match release code version (%s)' % (version, __version__))
    check_condition(code_version[1] == given_version[1], 'Given major version (%s) does not match major code version (%s)' % (version, __version__))

def load_version(fname: str) -> str:
    """
    Loads version from file.

    :param fname: Name of file to load version from.
    :return: Version string.
    """
    if not os.path.exists(fname):
        logger.warning('No version file found. Defaulting to 1.0.3')
        return '1.0.3'
    with open(fname) as inp:
        return inp.read().strip()

def parse_version(version_string: str) -> Tuple[str, str, str]:
    """
    Parse version string into release, major, minor version.

    :param version_string: Version string.
    :return: Tuple of strings.
    """
    release, major, minor = version_string.split('.', 2)
    return (release, major, minor)

def log_basic_info(args: argparse.Namespace) -> None:
    """
    Log basic information like version number, arguments, etc.

    :param args: Arguments as returned by argparse.
    """
    log_sockeye_version(logger)
    log_torch_version(logger)
    logger.info('Command: %s', ' '.join(sys.argv))
    logger.info('Arguments: %s', args)

def seed_rngs(seed: int) -> None:
    """
    Seed the random number generators (Python, Numpy and MXNet).

    :param seed: The random seed.
    """
    logger.info(f'Random seed: {seed}')
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        logger.info(f'PyTorch seed: {seed}')
    except ImportError:
        pass

def check_condition(condition: bool, error_message: str) -> None:
    """
    Check the condition and if it is not met, exit with the given error message
    and error_code, similar to assertions.

    :param condition: Condition to check.
    :param error_message: Error message to show to the user.
    """
    if not condition:
        raise SockeyeError(error_message)

class OnlineMeanAndVariance:

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._M2 += delta * delta2

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        if self._count < 2:
            return float('nan')
        else:
            return self._M2 / self._count

    @property
    def std(self) -> float:
        variance = self.variance
        return math.sqrt(variance) if not math.isnan(variance) else 0.0

def chunks(some_list: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(some_list), n):
        yield some_list[i:i + n]

def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token

def is_gzip_file(filename: str) -> bool:
    with open(filename, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'

def smart_open(filename: str, mode: str = 'rt', ftype: str = 'auto', errors: str = 'replace') -> Any:
    """
    Returns a file descriptor for filename with UTF-8 encoding.
    If mode is "rt", file is opened read-only.
    If ftype is "auto", uses gzip iff filename endswith .gz.
    If ftype is {"gzip","gz"}, uses gzip.
    If ftype is "auto" and read mode requested, uses gzip iff is_gzip_file(filename).

    Note: encoding error handling defaults to "replace"

    :param filename: The filename to open.
    :param mode: Reader mode.
    :param ftype: File type. If 'auto' checks filename suffix for gz to try gzip.open.
    :param errors: Encoding error handling during reading. Defaults to 'replace'.
    :return: File descriptor.
    """
    if ftype in ('gzip', 'gz') or (ftype == 'auto' and filename.endswith('.gz')) or (ftype == 'auto' and 'r' in mode and is_gzip_file(filename)):
        if mode == 'rb' or mode == 'wb':
            return gzip.open(filename, mode=mode)
        else:
            return gzip.open(filename, mode=mode, encoding='utf-8', errors=errors)
    elif mode == 'rb' or mode == 'wb':
        return open(filename, mode=mode)
    else:
        return open(filename, mode=mode, encoding='utf-8', errors=errors)

def combine_means(means: List[float], num_sents: List[int]) -> float:
    """
    Takes a list of means and number of sentences of the same length and computes the combined mean.

    :param means: A list of mean values.
    :param num_sents: A list with the number of sentences used to compute each mean value.
    :return: The combined mean of the list of means.
    """
    if not means or not num_sents:
        raise ValueError('Invalid input list.')
    check_condition(len(means) == len(num_sents), 'List lengths do not match')
    return sum((num_sent * mean for num_sent, mean in zip(num_sents, means) if mean is not None)) / sum(num_sents)

def combine_stds(stds: List[float], means: List[float], num_sents: List[int]) -> float:
    """
    Takes a list of standard deviations, means and number of sentences of the same length and computes
    the combined standard deviation.

    :param stds: A list of standard deviations.
    :param means: A list of mean values.
    :param num_sents: A list with number of sentences used to compute each mean value.
    :return: The combined standard deviation.
    """
    if not stds or not means or (not num_sents):
        raise ValueError('Invalid input list.')
    check_condition(all((len(stds) == len(l) for l in [means, num_sents])), 'List lengths do not match')
    total_mean = combine_means(means, num_sents)
    return math.sqrt(sum((num_sent * (std ** 2 + (mean - total_mean) ** 2) for num_sent, std, mean in zip(num_sents, stds, means) if std is not None and mean is not None)) / sum(num_sents))

def average_tensors(tensors: List[pt.Tensor]) -> pt.Tensor:
    """
    Compute the element-wise average of a list of tensors of the same shape.

    :param tensors: A list of input tensors with the same shape.
    :return: The average of the tensors on the same device as tensors[0].
    """
    if not tensors:
        raise ValueError('tensors is empty.')
    if len(tensors) == 1:
        return tensors[0]
    check_condition(all((tensors[0].shape == t.shape for t in tensors)), 'tensor shapes do not match')
    return sum(tensors) / len(tensors)

def gen_prefix_masking(prefix: pt.Tensor, vocab_size: int, dtype: pt.dtype) -> Tuple[pt.Tensor, int]:
    """
    Generate prefix masks from prefix ids, which are inf everywhere except zero for prefix ids.

    :param prefix: Target prefix token or factors in ids. Shape (batch size, max length of prefix).
    :param vocab_size: vocabulary size
    :param dtype: dtype of the retuning output
    :return prefix_masks (batch size, max length of prefix, vocab_size), with type as dtype

    """
    prefix_masks_sizes = list(prefix.size())
    max_length = prefix_masks_sizes[1]
    prefix_masks_sizes.append(vocab_size)
    prefix_masks = pt.full(prefix_masks_sizes, fill_value=np.inf, device=prefix.device, dtype=dtype)
    prefix_masks.scatter_(-1, prefix.to(pt.int64).unsqueeze(-1), 0.0)
    prefix_masks.masked_fill_(prefix.unsqueeze(-1) == 0, 0)
    return (prefix_masks, max_length)

def shift_prefix_factors(prefix_factors: pt.Tensor) -> pt.Tensor:
    """
    Shift prefix factors one step to the right

    :param prefix_factors: tensor ids. Shape (batch size, length, num of factors).
    :return new prefix_factors_shift (batch size, length + 1, num of factors)
    """
    prefix_factors_sizes = prefix_factors.size()
    prefix_factors_shift = pt.zeros(prefix_factors_sizes[0], prefix_factors_sizes[1] + 1, prefix_factors_sizes[2], dtype=prefix_factors.dtype, device=prefix_factors.device)
    prefix_factors_shift[:, 1:] = prefix_factors
    return prefix_factors_shift

def adjust_first_step_masking(target_prefix: pt.Tensor, first_step_mask: pt.Tensor) -> pt.Tensor:
    """
    Adjust first_step_masking based on the target prefix
    (Target prefix for each input in the same batch may have a different length.     Thus first_step_mask needs to be adjusted accordingly.)

    :param target_prefix: Shape (batch size, max target prefix length).
    :param first_step_mask: Shape (batch_size * beam_size, 1)
    :return (adjusted) first_steps_masking (batch_size * beam_size, max target prefix length + 1).

    An illustrative example of how first_step_masking is adjusted

    Inputs:

    target_prefix (batch_size = 2, max target prefix length = 2)

    tensor([1 2]
           [1 0])
    Note: Two target prefix tokens in the first sentence,     one target prefix token in the second sentence.

    first_step_mask (batch_size = 2 * beam_size = 5, 1)

    tensor([[0],
    [inf],
    [inf],
    [inf],
    [inf],
    [0],
    [inf],
    [inf],
    [inf],
    [inf])

    Output:
    Adjusted first_step_mask (batch_size * beam_size, max target prefix length + 1):

    tensor([[0 0 0],
            [inf inf inf],
            [inf inf inf],
            [inf inf inf],
            [inf inf, inf],
            [0 0 0],
            [inf inf 0],
            [inf inf 0],
            [inf inf 0],
            [inf inf 0]])

    The concrete steps of what this function does are as follows:

    Step 1: Create a zero masking matrix with shape (batch size, max target prefix length + 1)
    Fill 1 into this masking matrix based on the target prefix

    target prefix  initialize masking    masking      roll one step to the right
                   from target prefix    is not 0      and assign 1 at index 0
        [1 2]    ->    [1 2 0]        -> [1 1 0]  ->           [1 1 1]
        [1 0]          [1 0 0]           [1 0 0]               [1 1 0]

    Step 2: Adjust first_step_mask based on masking

    masking     expand masking with     expand first_step_mask with max target
                     beam size         prefix length, fill 0 where masking is 0
    [1 1 1]  ->      [1 1 1]        ->             [0 0 0]
    [1 1 0]          [1 1 1]                       [inf inf inf]
                     [1 1 1]                       [inf inf inf]
                     [1 1 1]                       [inf inf inf]
                     [1 1 1]                       [inf inf inf]
                     [1 1 0]                       [0 0 0]
                     [1 1 0]                       [inf inf 0]
                     [1 1 0]                       [inf inf 0]
                     [1 1 0]                       [inf inf 0]
                     [1 1 0]                       [inf inf 0]
    """
    batch_beam, _ = first_step_mask.size()
    batch, max_prefix_len = target_prefix.size()
    beam_size = batch_beam // batch
    masking = pt.zeros((batch, max_prefix_len + 1), device=target_prefix.device)
    masking[:, :max_prefix_len] = target_prefix
    masking = pt.clamp(masking, 0.0, 1.0)
    masking = pt.roll(masking, 1, -1)
    masking[:, 0] = 1.0
    masking = masking.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_beam, -1)
    first_step_mask = first_step_mask.expand(-1, masking.size(-1)).clone()
    first_step_mask.masked_fill_(masking == 0.0, 0.0)
    return first_step_mask

def parse_metrics_line(line_number: int, line: str) -> Dict[str, Any]:
    """
    Parse a line of metrics into a mappings of key and values.

    :param line_number: Line's number for checking if checkpoints are aligned to it.
    :param line: A line from the Sockeye metrics file.
    :return: Dictionary of metric names (e.g. perplexity-train) mapping to a list of values.
    """
    fields = line.split('\t')
    checkpoint = int(fields[0])
    check_condition(line_number == checkpoint, 'Line (%d) and loaded checkpoint (%d) do not align.' % (line_number, checkpoint))
    metric = dict()
    for field in fields[1:]:
        key, value = field.split('=', 1)
        if value == 'True' or value == 'False':
            metric[key] = value == 'True'
        elif value == 'None':
            metric[key] = None
        else:
            metric[key] = float(value)
    return metric

def read_metrics_file(path: str) -> List[Dict[str, Any]]:
    """
    Reads lines metrics file and returns list of mappings of key and values.

    :param path: File to read metric values from.
    :return: Dictionary of metric names (e.g. perplexity-train) mapping to a list of values.
    """
    with open(path) as fin:
        metrics = [parse_metrics_line(i, line.strip()) for i, line in enumerate(fin, 1)]
    return metrics

def write_metrics_file(metrics: List[Dict[str, Any]], path: str) -> None:
    """
    Write metrics data to tab-separated file.

    :param metrics: metrics data.
    :param path: Path to write to.
    """
    with open(path, 'w') as metrics_out:
        for checkpoint, metric_dict in enumerate(metrics, 1):
            metrics_str = '\t'.join(['{}={}'.format(name, value) for name, value in sorted(metric_dict.items())])
            metrics_out.write('{}\t{}\n'.format(checkpoint, metrics_str))

def get_validation_metric_points(model_path: str, metric: str) -> List[Tuple[float, int]]:
    """
    Returns tuples of value and checkpoint for given metric from metrics file at model_path.
    :param model_path: Model path containing .metrics file.
    :param metric: Metric values to extract.
    :return: List of tuples (value, checkpoint).
    """
    metrics_path = os.path.join(model_path, C.METRICS_NAME)
    data = read_metrics_file(metrics_path)
    return [(d['%s-val' % metric], cp) for cp, d in enumerate(data, 1)]

def grouper(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """
    Collect data into fixed-length chunks or blocks without discarding underfilled chunks or padding them.

    :param iterable: A sequence of inputs.
    :param size