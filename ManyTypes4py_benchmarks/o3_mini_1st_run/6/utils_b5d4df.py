#!/usr/bin/env python3
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
from typing import Any, List, Iterator, Iterable, Tuple, Dict, Optional, Union, TypeVar, Callable, ContextManager

import numpy as np
import torch as pt
import torch.distributed
try:
    import deepspeed  # type: ignore
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
    """
    code_version: Tuple[str, str, str] = parse_version(__version__)
    given_version: Tuple[str, str, str] = parse_version(version)
    if given_version[0] == '3' and given_version[1] == '0':
        logger.info(f'Code version: {__version__}')
        logger.warning(f'Given release version ({version}) does not match code version ({__version__}). Models with version {version} should be compatible though.')
        return
    check_condition(code_version[0] == given_version[0], 'Given release version (%s) does not match release code version (%s)' % (version, __version__))
    check_condition(code_version[1] == given_version[1], 'Given major version (%s) does not match major code version (%s)' % (version, __version__))

def load_version(fname: str) -> str:
    """
    Loads version from file.
    """
    if not os.path.exists(fname):
        logger.warning('No version file found. Defaulting to 1.0.3')
        return '1.0.3'
    with open(fname) as inp:
        return inp.read().strip()

def parse_version(version_string: str) -> Tuple[str, str, str]:
    """
    Parse version string into release, major, minor version.
    """
    release, major, minor = version_string.split('.', 2)
    return (release, major, minor)

def log_basic_info(args: Any) -> None:
    """
    Log basic information like version number, arguments, etc.
    """
    log_sockeye_version(logger)
    log_torch_version(logger)
    logger.info('Command: %s', ' '.join(sys.argv))
    logger.info('Arguments: %s', args)

def seed_rngs(seed: int) -> None:
    """
    Seed the random number generators (Python, Numpy and MXNet).
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
    Check the condition and if it is not met, exit with the given error message.
    """
    if not condition:
        raise SockeyeError(error_message)

class OnlineMeanAndVariance:
    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def update(self, value: float) -> None:
        self._count += 1
        delta: float = value - self._mean
        self._mean += delta / self._count
        delta2: float = value - self._mean
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
        var: float = self.variance
        return math.sqrt(var) if not math.isnan(var) else 0.0

T = TypeVar('T')

def chunks(some_list: List[T], n: int) -> Iterator[List[T]]:
    """Yield successive n-sized chunks from some_list."""
    for i in range(0, len(some_list), n):
        yield some_list[i:i + n]

def get_tokens(line: str) -> Iterator[str]:
    """
    Yields tokens from input string.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token

def is_gzip_file(filename: str) -> bool:
    with open(filename, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'

from io import IOBase  # For type annotation of file objects
from typing import BinaryIO, TextIO

def smart_open(filename: str, mode: str = 'rt', ftype: str = 'auto', errors: str = 'replace') -> Union[TextIO, BinaryIO]:
    """
    Returns a file descriptor for filename with UTF-8 encoding.
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

def combine_means(means: List[Optional[float]], num_sents: List[int]) -> float:
    """
    Takes a list of means and number of sentences of the same length and computes the combined mean.
    """
    if not means or not num_sents:
        raise ValueError('Invalid input list.')
    check_condition(len(means) == len(num_sents), 'List lengths do not match')
    return sum((num_sent * mean for num_sent, mean in zip(num_sents, means) if mean is not None)) / sum(num_sents)

def combine_stds(stds: List[Optional[float]], means: List[Optional[float]], num_sents: List[int]) -> float:
    """
    Takes lists of standard deviations, means, and number of sentences to compute combined standard deviation.
    """
    if not stds or not means or not num_sents:
        raise ValueError('Invalid input list.')
    check_condition(all((len(stds) == len(l) for l in [means, num_sents])), 'List lengths do not match')
    total_mean: float = combine_means(means, num_sents)
    return math.sqrt(sum((num_sent * (std ** 2 + (mean - total_mean) ** 2) for num_sent, std, mean in zip(num_sents, stds, means) if std is not None and mean is not None)) / sum(num_sents))

def average_tensors(tensors: List[pt.Tensor]) -> pt.Tensor:
    """
    Compute the element-wise average of a list of tensors of the same shape.
    """
    if not tensors:
        raise ValueError('tensors is empty.')
    if len(tensors) == 1:
        return tensors[0]
    check_condition(all((tensors[0].shape == t.shape for t in tensors)), 'tensor shapes do not match')
    return sum(tensors) / len(tensors)

def gen_prefix_masking(prefix: pt.Tensor, vocab_size: int, dtype: pt.dtype) -> Tuple[pt.Tensor, int]:
    """
    Generate prefix masks from prefix ids.
    """
    prefix_masks_sizes: List[int] = list(prefix.size())
    max_length: int = prefix_masks_sizes[1]
    prefix_masks_sizes.append(vocab_size)
    prefix_masks: pt.Tensor = pt.full(prefix_masks_sizes, fill_value=np.inf, device=prefix.device, dtype=dtype)
    prefix_masks.scatter_(-1, prefix.to(pt.int64).unsqueeze(-1), 0.0)
    prefix_masks.masked_fill_(prefix.unsqueeze(-1) == 0, 0)
    return (prefix_masks, max_length)

def shift_prefix_factors(prefix_factors: pt.Tensor) -> pt.Tensor:
    """
    Shift prefix factors one step to the right.
    """
    prefix_factors_sizes = prefix_factors.size()
    prefix_factors_shift: pt.Tensor = pt.zeros(prefix_factors_sizes[0], prefix_factors_sizes[1] + 1, prefix_factors_sizes[2],
                                                dtype=prefix_factors.dtype, device=prefix_factors.device)
    prefix_factors_shift[:, 1:] = prefix_factors
    return prefix_factors_shift

def adjust_first_step_masking(target_prefix: pt.Tensor, first_step_mask: pt.Tensor) -> pt.Tensor:
    """
    Adjust first_step_masking based on the target prefix.
    """
    batch_beam, _ = first_step_mask.size()
    batch, max_prefix_len = target_prefix.size()
    beam_size: int = batch_beam // batch
    masking: pt.Tensor = pt.zeros((batch, max_prefix_len + 1), device=target_prefix.device)
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
    Parse a line of metrics into a mapping of key and values.
    """
    fields: List[str] = line.split('\t')
    checkpoint: int = int(fields[0])
    check_condition(line_number == checkpoint, 'Line (%d) and loaded checkpoint (%d) do not align.' % (line_number, checkpoint))
    metric: Dict[str, Any] = {}
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
    Reads lines from a metrics file and returns a list of mappings of key and values.
    """
    with open(path) as fin:
        metrics: List[Dict[str, Any]] = [parse_metrics_line(i, line.strip()) for i, line in enumerate(fin, 1)]
    return metrics

def write_metrics_file(metrics: List[Dict[str, Any]], path: str) -> None:
    """
    Write metrics data to a tab-separated file.
    """
    with open(path, 'w') as metrics_out:
        for checkpoint, metric_dict in enumerate(metrics, 1):
            metrics_str = '\t'.join(['{}={}'.format(name, value) for name, value in sorted(metric_dict.items())])
            metrics_out.write('{}\t{}\n'.format(checkpoint, metrics_str))

def get_validation_metric_points(model_path: str, metric: str) -> List[Tuple[float, int]]:
    """
    Returns tuples of value and checkpoint for given metric from metrics file at model_path.
    """
    metrics_path: str = os.path.join(model_path, C.METRICS_NAME)
    data: List[Dict[str, Any]] = read_metrics_file(metrics_path)
    return [(d[f'{metric}-val'], cp) for cp, d in enumerate(data, 1)]

def grouper(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """
    Collect data into fixed-length chunks without discarding underfilled chunks.
    """
    it = iter(iterable)
    while True:
        chunk: List[T] = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk

def metric_value_is_better(new: float, old: float, metric: str) -> bool:
    """
    Returns true if new value is strictly better than old for a given metric.
    """
    if C.METRIC_MAXIMIZE[metric]:
        return new > old
    else:
        return new < old

_dtype_to_string: Dict[Any, str] = {np.float16: C.DTYPE_FP16, np.float32: C.DTYPE_FP32, np.int8: C.DTYPE_INT8,
                                      np.int32: C.DTYPE_INT32, pt.bfloat16: C.DTYPE_BF16, pt.float16: C.DTYPE_FP16,
                                      pt.float32: C.DTYPE_FP32, pt.int8: C.DTYPE_INT8, pt.int32: C.DTYPE_INT32}

def dtype_to_str(dtype: Any) -> str:
    return _dtype_to_string.get(dtype, str(dtype))

_STRING_TO_TORCH_DTYPE: Dict[str, pt.dtype] = {C.DTYPE_BF16: pt.bfloat16, C.DTYPE_FP16: pt.float16,
                                                 C.DTYPE_FP32: pt.float32, C.DTYPE_INT8: pt.int8,
                                                 C.DTYPE_INT32: pt.int32}

def get_torch_dtype(dtype: Union[pt.dtype, str]) -> pt.dtype:
    if isinstance(dtype, pt.dtype):
        return dtype
    if dtype in _STRING_TO_TORCH_DTYPE:
        return _STRING_TO_TORCH_DTYPE[dtype]
    raise ValueError(f'Cannot convert to Torch dtype: {dtype}')

_STRING_TO_NUMPY_DTYPE: Dict[str, Any] = {C.DTYPE_FP16: np.float16, C.DTYPE_FP32: np.float32,
                                           C.DTYPE_INT8: np.int8, C.DTYPE_INT16: np.int16, C.DTYPE_INT32: np.int32}

def get_numpy_dtype(dtype: Union[np.dtype, str]) -> np.dtype:
    if isinstance(dtype, np.dtype):
        return dtype
    if dtype in _STRING_TO_NUMPY_DTYPE:
        return _STRING_TO_NUMPY_DTYPE[dtype]
    raise ValueError(f'Cannot convert to NumPy dtype: {dtype}')

def log_parameters(model: pt.nn.Module) -> None:
    """
    Logs information about model parameters.
    """
    fixed_parameter_names: List[str] = []
    learned_parameter_names: List[str] = []
    total_learned: int = 0
    total_fixed: int = 0
    visited: Dict[pt.Tensor, List[str]] = defaultdict(list)
    for name, module in model.named_modules(remove_duplicate=False):
        for param_name, param in module.named_parameters(prefix=name, recurse=False):
            repr_str: str = '%s [%s, %s]' % (name, tuple(param.shape), dtype_to_str(param.dtype))
            size: int = param.shape.numel()
            if not param.requires_grad:
                fixed_parameter_names.append(repr_str)
                total_fixed += size if param not in visited else 0
            else:
                total_learned += size if param not in visited else 0
                learned_parameter_names.append(repr_str)
            visited[param].append(param_name)
    shared_parameter_names: List[str] = []
    total_shared: int = 0
    for param, names in visited.items():
        if len(names) > 1:
            total_shared += param.shape.numel()
            shared_parameter_names.append(' = '.join(names))
    total_parameters: int = total_learned + total_fixed
    logger.info('# of parameters: %d | trainable: %d (%.2f%%) | shared: %d (%.2f%%) | fixed: %d (%.2f%%)',
                total_parameters, total_learned, total_learned / total_parameters * 100,
                total_shared, total_shared / total_parameters * 100, total_fixed, total_fixed / total_parameters * 100)
    logger.info('Trainable parameters: \n%s', pprint.pformat(learned_parameter_names))
    logger.info('Shared parameters: \n%s', pprint.pformat(shared_parameter_names, width=120))
    logger.info('Fixed parameters:\n%s', pprint.pformat(fixed_parameter_names))

@contextmanager
def no_context() -> Iterator[None]:
    """
    No-op context manager that can be used in "with" statements.
    """
    yield None

class SingleProcessPool:
    def map(self, func: Callable[[Any], T], iterable: Iterable[Any]) -> List[T]:
        return list(map(func, iterable))

    def starmap(self, func: Callable[..., T], iterable: Iterable[Any]) -> List[T]:
        return list(starmap(func, iterable))

    def __enter__(self) -> "SingleProcessPool":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

def create_pool(max_processes: int) -> Union[SingleProcessPool, multiprocessing.pool.Pool]:
    if max_processes == 1:
        return SingleProcessPool()
    else:
        return multiprocessing.pool.Pool(processes=max_processes)

def update_dict(dest: Dict[Any, Any], source: Dict[Any, Any]) -> None:
    for key in source:
        if isinstance(source[key], dict):
            if key not in dest:
                dest[key] = {}
            if not isinstance(dest[key], dict):
                raise ValueError(f'Type mismatch for key {key}: {type(source[key])} vs {type(dest[key])}')
            update_dict(dest[key], source[key])
        else:
            dest[key] = source[key]

def update_dict_with_prefix_kv(dest: Dict[Any, Any], prefix_kv: Dict[str, Any]) -> None:
    """
    Update a dictionary in place with prefix key-value dictionary.
    """
    for keys, value in prefix_kv.items():
        split_keys: List[str] = keys.split('.')
        prefix: List[str] = split_keys[:-1]
        key: str = split_keys[-1]
        _dict: Dict[Any, Any] = dest
        for prefix_field in prefix:
            if prefix_field not in _dict:
                _dict[prefix_field] = {}
            _dict = _dict[prefix_field]
        _dict[key] = value

def is_distributed() -> bool:
    return pt.distributed.is_initialized()

def is_primary_worker() -> bool:
    """
    True when current process is the primary worker (rank 0) or the only worker.
    """
    return not pt.distributed.is_initialized() or pt.distributed.get_rank() == 0

def get_local_rank() -> int:
    return int(os.environ[C.DIST_ENV_LOCAL_RANK])

U = TypeVar('U')

def broadcast_object(obj: U, src: int = 0) -> U:
    """
    Broadcast a single Python object across workers.
    """
    obj_list: List[U] = [obj]
    pt.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

def all_gather_object(obj: U) -> List[U]:
    """Gather each worker's instance of an object, returned as a list."""
    obj_list: List[Optional[U]] = [None] * pt.distributed.get_world_size()
    pt.distributed.all_gather_object(obj_list, obj)
    # The following cast is safe, assuming all_gather_object fills in all values.
    return obj_list  # type: ignore

_using_deepspeed: bool = False

def init_deepspeed() -> None:
    """
    Initialize DeepSpeed and set the global flag.
    """
    global _using_deepspeed
    try:
        import deepspeed  # type: ignore
        import deepspeed.utils.zero_to_fp32  # type: ignore
        deepspeed.init_distributed()
        _using_deepspeed = True
    except:
        raise RuntimeError('To train models with DeepSpeed (https://www.deepspeed.ai/), install the module with `pip install deepspeed`.')

def using_deepspeed() -> bool:
    """Check whether DeepSpeed has been initialized via this module."""
    return _using_deepspeed

_faiss_checked: bool = False

def check_import_faiss() -> None:
    """
    Make sure the faiss module can be imported.
    """
    global _faiss_checked
    if not _faiss_checked:
        try:
            import faiss  # type: ignore
            _faiss_checked = True
        except:
            raise RuntimeError('To run kNN-MT models, please install faiss by following https://github.com/facebookresearch/faiss/blob/main/INSTALL.md')

def count_seq_len(sample: str, count_type: str = C.SEQ_LEN_IN_CHARACTERS, replace_tokens: Optional[Iterable[str]] = None) -> int:
    """
    Count sequence length, after optionally replacing tokens.
    """
    if replace_tokens is not None:
        for tokens in replace_tokens:
            sample = sample.replace(tokens, '')
    if count_type == C.SEQ_LEN_IN_CHARACTERS:
        return len(sample.replace(C.TOKEN_SEPARATOR, ''))
    elif count_type == C.SEQ_LEN_IN_TOKENS:
        return len(sample.split(C.TOKEN_SEPARATOR))
    else:
        raise SockeyeError("Sequence length count type '%s' unknown. Choices are: %s" % (count_type, [C.SEQ_LEN_IN_CHARACTERS, C.SEQ_LEN_IN_TOKENS]))

def compute_isometric_score(hypothesis: str, hypothesis_score: float, source: str, isometric_metric: str = C.RERANK_ISOMETRIC_RATIO, isometric_alpha: float = 0.5) -> float:
    """
    Compute hypothesis to source isometric score using sample char length.
    """
    count_type: str = C.SEQ_LEN_IN_CHARACTERS
    replace_tokens: Iterable[str] = C.TOKEN_SEGMENTATION_MARKERS
    hypothesis_len: int = count_seq_len(hypothesis, count_type, replace_tokens)
    source_len: int = count_seq_len(source, count_type, replace_tokens)
    if isometric_metric == C.RERANK_ISOMETRIC_LC:
        abs_len_diff: int = abs(hypothesis_len - source_len)
        isometric_score: float = abs_len_diff * 100 / source_len if source_len else abs_len_diff * 100
        return isometric_score
    else:
        if isometric_metric == C.RERANK_ISOMETRIC_RATIO:
            len_ratio: float = hypothesis_len / source_len if source_len else float(hypothesis_len)
            synchrony_score: float = 1 / (1 + len_ratio)
        if isometric_metric == C.RERANK_ISOMETRIC_DIFF:
            abs_len_diff = abs(hypothesis_len - source_len)
            synchrony_score = 1 / (1 + abs_len_diff)
        pred_sub_score: float = (1 - isometric_alpha) * hypothesis_score
        synchrony_sub_score: float = isometric_alpha * synchrony_score
        isometric_score = pred_sub_score + synchrony_sub_score
        return isometric_score

def init_device(args: Any) -> pt.device:
    """
    Select Torch device based on CLI args.
    """
    use_cpu: bool = args.use_cpu
    if not use_cpu and (not pt.cuda.is_available()):
        logger.info('CUDA not available, defaulting to CPU device')
        use_cpu = True
    if use_cpu:
        return pt.device('cpu')
    device: pt.device = pt.device('cuda', get_local_rank() if is_distributed() else args.device_id)
    pt.cuda.set_device(device)
    if args.tf32:
        pt.backends.cuda.matmul.allow_tf32 = True
        logger.info('CUDA: allow tf32 (float32 but with 10 bits precision)')
    return device

def fault_tolerant_symlink(src: str, dst: str, max_retries: int = 6) -> None:
    """
    Attempt to create a symbolic link from source to destination. Retry on failure.
    """
    retries: int = 0
    while True:
        try:
            os.symlink(src, dst)
            return
        except FileExistsError as error:
            if retries >= max_retries:
                break
            wait_time: int = 2 ** retries
            logger.warn(f'Error detected when calling symlink: {error}. Retrying in {wait_time} seconds.')
            time.sleep(wait_time)
            retries += 1
    raise OSError(f"Max retries exceeded when attempting to create symlink: '{src}' -> '{dst}'")