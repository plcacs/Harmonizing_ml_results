#!/usr/bin/env python3
"""
Defines commandline arguments for the main CLIs with reasonable defaults.
"""
import argparse
import os
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, IO
import yaml
from sockeye.utils import smart_open
from . import constants as C


class ConfigArgumentParser(argparse.ArgumentParser):
    """
    Extension of argparse.ArgumentParser supporting config files.
    The option --config is added automatically and expects a YAML serialized
    dictionary, similar to the return value of parse_args(). Command line
    parameters have precedence over config file values. Usage should be
    transparent, just substitute argparse.ArgumentParser with this class.
    Extended from
    https://stackoverflow.com/questions/28579661/getting-required-option-from-namespace-in-python
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Keys are a tuple of argument names, values are the keyword arguments.
        self.argument_definitions: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.argument_actions: List[argparse.Action] = []
        self._overwrite_add_argument(self)
        self.add_argument(
            '--config',
            help=("Path to CLI arguments in yaml format (as saved in Sockeye model directories as 'args.yaml'). "
                  "Commandline arguments have precedence over values in this file."),
            type=str,
        )

    def _register_argument(self, _action: argparse.Action, *args: Any, **kwargs: Any) -> None:
        self.argument_definitions[args] = kwargs
        self.argument_actions.append(_action)

    def _overwrite_add_argument(self, original_object: Any) -> Any:
        def _new_add_argument(this_self: Any, *args: Any, **kwargs: Any) -> argparse.Action:
            action: argparse.Action = this_self.original_add_argument(*args, **kwargs)
            this_self.config_container._register_argument(action, *args, **kwargs)
            return action

        original_object.original_add_argument = original_object.add_argument
        original_object.config_container = self
        original_object.add_argument = types.MethodType(_new_add_argument, original_object)
        return original_object

    def add_argument_group(self, *args: Any, **kwargs: Any) -> argparse._ArgumentGroup:
        group: argparse._ArgumentGroup = super().add_argument_group(*args, **kwargs)
        return self._overwrite_add_argument(group)

    def parse_args(self, args: Optional[List[str]] = None, namespace: Optional[argparse.Namespace] = None) -> argparse.Namespace:
        config_parser: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument('--config', type=regular_file())
        config_args, _ = config_parser.parse_known_args(args=args)
        initial_args: argparse.Namespace = argparse.Namespace()
        if config_args.config:
            initial_args = load_args(config_args.config)
            for action in self.argument_actions:
                if action.dest in initial_args.__dict__:
                    action.required = False
        return super().parse_args(args=args, namespace=initial_args)


class StoreDeprecatedAction(argparse.Action):
    def __init__(self, option_strings: List[str], dest: str, deprecated_dest: str, nargs: Optional[Any] = None, **kwargs: Any) -> None:
        super(StoreDeprecatedAction, self).__init__(option_strings, dest, **kwargs)
        self.deprecated_dest: str = deprecated_dest

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, value: Any, option_string: Optional[str] = None) -> None:
        setattr(namespace, self.dest, value)
        setattr(namespace, self.deprecated_dest, value)


def save_args(args: argparse.Namespace, fname: str) -> None:
    with open(fname, 'w') as out:
        yaml.safe_dump(args.__dict__, out, default_flow_style=False)


def load_args(fname: str) -> argparse.Namespace:
    with open(fname, 'r') as inp:
        return argparse.Namespace(**yaml.safe_load(inp))


def regular_file() -> Callable[[Union[str, bytes]], str]:
    """
    Returns a method that can be used in argument parsing to check the argument is a regular file or a symbolic link,
    but not, e.g., a process substitution.
    :return: A method that can be used as a type in argparse.
    """
    def check_regular_file(value_to_check: Union[str, bytes]) -> str:
        path: str = str(value_to_check)
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError('must exist and be a regular file.')
        return path
    return check_regular_file


def regular_folder() -> Callable[[Union[str, bytes]], str]:
    """
    Returns a method that can be used in argument parsing to check the argument is a directory.
    :return: A method that can be used as a type in argparse.
    """
    def check_regular_directory(value_to_check: Union[str, bytes]) -> str:
        path: str = str(value_to_check)
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError('must be a directory.')
        return path
    return check_regular_directory


def int_greater_or_equal(threshold: int) -> Callable[[Any], int]:
    """
    Returns a method that can be used in argument parsing to check that the int argument is greater or equal to `threshold`.
    :param threshold: The threshold that we assume the cli argument value is greater or equal to.
    :return: A method that can be used as a type in argparse.
    """
    def check_greater_equal(value: Any) -> int:
        value_to_check: int = int(value)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError('must be greater or equal to %d.' % threshold)
        return value_to_check
    return check_greater_equal


def float_greater_or_equal(threshold: float) -> Callable[[Any], float]:
    """
    Returns a method that can be used in argument parsing to check that the float argument is greater or equal to `threshold`.
    :param threshold: The threshold that we assume the cli argument value is greater or equal to.
    :return: A method that can be used as a type in argparse.
    """
    def check_greater_equal(value: Any) -> float:
        value_to_check: float = float(value)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError('must be greater or equal to %f.' % threshold)
        return value_to_check
    return check_greater_equal


def bool_str() -> Callable[[str], bool]:
    """
    Returns a method that can be used in argument parsing to check that the argument is a valid representation of
    a boolean value.
    :return: A method that can be used as a type in argparse.
    """
    def parse(value: str) -> bool:
        lower_value: str = value.lower()
        if lower_value in ['true', 'yes', '1']:
            return True
        elif lower_value in ['false', 'no', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Invalid value for bool argument. Use true/false, yes/no or 1/0.')
    return parse


def simple_dict() -> Callable[[str], Dict[str, Any]]:
    """
    A simple dictionary format that does not require spaces or quoting.
    Format: key1:value1,key2:value2,...
    Supported types: bool, int, float, str (that doesn't parse as other types).
    :return: A method that can be used as a type in argparse.
    """
    def parse(dict_str: str) -> Dict[str, Any]:
        def _parse(value: str) -> Union[bool, int, float, str]:
            lower_val: str = value.lower()
            if lower_val == 'true':
                return True
            if lower_val == 'false':
                return False
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
            try:
                return float(value)
            except Exception:
                return value
        _dict: Dict[str, Any] = {}
        try:
            for entry in dict_str.split(','):
                key, value = entry.split(':')
                _dict[key] = _parse(value)
        except ValueError:
            raise argparse.ArgumentTypeError('Specify argument dictionary as key1:value1,key2:value2,...')
        return _dict
    return parse


def multiple_values(num_values: int = 0, greater_or_equal: Optional[float] = None, data_type: Callable[[str], Any] = int) -> Callable[[str], Tuple[Any, ...]]:
    """
    Returns a method to be used in argument parsing to parse a string of the form "<val>:<val>[:<val>...]" into
    a tuple of values of type data_type.
    :param num_values: Optional number of values required.
    :param greater_or_equal: Optional constraint that all values should be greater or equal to this value.
    :param data_type: Type of values. Default: int.
    :return: Method for parsing.
    """
    def parse(value_to_check: str) -> Tuple[Any, ...]:
        if ':' in value_to_check:
            expected_num_separators: int = num_values - 1 if num_values else 0
            if expected_num_separators > 0 and value_to_check.count(':') != expected_num_separators:
                raise argparse.ArgumentTypeError('Expected either a single value or %d values separated by %s' %
                                                 (num_values, C.ARG_SEPARATOR))
            values: Tuple[Any, ...] = tuple(map(data_type, value_to_check.split(C.ARG_SEPARATOR, num_values - 1)))
        else:
            values = tuple([data_type(value_to_check)] * num_values)
        if greater_or_equal is not None:
            if any((value < greater_or_equal for value in values)):
                raise argparse.ArgumentTypeError('Must provide value greater or equal to %d' % greater_or_equal)
        return values
    return parse


def file_or_stdin() -> Callable[[Optional[str]], IO[Any]]:
    """
    Returns a file descriptor from stdin or opening a file from a given path.
    """
    def parse(path: Optional[str]) -> IO[Any]:
        if path is None or path == '-':
            return sys.stdin
        else:
            return smart_open(path)
    return parse


def add_average_args(params: argparse.ArgumentParser) -> None:
    average_params = params.add_argument_group('Averaging')
    average_params.add_argument('inputs', metavar='INPUT', type=str, nargs='+',
                                help=('either a single model directory (automatic checkpoint selection) or multiple '
                                      '.params files (manual checkpoint selection)'))
    average_params.add_argument('--metric', help='Name of the metric to choose n-best checkpoints from. Default: %(default)s.',
                                default=C.PERPLEXITY, choices=C.METRICS)
    average_params.add_argument('-n', type=int, default=4, help='number of checkpoints to find. Default: %(default)s.')
    average_params.add_argument('--output', '-o', required=True, type=str,
                                help='File to write averaged parameters to.')
    average_params.add_argument('--strategy', choices=C.AVERAGE_CHOICES, default=C.AVERAGE_BEST,
                                help='selection method. Default: %(default)s.')


def add_rerank_args(params: argparse.ArgumentParser) -> None:
    rerank_params = params.add_argument_group('Reranking')
    rerank_params.add_argument('--reference', '-r', type=str, required=True,
                               help='File where target reference translations are stored.')
    rerank_params.add_argument('--hypotheses', '-hy', type=str, required=True,
                               help=('File with nbest translations, one nbest list per line, in JSON format as returned by '
                                     'sockeye.translate with --nbest-size x.'))
    rerank_params.add_argument('--metric', '-m', type=str, required=False, default=C.RERANK_BLEU,
                               choices=C.RERANK_METRICS,
                               help='Sentence-level metric used to compare each nbest translation to the reference or the source.'
                                    'Default: %(default)s.')
    rerank_params.add_argument('--isometric-alpha', required=False, type=float_greater_or_equal(0.0), default=0.5,
                               help='Alpha factor used for reranking (--isometric-[ratio/diff]) nbest list. Requires optimization on dev set.'
                                    'Default: %(default)s.')
    rerank_params.add_argument('--output', '-o', default=None, help='File to write output to. Default: STDOUT.')
    rerank_params.add_argument('--output-best', action='store_true', help='Output only the best hypothesis from each nbest list.')
    rerank_params.add_argument('--output-best-non-blank', action='store_true',
                               help=('When outputting only the best hypothesis (--output-best) and the best hypothesis is a blank line, '
                                     'output following non-blank best from the nbest list.'))
    rerank_params.add_argument('--output-reference-instead-of-blank', action='store_true',
                               help=('When outputting only the best hypothesis (--output-best) and the best hypothesis is a blank line, '
                                     'output the reference instead.'))
    rerank_params.add_argument('--return-score', action='store_true',
                               help='Returns the reranking scores as scores in output JSON objects.')


def add_lexicon_args(params: argparse.ArgumentParser, is_for_block_lexicon: bool = False) -> None:
    lexicon_params = params.add_argument_group('Model & Top-k')
    lexicon_params.add_argument('--model', '-m', required=True,
                                help='Model directory containing source and target vocabularies.')
    if not is_for_block_lexicon:
        lexicon_params.add_argument('-k', type=int, default=200,
                                    help='Number of target translations to keep per source. Default: %(default)s.')


def add_lexicon_create_args(params: argparse.ArgumentParser, is_for_block_lexicon: bool = False) -> None:
    lexicon_params = params.add_argument_group('I/O')
    if is_for_block_lexicon:
        input_help: str = 'A text file with tokens that shall be blocked. All token must be in the model vocabulary.'
    else:
        input_help = 'Probabilistic lexicon (fast_align format) to build top-k lexicon from.'
    lexicon_params.add_argument('--input', '-i', required=True, help=input_help)
    lexicon_params.add_argument('--output', '-o', required=True, help='File name to write top-k lexicon to.')


def add_lexicon_inspect_args(params: argparse.ArgumentParser) -> None:
    lexicon_params = params.add_argument_group('Lexicon to inspect')
    lexicon_params.add_argument('--lexicon', '-l', required=True, help='File name of top-k lexicon to inspect.')


def add_logging_args(params: argparse.ArgumentParser) -> None:
    logging_params = params.add_argument_group('Logging')
    logging_params.add_argument('--quiet', '-q', default=False, action='store_true', help='Suppress console logging.')
    logging_params.add_argument('--quiet-secondary-workers', '-qsw', default=False, action='store_true',
                                  help='Suppress console logging for secondary workers in distributed training.')
    logging_params.add_argument('--no-logfile', default=False, action='store_true', help='Suppress file logging')
    log_levels: List[str] = ['INFO', 'DEBUG', 'ERROR']
    logging_params.add_argument('--loglevel', '--log-level', default='INFO', choices=log_levels,
                                  help='Log level. Default: %(default)s.')
    logging_params.add_argument('--loglevel-secondary-workers', default='INFO', choices=log_levels,
                                  help='Console log level for secondary workers. Default: %(default)s.')


def add_quantize_args(params: argparse.ArgumentParser) -> None:
    quant_params = params.add_argument_group('Quantization')
    quant_params.add_argument('--model', '-m', required=True,
                              help=(f'Model (directory) to quantize in place. "{C.PARAMS_BEST_NAME}" will be replaced with a quantized '
                                    f'version and "{C.CONFIG_NAME}" will be updated with the new dtype. The original files will be backed up with '
                                    f'suffixes indicating the starting dtype (e.g., "{C.PARAMS_BEST_NAME}.{C.DTYPE_FP32}" and '
                                    f'"{C.CONFIG_NAME}.{C.DTYPE_FP32}").'))
    quant_params.add_argument('--dtype', default=C.DTYPE_FP16, choices=[C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_BF16],
                              help='Target data type for quantization. Default: %(default)s.')


def add_training_data_args(params: argparse.ArgumentParser, required: bool = False) -> None:
    params.add_argument(C.TRAINING_ARG_SOURCE, '-s', required=required, type=regular_file(),
                        help='Source side of parallel training data.')
    params.add_argument('--source-factors', '-sf', required=False, nargs='+', type=regular_file(), default=[],
                        help='File(s) containing additional token-parallel source-side factors. Default: %(default)s.')
    params.add_argument('--source-factors-use-source-vocab', required=False, nargs='+', type=bool_str(),
                        default=[], help='List of bools signaling whether to use the source vocabulary for the source factors. '
                                          'If empty (default) each factor has its own vocabulary.')
    params.add_argument('--target-factors', '-tf', required=False, nargs='+', type=regular_file(), default=[],
                        help='File(s) containing additional token-parallel target-side factors. Default: %(default)s.')
    params.add_argument('--target-factors-use-target-vocab', required=False, nargs='+', type=bool_str(),
                        default=[], help='List of bools signaling whether to use the target vocabulary for the target factors. '
                                          'If empty (default) each factor has its own vocabulary.')
    params.add_argument(C.TRAINING_ARG_TARGET, '-t', required=required, type=regular_file(),
                        help='Target side of parallel training data.')
    params.add_argument('--end-of-prepending-tag', type=str, default=None,
                        help='Tag indicating the end of prepended text. Prepended tokens before this tag (inclusive) will be marked, and they will not be counted toward source length when calculating maximum output length for beam search.')


def add_validation_data_params(params: argparse.ArgumentParser) -> None:
    params.add_argument('--validation-source', '-vs', required=True, type=regular_file(),
                        help='Source side of validation data.')
    params.add_argument('--validation-source-factors', '-vsf', required=False, nargs='+', type=regular_file(), default=[],
                        help='File(s) containing additional token-parallel validation source side factors. Default: %(default)s.')
    params.add_argument('--validation-target', '-vt', required=True, type=regular_file(),
                        help='Target side of validation data.')
    params.add_argument('--validation-target-factors', '-vtf', required=False, nargs='+', type=regular_file(), default=[],
                        help='File(s) containing additional token-parallel validation target side factors. Default: %(default)s.')


def add_prepared_data_args(params: argparse.ArgumentParser) -> None:
    params.add_argument(C.TRAINING_ARG_PREPARED_DATA, '-d', type=regular_folder(),
                        help='Prepared training data directory created through python -m sockeye.prepare_data.')


def add_training_output_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--output', '-o', required=True,
                        help='Folder where model & training results are written to.')
    params.add_argument('--overwrite-output', action='store_true',
                        help='Delete all contents of the model directory if it already exists.')


def add_training_io_args(params: argparse.ArgumentParser) -> None:
    io_params = params.add_argument_group('Data & I/O')
    add_training_data_args(io_params, required=False)
    add_prepared_data_args(io_params)
    add_validation_data_params(io_params)
    add_bucketing_args(io_params)
    add_vocab_args(io_params)


def add_bucketing_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--no-bucketing', action='store_true',
                        help='Disable bucketing: always unroll the graph to --max-seq-len. Default: %(default)s.')
    params.add_argument('--bucket-width', type=int_greater_or_equal(1), default=8,
                        help='Width of buckets in tokens. Default: %(default)s.')
    params.add_argument('--bucket-scaling', action='store_true',
                        help='Scale source/target buckets based on length ratio to reduce padding. Default: %(default)s.')
    params.add_argument(C.TRAINING_ARG_MAX_SEQ_LEN, type=multiple_values(num_values=2, greater_or_equal=1),
                        default=(95, 95), help='Maximum sequence length in tokens, not counting BOS/EOS tokens (internal max sequence length is X+1). '
                                               'Use "x:x" to specify separate values for src&tgt. Default: %(default)s.')


def add_process_pool_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--max-processes', type=int_greater_or_equal(1), default=1,
                        help='Process the shards in parallel using max-processes processes.')


def add_prepare_data_cli_args(params: argparse.ArgumentParser) -> None:
    add_training_data_args(params, required=True)
    add_vocab_args(params)
    add_bucketing_args(params)
    params.add_argument('--num-samples-per-shard', type=int_greater_or_equal(1), default=10000000,
                        help='The approximate number of samples per shard. Default: %(default)s.')
    params.add_argument('--min-num-shards', default=1, type=int_greater_or_equal(1),
                        help='The minimum number of shards to use, even if they would not reach the desired number of samples per shard. Default: %(default)s.')
    params.add_argument('--seed', type=int, default=13, help='Random seed used that makes shard assignments deterministic. Default: %(default)s.')
    params.add_argument('--output', '-o', required=True,
                        help='Folder where the prepared and possibly sharded data is written to.')
    add_logging_args(params)
    add_process_pool_args(params)


def add_device_args(params: argparse.ArgumentParser) -> None:
    device_params = params.add_argument_group('Device parameters')
    device_params.add_argument('--device-id', type=int_greater_or_equal(0), default=0,
                               help='GPU to use. 0 translates to "cuda:0", etc. When running in distributed mode (--dist), each process\'s device is set automatically. Default: %(default)s.')
    device_params.add_argument('--use-cpu', action='store_true', help='Use CPU device instead of GPU.')
    device_params.add_argument('--env', help='List of environment variables to be set before importing PyTorch. Separated by ",", e.g. --env=OMP_NUM_THREADS=1,PYTORCH_JIT=0 etc.')
    device_params.add_argument('--tf32', type=bool_str(), default=True,
                               help='Globally enable transparent tf32 acceleration of float32 at the cost of reducing precision to 10 bits. Default: %(default)s.')


def add_vocab_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--source-vocab', required=False, default=None,
                        help='Existing source vocabulary (JSON).')
    params.add_argument('--target-vocab', required=False, default=None,
                        help='Existing target vocabulary (JSON).')
    params.add_argument('--source-factor-vocabs', required=False, nargs='+', type=regular_file(), default=[],
                        help='Existing source factor vocabulary (-ies) (JSON).')
    params.add_argument('--target-factor-vocabs', required=False, nargs='+', type=regular_file(), default=[],
                        help='Existing target factor vocabulary (-ies) (JSON).')
    params.add_argument(C.VOCAB_ARG_SHARED_VOCAB, action='store_true', default=False,
                        help='Share source and target vocabulary. Will be automatically turned on when using weight tying. Default: %(default)s.')
    params.add_argument('--num-words', type=multiple_values(num_values=2, greater_or_equal=0), default=(0, 0),
                        help='Maximum vocabulary size. Use "x:x" to specify separate values for src&tgt. A value of 0 indicates that the vocabulary is unrestricted and determined from the data by creating an entry for all words that occur at least --word-min-count times. Default: %(default)s.')
    params.add_argument('--word-min-count', type=multiple_values(num_values=2, greater_or_equal=1), default=(1, 1),
                        help='Minimum frequency of words to be included in vocabularies. Default: %(default)s.')
    params.add_argument('--pad-vocab-to-multiple-of', type=int, default=8,
                        help='Pad vocabulary to a multiple of this integer. Default: %(default)s.')


def add_model_parameters(params: argparse.ArgumentParser) -> None:
    model_params = params.add_argument_group('ModelConfig')
    model_params.add_argument('--params', '-p', type=str, default=None,
                              help='Initialize model parameters from file. Overrides random initializations.')
    model_params.add_argument('--allow-missing-params', action='store_true', default=False,
                              help='Allow missing parameters when initializing model parameters from file. Default: %(default)s.')
    model_params.add_argument('--ignore-extra-params', action='store_true', default=False,
                              help='Allow extra parameters when initializing model parameters from file. Default: %(default)s.')
    model_params.add_argument('--encoder', choices=C.ENCODERS, default=C.TRANSFORMER_TYPE,
                              help='Type of encoder. Default: %(default)s.')
    model_params.add_argument('--decoder', choices=C.DECODERS, default=C.TRANSFORMER_TYPE,
                              help="Type of decoder. Default: %(default)s. 'ssru_transformer' uses Simpler Simple Recurrent Units (Kim et al, 2019) as replacement for self-attention layers.")
    model_params.add_argument('--num-layers', type=multiple_values(num_values=2, greater_or_equal=1), default=(6, 6),
                              help='Number of layers for encoder & decoder. Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    model_params.add_argument('--transformer-model-size', type=multiple_values(num_values=2, greater_or_equal=1), default=(512, 512),
                              help='Number of hidden units in transformer layers. Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    model_params.add_argument('--transformer-attention-heads', type=multiple_values(num_values=2, greater_or_equal=1), default=(8, 8),
                              help='Number of heads for all self-attention when using transformer layers. Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    model_params.add_argument('--transformer-feed-forward-num-hidden', type=multiple_values(num_values=2, greater_or_equal=1), default=(2048, 2048),
                              help='Number of hidden units in transformers feed forward layers. Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    model_params.add_argument('--transformer-feed-forward-use-glu', action='store_true', default=False,
                              help='Use Gated Linear Units in transformer feed forward networks (Daupin et al. 2016, arxiv.org/abs/1612.08083; Shazeer 2020, arxiv.org/abs/2002.05202). Default: %(default)s.')
    model_params.add_argument('--transformer-activation-type', type=multiple_values(num_values=2, greater_or_equal=None, data_type=str), default=(C.RELU, C.RELU),
                              help='Type of activation to use for each feed forward layer. Use "x:x" to specify different values for encoder & decoder. Supported: {}. Default: %(default)s.'.format(' '.join(C.TRANSFORMER_ACTIVATION_TYPES)))
    model_params.add_argument('--transformer-positional-embedding-type', choices=C.POSITIONAL_EMBEDDING_TYPES, default=C.FIXED_POSITIONAL_EMBEDDING,
                              help='The type of positional embedding. Default: %(default)s.')
    model_params.add_argument('--transformer-block-prepended-cross-attention', action='store_true', default=False,
                              help='Block cross-attention between decoder and encoded prepended tokens. Default: %(default)s.')
    model_params.add_argument('--transformer-preprocess', type=multiple_values(num_values=2, greater_or_equal=None, data_type=str), default=('n', 'n'),
                              help='Transformer preprocess sequence for encoder and decoder. Supports three types of operations: d=dropout, r=residual connection, n=layer normalization. You can combine in any order, for example: "ndr". Leave empty to not use any of these operations. You can specify separate sequences for encoder and decoder by separating with ":" For example: n:drn Default: %(default)s.')
    model_params.add_argument('--transformer-postprocess', type=multiple_values(num_values=2, greater_or_equal=None, data_type=str), default=('dr', 'dr'),
                              help='Transformer postprocess sequence for encoder and decoder. Supports three types of operations: d=dropout, r=residual connection, n=layer normalization. You can combine in any order, for example: "ndr". Leave empty to not use any of these operations. You can specify separate sequences for encoder and decoder by separating with ":" For example: n:drn Default: %(default)s.')
    model_params.add_argument('--lhuc', nargs='+', default=None, choices=C.LHUC_CHOICES, metavar='COMPONENT',
                              help='Use LHUC (Vilar 2018). Include an amplitude parameter to hidden units for domain adaptation. Needs a pre-trained model. Valid values: {values}. Default: %(default)s.'.format(values=', '.join(C.LHUC_CHOICES)))
    model_params.add_argument('--num-embed', type=multiple_values(num_values=2, greater_or_equal=1), default=(None, None),
                              help='Embedding size for source and target tokens. Use "x:x" to specify separate values for src&tgt. Default: %d.' % C.DEFAULT_NUM_EMBED)
    model_params.add_argument('--source-factors-num-embed', type=int, nargs='+', default=[],
                              help='Embedding size for additional source factors. You must provide as many dimensions as (validation) source factor files. Default: %(default)s.')
    model_params.add_argument('--target-factors-num-embed', type=int, nargs='+', default=[],
                              help='Embedding size for additional target factors. You must provide as many dimensions as (validation) target factor files. Default: %(default)s.')
    model_params.add_argument('--source-factors-combine', '-sfc', choices=C.FACTORS_COMBINE_CHOICES, default=[C.FACTORS_COMBINE_SUM], nargs='+',
                              help='How to combine source factors. Can be either one value which will be applied to all source factors, or a list of values. Default: %(default)s.')
    model_params.add_argument('--target-factors-combine', '-tfc', choices=C.FACTORS_COMBINE_CHOICES, default=[C.FACTORS_COMBINE_SUM], nargs='+',
                              help='How to combine target factors. Can be either one value which will be applied to all target factors, or a list of values. Default: %(default)s.')
    model_params.add_argument('--source-factors-share-embedding', type=bool_str(), nargs='+', default=[False],
                              help='Share the embeddings with the source language. Can be either one value which will be applied to all source factors, or a list of values. Default: %(default)s.')
    model_params.add_argument('--target-factors-share-embedding', type=bool_str(), nargs='+', default=[False],
                              help='Share the embeddings with the target language. Can be either one value which will be applied to all target factors, or a list of values. Default: %(default)s.')
    model_params.add_argument('--weight-tying-type', default=C.WEIGHT_TYING_SRC_TRG_SOFTMAX, choices=C.WEIGHT_TYING_TYPES,
                              help='The type of weight tying. source embeddings=src, target embeddings=trg, target softmax weight matrix=softmax. Default: %(default)s.')
    model_params.add_argument('--dtype', default=C.DTYPE_FP32, choices=[C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_BF16],
                              help='Data type.')
    add_clamp_to_dtype_arg(model_params)
    model_params.add_argument('--amp', action='store_true', help='Use PyTorch automatic mixed precision (AMP) to run compatible operations in float16 mode instead of float32.')
    model_params.add_argument('--apex-amp', action='store_true', help='Use NVIDIA Apex automatic mixed precision (AMP) to run the entire model in float16 mode with float32 master weights and dynamic loss scaling. This is faster than PyTorch AMP with some additional risk and requires installing Apex: https://github.com/NVIDIA/apex')
    model_params.add_argument('--neural-vocab-selection', type=str, default=None, choices=C.NVS_TYPES,
                              help='When enabled the model contains a neural vocabulary selection model that restricts the target output vocabulary to speed up inference.logit_max: predictions are made per source token and combined by max pooling.eos: the prediction is based on the hidden representation of the <eos> token.')
    model_params.add_argument('--neural-vocab-selection-block-loss', action='store_true',
                              help="When enabled, gradients for NVS are blocked from propagating back to the encoder. This means that NVS learns to work with the main model's representations but does not influence its training.")


def add_batch_args(params: argparse.ArgumentParser, default_batch_size: int = 4096, default_batch_type: str = C.BATCH_TYPE_WORD) -> None:
    params.add_argument('--batch-size', '-b', type=int_greater_or_equal(1), default=default_batch_size,
                        help=('Mini-batch size per process. Depending on --batch-type, this either refers to words or sentences. '
                              'The effective batch size (update size) is num_processes * batch_size * update_interval. Default: %(default)s.'))
    params.add_argument('--batch-type', type=str, default=default_batch_type, choices=C.BATCH_TYPES,
                        help='sentence: each batch contains exactly X sentences. word: each batch contains approximately X target words. max-word: each batch contains at most X target words. Default: %(default)s.')
    params.add_argument('--batch-sentences-multiple-of', type=int, default=8,
                        help='For word and max-word batching, guarantee that each batch contains a multiple of X sentences. For word batching, round up or down to nearest multiple. For max-word batching, always round down. Default: %(default)s.')
    params.add_argument('--update-interval', type=int, default=1,
                        help='Accumulate gradients over X batches for each model update. Set a value higher than 1 to simulate large batches (ex: batch_size 2560 with update_interval 4 gives effective batch size 10240). Default: %(default)s.')


def add_nvs_train_parameters(params: argparse.ArgumentParser) -> None:
    params.add_argument('--bow-task-weight', type=float_greater_or_equal(0.0), default=1.0,
                        help='The weight of the auxiliary Bag-of-word (BOW) loss when --neural-vocab-selection is enabled. Default %(default)s.')
    params.add_argument('--bow-task-pos-weight', type=float_greater_or_equal(0.0), default=10,
                        help='The weight of the positive class (the set of words present on the target side) for the BOW loss when --neural-vocab-selection is set as x * num_negative_class / num_positive_class where x is the --bow-task-pos-weight. Higher values will bias more towards recall, resulting in larger vocabularies at test time trading off larger vocabularies for higher translation quality. Default %(default)s.')


def add_training_args(params: argparse.ArgumentParser) -> None:
    train_params = params.add_argument_group('Training parameters')
    add_batch_args(train_params)
    train_params.add_argument('--label-smoothing', default=0.1, type=float,
                              help='Smoothing constant for label smoothing. Default: %(default)s.')
    train_params.add_argument('--label-smoothing-impl', default='mxnet', choices=['mxnet', 'fairseq', 'torch'],
                              help='Choose label smoothing implementation. Default: %(default)s. `torch` requires PyTorch 1.10.')
    train_params.add_argument('--length-task', type=str, default=None, choices=[C.LENGTH_TASK_RATIO, C.LENGTH_TASK_LENGTH],
                              help='If specified, adds an auxiliary task during training to predict source/target length ratios (mean squared error loss), or absolute lengths (Poisson) loss. Default %(default)s.')
    train_params.add_argument('--length-task-weight', type=float_greater_or_equal(0.0), default=1.0,
                              help='The weight of the auxiliary --length-task loss. Default %(default)s.')
    train_params.add_argument('--length-task-layers', type=int_greater_or_equal(1), default=1,
                              help='Number of fully-connected layers for predicting the length ratio. Default %(default)s.')
    add_nvs_train_parameters(train_params)
    train_params.add_argument('--target-factors-weight', type=float, nargs='+', default=[1.0],
                              help='Weights of target factor losses. If one value is given, it applies to all secondary target factors. For multiple values, the number of weights given has to match the number of target factors. Default: %(default)s.')
    train_params.add_argument('--optimized-metric', default=C.PERPLEXITY, choices=C.METRICS,
                              help='Metric to optimize with early stopping {%(choices)s}. Default: %(default)s.')
    train_params.add_argument(C.TRAIN_ARGS_CHECKPOINT_INTERVAL, type=int_greater_or_equal(1), default=4000,
                              help='Checkpoint and evaluate every x updates (update-interval * batches). Default: %(default)s.')
    train_params.add_argument('--min-samples', type=int, default=None,
                              help='Minimum number of samples before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-samples', type=int, default=None,
                              help='Maximum number of samples. Default: %(default)s.')
    train_params.add_argument('--min-updates', type=int, default=None,
                              help='Minimum number of updates before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-updates', type=int, default=None,
                              help='Maximum number of updates. Default: %(default)s.')
    train_params.add_argument('--max-seconds', type=int, default=None,
                              help='Training will stop on the next checkpoint after reaching the maximum seconds. Default: %(default)s.')
    train_params.add_argument('--max-checkpoints', type=int, default=None,
                              help='Maximum number of checkpoints to continue training the model before training is stopped. Default: %(default)s.')
    train_params.add_argument('--max-num-checkpoint-not-improved', type=int, default=None,
                              help='Maximum number of checkpoints the model is allowed to not improve in <optimized-metric> on validation data before training is stopped. Default: %(default)s.')
    train_params.add_argument('--checkpoint-improvement-threshold', type=float, default=0.0,
                              help='Improvement in <optimized-metric> over specified number of checkpoints must exceed this value to be considered actual improvement. Default: %(default)s.')
    train_params.add_argument('--min-num-epochs', type=int, default=None,
                              help='Minimum number of epochs (passes through the training data) before training can stop. Default: %(default)s.')
    train_params.add_argument('--max-num-epochs', type=int, default=None,
                              help='Maximum number of epochs (passes through the training data) Default: %(default)s.')
    train_params.add_argument('--embed-dropout', type=multiple_values(2, data_type=float), default=(0.0, 0.0),
                              help='Dropout probability for source & target embeddings. Use "x:x" to specify separate values. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-attention', type=multiple_values(2, data_type=float), default=(0.1, 0.1),
                              help='Dropout probability for multi-head attention. Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-act', type=multiple_values(2, data_type=float), default=(0.1, 0.1),
                              help='Dropout probability before activation in feed-forward block. Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    train_params.add_argument('--transformer-dropout-prepost', type=multiple_values(2, data_type=float), default=(0.1, 0.1),
                              help='Dropout probability for pre/postprocessing blocks. Use "x:x" to specify separate values for encoder & decoder. Default: %(default)s.')
    train_params.add_argument('--optimizer', default=C.OPTIMIZER_ADAM, choices=C.OPTIMIZERS,
                              help='SGD update rule. Default: %(default)s.')
    train_params.add_argument('--optimizer-betas', type=multiple_values(2, data_type=float), default=(0.9, 0.999),
                              help='Beta1 and beta2 for Adam-like optimizers, specified "x:x". Default: %(default)s.')
    train_params.add_argument('--optimizer-eps', type=float_greater_or_equal(0), default=1e-08,
                              help='Optimizer epsilon. Default: %(default)s.')
    train_params.add_argument('--dist', action='store_true',
                              help='Run in distributed training mode. When using this option, launch training with `torchrun --nproc_per_node N -m sockeye.train`. Increasing the number of processes multiplies the effective batch size (ex: batch_size 2560 with `--nproc_per_node 4` gives effective batch size 10240).')
    train_params.add_argument('--initial-learning-rate', type=float, default=0.0002,
                              help='Initial learning rate. Default: %(default)s.')
    train_params.add_argument('--weight-decay', type=float, default=0.0,
                              help='Weight decay constant. Default: %(default)s.')
    train_params.add_argument('--momentum', type=float, default=0.0,
                              help='Momentum constant. Default: %(default)s.')
    train_params.add_argument('--gradient-clipping-threshold', type=float, default=1.0,
                              help='Clip absolute gradients values greater than this value. Set to negative to disable. Default: %(default)s.')
    train_params.add_argument('--gradient-clipping-type', choices=C.GRADIENT_CLIPPING_TYPES, default=C.GRADIENT_CLIPPING_TYPE_NONE,
                              help='The type of gradient clipping. Default: %(default)s.')
    train_params.add_argument('--learning-rate-scheduler-type', default=C.LR_SCHEDULER_PLATEAU_REDUCE, choices=C.LR_SCHEDULERS,
                              help='Learning rate scheduler type. Default: %(default)s.')
    train_params.add_argument('--learning-rate-reduce-factor', type=float, default=0.9,
                              help="Factor to multiply learning rate with (for 'plateau-reduce' learning rate scheduler). Default: %(default)s.")
    train_params.add_argument('--learning-rate-reduce-num-not-improved', type=int, default=8,
                              help="For 'plateau-reduce' learning rate scheduler. Adjust learning rate if <optimized-metric> did not improve for x checkpoints. Default: %(default)s.")
    train_params.add_argument('--learning-rate-warmup', type=int, default=0,
                              help='Number of warmup steps. If set to x, linearly increases learning rate from 10%% to 100%% of the initial learning rate. Default: %(default)s.')
    train_params.add_argument('--no-reload-on-learning-rate-reduce', action='store_true', default=False,
                              help='Do not reload the best training checkpoint when reducing the learning rate. Default: %(default)s.')
    train_params.add_argument('--fixed-param-strategy', default=None, choices=C.FIXED_PARAM_STRATEGY_CHOICES,
                              help='Fix various parameters during training using a named strategy. The strategy name indicates which parameters will be fixed (Wuebker et al., 2018). Default: %(default)s.')
    train_params.add_argument('--fixed-param-names', default=[], nargs='*',
                              help='Manually specify names of parameters to fix during training. Default: %(default)s.')
    train_params.add_argument('--local_rank', type=int_greater_or_equal(0), default=None,
                              help='The DeepSpeed launcher (`deepspeed`) automatically adds this argument. When it is present, training runs in DeepSpeed mode. This argument does not need to be specified manually.')
    train_params.add_argument('--deepspeed-fp16', action='store_true', default=False,
                              help='Run the model in float16 mode with float32 master weights and dynamic loss scaling. This is similar to --apex-amp. Default: %(default)s.')
    train_params.add_argument('--deepspeed-bf16', action='store_true', default=False,
                              help='Run the model in bfloat16 mode, which does not require loss scaling. Default: %(default)s.')
    train_params.add_argument(C.TRAIN_ARGS_MONITOR_BLEU, default=500, type=int,
                              help='x>0: decode x sampled sentences from validation data and compute evaluation metrics. x==-1: use full validation data. Default: %(default)s.')
    train_params.add_argument(C.TRAIN_ARGS_STOP_ON_DECODER_FAILURE, action='store_true',
                              help='Stop training as soon as any checkpoint decoder fails (e.g. because there is not enough GPU memory). Default: %(default)s.')
    train_params.add_argument('--seed', type=int, default=1,
                              help='Random seed. Default: %(default)s.')
    train_params.add_argument('--keep-last-params', type=int, default=-1,
                              help='Keep only the last n params files, use -1 to keep all files. Default: %(default)s')
    train_params.add_argument('--keep-initializations', action='store_true',
                              help='In addition to keeping the last n params files, also keep params from checkpoint 0.')
    train_params.add_argument('--cache-last-best-params', required=False, type=int, default=0,
                              help='Cache the last n best params files, as distinct from the last n in sequence. Use 0 or negative to disable. Default: %(default)s')
    train_params.add_argument('--cache-strategy', required=False, type=str, default=C.AVERAGE_BEST, choices=C.AVERAGE_CHOICES,
                              help='Strategy to use when deciding which are the "best" params files. Default: %(default)s')
    train_params.add_argument('--cache-metric', required=False, type=str, default=C.PERPLEXITY, choices=C.METRICS,
                              help='Metric to use when deciding which are the "best" params files. Default: %(default)s')
    train_params.add_argument('--dry-run', action='store_true',
                              help='Do not perform any actual training, but print statistics about the model and mode of operation.')


def add_train_cli_args(params: argparse.ArgumentParser) -> None:
    add_training_io_args(params)
    add_model_parameters(params)
    add_training_args(params)
    add_device_args(params)
    add_logging_args(params)


def add_translate_cli_args(params: argparse.ArgumentParser) -> None:
    add_inference_args(params)
    add_device_args(params)
    add_logging_args(params)
    add_knn_mt_args(params)


def add_score_cli_args(params: argparse.ArgumentParser) -> None:
    add_training_data_args(params, required=True)
    add_vocab_args(params)
    add_device_args(params)
    add_batch_args(params, default_batch_size=56, default_batch_type=C.BATCH_TYPE_SENTENCE)
    score_params = params.add_argument_group('Scoring parameters')
    score_params.add_argument('--model', '-m', required=True,
                              help='Model directory containing trained model.')
    score_params.add_argument(C.TRAINING_ARG_MAX_SEQ_LEN, type=multiple_values(num_values=2, greater_or_equal=1), default=None,
                              help='Maximum sequence length in tokens.Use "x:x" to specify separate values for src&tgt. Default: Read from model.')
    add_length_penalty_args(score_params)
    add_brevity_penalty_args(score_params)
    score_params.add_argument('--output', '-o', default=None,
                              help='File to write output to. Default: STDOUT.')
    score_params.add_argument('--output-type', default=C.OUTPUT_HANDLER_SCORE, choices=C.OUTPUT_HANDLERS_SCORING,
                              help='Output type. Default: %(default)s.')
    score_params.add_argument('--score-type', choices=C.SCORING_TYPE_CHOICES, default=C.SCORING_TYPE_DEFAULT,
                              help='Score type to output. Default: %(default)s')
    score_params.add_argument('--softmax-temperature', type=float, default=None,
                              help='Controls peakiness of model predictions. Values < 1.0 produce peaked predictions, values > 1.0 produce smoothed distributions.')
    score_params.add_argument('--dtype', default=None, choices=[None, C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_BF16, C.DTYPE_INT8],
                              help='Data type. Default: infers from saved model.')
    add_logging_args(score_params)


def add_state_generation_args(params: argparse.ArgumentParser) -> None:
    add_training_data_args(params, required=True)
    add_vocab_args(params)
    add_device_args(params)
    add_batch_args(params, default_batch_size=56, default_batch_type=C.BATCH_TYPE_SENTENCE)
    decode_params = params.add_argument_group('Decoder state generation parameters')
    params.add_argument('--state-dtype', default=None, choices=[None, C.DTYPE_FP32, C.DTYPE_FP16],
                        help='Data type of the decoder state store. Default: infers from saved model.')
    params.add_argument('--model', '-m', required=True,
                        help='Model directory containing trained model.')
    params.add_argument(C.TRAINING_ARG_MAX_SEQ_LEN, type=multiple_values(num_values=2, greater_or_equal=1), default=None,
                        help='Maximum sequence length in tokens.Use "x:x" to specify separate values for src&tgt. Default: Read from model.')
    add_length_penalty_args(params)
    add_brevity_penalty_args(params)
    params.add_argument('--output-dir', '-o', default=None,
                        help='The path to the directory that stores the decoder states.')
    params.add_argument('--dtype', default=None, choices=[None, C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_INT8],
                        help='Data type. Default: infers from saved model.')
    add_logging_args(params)


def add_inference_args(params: argparse.ArgumentParser) -> None:
    decode_params = params.add_argument_group('Inference parameters')
    decode_params.add_argument(C.INFERENCE_ARG_INPUT_LONG, C.INFERENCE_ARG_INPUT_SHORT, default=None,
                               help='Input file to translate. One sentence per line. If not given, will read from stdin.')
    decode_params.add_argument(C.INFERENCE_ARG_INPUT_FACTORS_LONG, C.INFERENCE_ARG_INPUT_FACTORS_SHORT, required=False, nargs='+', type=regular_file(), default=None,
                               help='List of input files containing additional source factors,each token-parallel to the source. Default: %(default)s.')
    decode_params.add_argument('--json-input', action='store_true', default=False,
                               help="If given, the CLI expects string-serialized json objects as input. Requires at least the input text field, for example: {'text': 'some input string'} Optionally, a list of factors can be provided: {'text': 'some input string', 'factors': ['C C C', 'X X X']}.")
    decode_params.add_argument(C.INFERENCE_ARG_OUTPUT_LONG, C.INFERENCE_ARG_OUTPUT_SHORT, default=None,
                               help='Output file to write translations to. If not given, will write to stdout.')
    decode_params.add_argument('--models', '-m', required=True, nargs='+',
                               help='Model folder(s). Use multiple for ensemble decoding. Model determines config, best parameters and vocab files.')
    decode_params.add_argument('--checkpoints', '-c', default=None, type=int, nargs='+',
                               help='If not given, chooses best checkpoints for model(s). If specified, must have the same length as --models and be integer')
    decode_params.add_argument('--nbest-size', type=int_greater_or_equal(1), default=1,
                               help='Size of the nbest list of translations. Default: %(default)s.')
    decode_params.add_argument('--beam-size', '-b', type=int_greater_or_equal(1), default=5,
                               help='Size of the beam. Default: %(default)s.')
    decode_params.add_argument('--greedy', '-g', action='store_true', default=False,
                               help='Enables an alternative, faster greedy decoding implementation. It does not support batch decoding, ensembles, and hypothesis scores are not normalized. Default: %(default)s.')
    decode_params.add_argument('--beam-search-stop', choices=[C.BEAM_SEARCH_STOP_ALL, C.BEAM_SEARCH_STOP_FIRST], default=C.BEAM_SEARCH_STOP_ALL,
                               help='Stopping criteria. Quit when (all) hypotheses are finished or when a finished hypothesis is in (first) position. Default: %(default)s.')
    decode_params.add_argument('--batch-size', type=int_greater_or_equal(1), default=1,
                               help='Batch size during decoding. Determines how many sentences are translated simultaneously. Default: %(default)s.')
    decode_params.add_argument('--chunk-size', type=int_greater_or_equal(1), default=None,
                               help='Size of the chunks to be read from input at once. The chunks are sorted and then split into batches. Therefore the larger the chunk size the better the grouping of segments of similar length and therefore the higher the increase in throughput. Default: %d without batching and %d * batch_size with batching.' % (C.CHUNK_SIZE_NO_BATCHING, C.CHUNK_SIZE_PER_BATCH_SEGMENT))
    decode_params.add_argument('--sample', type=int_greater_or_equal(0), default=None, nargs='?', const=0,
                               help='Sample from softmax instead of taking best. Optional argument will restrict sampling to top N vocabulary items at each step. Default: %(default)s.')
    decode_params.add_argument('--seed', type=int, default=None,
                               help='Random seed used if sampling. Default: %(default)s.')
    decode_params.add_argument('--ensemble-mode', type=str, default='linear', choices=['linear', 'log_linear'],
                               help='Ensemble mode. Default: %(default)s.')
    decode_params.add_argument('--bucket-width', type=int_greater_or_equal(0), default=10,
                               help='Bucket width for encoder steps. 0 means no bucketing. Default: %(default)s.')
    decode_params.add_argument('--max-input-length', type=int_greater_or_equal(1), default=None,
                               help='Maximum input sequence length. Default: value from model(s).')
    decode_params.add_argument('--max-output-length-num-stds', type=int, default=C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                               help='Number of target-to-source length ratio standard deviations from training to add to calculate maximum output length for beam search for each sentence. Default: %(default)s.')
    decode_params.add_argument('--max-output-length', type=int_greater_or_equal(1), default=None,
                               help='Maximum number of words to generate during translation. If None, it will be computed automatically. Default: %(default)s.')
    decode_params.add_argument('--restrict-lexicon', nargs='+', type=multiple_values(num_values=2, data_type=str), default=None,
                               help='Specify block or top-k lexicon. A top-k lexicon will pose a positive constraint, by providing the set of allowed target words. While a blocking lexicon poses a negative constraint on providing a set of target words to be avoided. Specifically, a top-k lexicon will restrict the output vocabulary to the k most likely context-free translations of the source words in each sentence (Devlin, 2017). See the lexicon module for creating lexicons, i.e. by running sockeye-lexicon. To use multiple lexicons, provide \'--restrict-lexicon key1:path1 key2:path2 ...\' and use JSON input to specify the lexicon for each sentence: {"text": "some input string", "restrict_lexicon": "key"}. If a single lexicon is specified it will be applied to all inputs. If multiple lexica are specified they can be selected via the JSON input or it can be skipped by not providing a lexicon in the JSON input. Default: %(default)s.')
    decode_params.add_argument('--restrict-lexicon-topk', type=int, default=None,
                               help='Specify the number of translations to load for each source word from the lexicon given with --restrict-lexicon top-k lexicon. Default: Load all entries from the lexicon.')
    decode_params.add_argument('--skip-nvs', action='store_true', help='Manually turn off Neural Vocabulary Selection (NVS) to do a softmax over the full target vocabulary.', default=False)
    decode_params.add_argument('--nvs-thresh', type=float, help='The probability threshold for a word to be added to the set of target words. Default: 0.5.', default=0.5)
    decode_params.add_argument('--strip-unknown-words', action='store_true', default=False,
                               help='Remove any <unk> symbols from outputs. Default: %(default)s.')
    decode_params.add_argument('--prevent-unk', action='store_true', default=False,
                               help='Avoid generating <unk> during decoding. Default: %(default)s.')
    decode_params.add_argument('--output-type', default='translation', choices=C.OUTPUT_HANDLERS,
                               help='Output type. Default: %(default)s.')
    add_length_penalty_args(decode_params)
    add_brevity_penalty_args(decode_params)
    decode_params.add_argument('--dtype', default=None, choices=[None, C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_BF16, C.DTYPE_INT8],
                               help='Data type. Default: infers from saved model.')
    add_clamp_to_dtype_arg(decode_params)


def add_length_penalty_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--length-penalty-alpha', default=1.0, type=float,
                        help='Alpha factor for the length penalty used in beam search: (beta + len(Y))**alpha/(beta + 1)**alpha. A value of 0.0 will therefore turn off length normalization. Default: %(default)s.')
    params.add_argument('--length-penalty-beta', default=0.0, type=float,
                        help='Beta factor for the length penalty used in scoring: (beta + len(Y))**alpha/(beta + 1)**alpha. Default: %(default)s')


def add_brevity_penalty_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--brevity-penalty-type', default='none', type=str, choices=[C.BREVITY_PENALTY_NONE, C.BREVITY_PENALTY_LEARNED, C.BREVITY_PENALTY_CONSTANT],
                        help="If specified, adds brevity penalty to the hypotheses' scores, calculated with learned or constant length ratios. The latter, by default, uses the length ratio (|ref|/|hyp|) estimated from the training data and averaged over models. Default: %(default)s.")
    params.add_argument('--brevity-penalty-weight', default=1.0, type=float_greater_or_equal(0.0),
                        help='Scaler for the brevity penalty in beam search: weight * log(BP) + score. Default: %(default)s')
    params.add_argument('--brevity-penalty-constant-length-ratio', default=0.0, type=float_greater_or_equal(0.0),
                        help="Has effect if --brevity-penalty-type is set to 'constant'. If positive, overrides the length ratio, used for brevity penalty calculation, for all inputs. If zero, uses the average of length ratios from the training data over all models. Default: %(default)s.")


def add_clamp_to_dtype_arg(params: argparse.ArgumentParser) -> None:
    params.add_argument('--clamp-to-dtype', action='store_true',
                        help='Clamp outputs of transformer attention, feed-forward networks, and process blocks to the min/max finite values for the current dtype. This can prevent inf/nan values from overflow when running large models in float16 mode. See: https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139')


def add_evaluate_args(params: argparse.ArgumentParser) -> None:
    eval_params = params.add_argument_group('Evaluate parameters')
    eval_params.add_argument('--references', '-r', required=True, type=str,
                             help='File with references.')
    eval_params.add_argument('--hypotheses', '-i', type=file_or_stdin(), default=[sys.stdin], nargs='+',
                             help='File(s) with hypotheses. If none will read from stdin. Default: stdin.')
    eval_params.add_argument('--metrics', nargs='+', choices=C.EVALUATE_METRICS, default=[C.BLEU, C.CHRF, C.TER],
                             help='List of metrics to compute. Default: %(default)s.')
    eval_params.add_argument('--sentence', '-s', action='store_true',
                             help='Show sentence-level metrics. Default: %(default)s.')
    eval_params.add_argument('--offset', type=float, default=0.01,
                             help='Numerical value of the offset of zero n-gram counts for BLEU. Default: %(default)s.')
    eval_params.add_argument('--not-strict', '-n', action='store_true',
                             help='Do not fail if number of hypotheses does not match number of references. Default: %(default)s.')


def add_build_vocab_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('-i', '--inputs', required=True, nargs='+',
                        help='List of text files to build vocabulary from.')
    params.add_argument('-o', '--output', required=True, type=str,
                        help='Output filename to write vocabulary to.')
    add_vocab_args(params)
    add_process_pool_args(params)


def add_knn_mt_args(params: argparse.ArgumentParser) -> None:
    knn_params = params.add_argument_group('kNN MT parameters')
    knn_params.add_argument('--knn-index', type=str, help='Optionally use a KNN index during inference to retrieve similar hidden states and corresponding target tokens.', default=None)
    knn_params.add_argument('--knn-lambda', type=float, help='Interpolation parameter when using KNN index. Default: %(default)s.', default=C.DEFAULT_KNN_LAMBDA)


def add_build_knn_index_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('-i', '--input-dir', required=True, type=str,
                        help=f'The directory that contains the stored decoder states and values ({C.KNN_STATE_DATA_STORE_NAME} and {C.KNN_WORD_DATA_STORE_NAME}).')
    params.add_argument('-o', '--output-dir', default=None, type=str,
                        help='The path to the output directory. Will reuse input directory if not specified.')
    params.add_argument('-t', '--index-type', default=None, type=str,
                        help='An optional field to specify the type of the index. Will override settings in the config. The type is specified with a faiss index factory signature, see here: https://github.com/facebookresearch/faiss/wiki/The-index-factory')
    params.add_argument('--train-data-input-file', default=None, type=str,
                        help='An optional field to reuse an already-built training data sample for the index. Otherwise, a (slow) sampling step might need to be run.')
    params.add_argument('--train-data-size', default=None, type=int,
                        help='An optional field to specify the size of the training sample. Will override settings in the config.')


# End of file.
