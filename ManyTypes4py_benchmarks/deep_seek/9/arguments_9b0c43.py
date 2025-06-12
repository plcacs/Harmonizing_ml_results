"""
Defines commandline arguments for the main CLIs with reasonable defaults.
"""
import argparse
import os
import sys
import types
from typing import Any, Callable, Dict, List, Tuple, Optional, Union, IO, TypeVar, Generic, Sequence
import yaml
from sockeye.utils import smart_open
from . import constants as C

T = TypeVar('T')

class ConfigArgumentParser(argparse.ArgumentParser):
    """
    Extension of argparse.ArgumentParser supporting config files.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.argument_definitions: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self.argument_actions: List[argparse.Action] = []
        self._overwrite_add_argument(self)
        self.add_argument('--config', help="Path to CLI arguments in yaml format (as saved in Sockeye model directories as 'args.yaml'). Commandline arguments have precedence over values in this file.", type=str)

    def _register_argument(self, _action: argparse.Action, *args: str, **kwargs: Any) -> None:
        self.argument_definitions[args] = kwargs
        self.argument_actions.append(_action)

    def _overwrite_add_argument(self, original_object: Any) -> Any:
        def _new_add_argument(this_self: Any, *args: str, **kwargs: Any) -> argparse.Action:
            action = this_self.original_add_argument(*args, **kwargs)
            this_self.config_container._register_argument(action, *args, **kwargs)
            return action
        original_object.original_add_argument = original_object.add_argument
        original_object.config_container = self
        original_object.add_argument = types.MethodType(_new_add_argument, original_object)
        return original_object

    def add_argument_group(self, *args: Any, **kwargs: Any) -> argparse._ArgumentGroup:
        group = super().add_argument_group(*args, **kwargs)
        return self._overwrite_add_argument(group)

    def parse_args(self, args: Optional[Sequence[str]] = None, namespace: Optional[argparse.Namespace] = None) -> argparse.Namespace:
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument('--config', type=regular_file())
        config_args, _ = config_parser.parse_known_args(args=args)
        initial_args = argparse.Namespace()
        if config_args.config:
            initial_args = load_args(config_args.config)
            for action in self.argument_actions:
                if action.dest in initial_args.__dict__:
                    action.required = False
        return super().parse_args(args=args, namespace=initial_args)

class StoreDeprecatedAction(argparse.Action):
    def __init__(self, option_strings: List[str], dest: str, deprecated_dest: str, nargs: Optional[int] = None, **kwargs: Any) -> None:
        super(StoreDeprecatedAction, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
        self.deprecated_dest = deprecated_dest

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: Optional[str] = None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, self.deprecated_dest, values)

def save_args(args: argparse.Namespace, fname: str) -> None:
    with open(fname, 'w') as out:
        yaml.safe_dump(args.__dict__, out, default_flow_style=False)

def load_args(fname: str) -> argparse.Namespace:
    with open(fname, 'r') as inp:
        return argparse.Namespace(**yaml.safe_load(inp))

def regular_file() -> Callable[[str], str]:
    def check_regular_file(value_to_check: str) -> str:
        value_to_check = str(value_to_check)
        if not os.path.isfile(value_to_check):
            raise argparse.ArgumentTypeError('must exist and be a regular file.')
        return value_to_check
    return check_regular_file

def regular_folder() -> Callable[[str], str]:
    def check_regular_directory(value_to_check: str) -> str:
        value_to_check = str(value_to_check)
        if not os.path.isdir(value_to_check):
            raise argparse.ArgumentTypeError('must be a directory.')
        return value_to_check
    return check_regular_directory

def int_greater_or_equal(threshold: int) -> Callable[[str], int]:
    def check_greater_equal(value: str) -> int:
        value_to_check = int(value)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError('must be greater or equal to %d.' % threshold)
        return value_to_check
    return check_greater_equal

def float_greater_or_equal(threshold: float) -> Callable[[str], float]:
    def check_greater_equal(value: str) -> float:
        value_to_check = float(value)
        if value_to_check < threshold:
            raise argparse.ArgumentTypeError('must be greater or equal to %f.' % threshold)
        return value_to_check
    return check_greater_equal

def bool_str() -> Callable[[str], bool]:
    def parse(value: str) -> bool:
        lower_value = value.lower()
        if lower_value in ['true', 'yes', '1']:
            return True
        elif lower_value in ['false', 'no', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Invalid value for bool argument. Use true/false, yes/no or 1/0.')
    return parse

def simple_dict() -> Callable[[str], Dict[str, Union[bool, int, float, str]]]:
    def parse(dict_str: str) -> Dict[str, Union[bool, int, float, str]]:
        def _parse(value: str) -> Union[bool, int, float, str]:
            if value.lower() == 'true':
                return True
            if value.lower() == 'false':
                return False
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
            try:
                return float(value)
            except:
                return value
        _dict: Dict[str, Union[bool, int, float, str]] = dict()
        try:
            for entry in dict_str.split(','):
                key, value = entry.split(':')
                _dict[key] = _parse(value)
        except ValueError:
            raise argparse.ArgumentTypeError('Specify argument dictionary as key1:value1,key2:value2,...')
        return _dict
    return parse

def multiple_values(num_values: int = 0, greater_or_equal: Optional[int] = None, data_type: type = int) -> Callable[[str], Tuple[Any, ...]]:
    def parse(value_to_check: str) -> Tuple[Any, ...]:
        if ':' in value_to_check:
            expected_num_separators = num_values - 1 if num_values else 0
            if expected_num_separators > 0 and value_to_check.count(':') != expected_num_separators:
                raise argparse.ArgumentTypeError('Expected either a single value or %d values separated by %s' % (num_values, C.ARG_SEPARATOR))
            values = tuple(map(data_type, value_to_check.split(C.ARG_SEPARATOR, num_values - 1)))
        else:
            values = tuple([data_type(value_to_check)] * num_values)
        if greater_or_equal is not None:
            if any((value < greater_or_equal for value in values)):
                raise argparse.ArgumentTypeError('Must provide value greater or equal to %d' % greater_or_equal)
        return values
    return parse

def file_or_stdin() -> Callable[[Optional[str]], IO]:
    def parse(path: Optional[str]) -> IO:
        if path is None or path == '-':
            return sys.stdin
        else:
            return smart_open(path)
    return parse

def add_average_args(params: argparse.ArgumentParser) -> None:
    average_params = params.add_argument_group('Averaging')
    average_params.add_argument('inputs', metavar='INPUT', type=str, nargs='+', help='either a single model directory (automatic checkpoint selection) or multiple .params files (manual checkpoint selection)')
    average_params.add_argument('--metric', help='Name of the metric to choose n-best checkpoints from. Default: %(default)s.', default=C.PERPLEXITY, choices=C.METRICS)
    average_params.add_argument('-n', type=int, default=4, help='number of checkpoints to find. Default: %(default)s.')
    average_params.add_argument('--output', '-o', required=True, type=str, help='File to write averaged parameters to.')
    average_params.add_argument('--strategy', choices=C.AVERAGE_CHOICES, default=C.AVERAGE_BEST, help='selection method. Default: %(default)s.')

def add_rerank_args(params: argparse.ArgumentParser) -> None:
    rerank_params = params.add_argument_group('Reranking')
    rerank_params.add_argument('--reference', '-r', type=str, required=True, help='File where target reference translations are stored.')
    rerank_params.add_argument('--hypotheses', '-hy', type=str, required=True, help='File with nbest translations, one nbest list per line,in JSON format as returned by sockeye.translate with --nbest-size x.')
    rerank_params.add_argument('--metric', '-m', type=str, required=False, default=C.RERANK_BLEU, choices=C.RERANK_METRICS, help='Sentence-level metric used to compare each nbest translation to the reference or the source.Default: %(default)s.')
    rerank_params.add_argument('--isometric-alpha', required=False, type=float_greater_or_equal(0.0), default=0.5, help='Alpha factor used for reranking (--isometric-[ratio/diff]) nbest list. Requires optimization on dev set.Default: %(default)s.')
    rerank_params.add_argument('--output', '-o', default=None, help='File to write output to. Default: STDOUT.')
    rerank_params.add_argument('--output-best', action='store_true', help='Output only the best hypothesis from each nbest list.')
    rerank_params.add_argument('--output-best-non-blank', action='store_true', help='When outputting only the best hypothesis (--output-best) and the best hypothesis is a blank line, output following non-blank best from the nbest list.')
    rerank_params.add_argument('--output-reference-instead-of-blank', action='store_true', help='When outputting only the best hypothesis (--output-best) and the best hypothesis is a blank line, output the reference instead.')
    rerank_params.add_argument('--return-score', action='store_true', help='Returns the reranking scores as scores in output JSON objects.')

def add_lexicon_args(params: argparse.ArgumentParser, is_for_block_lexicon: bool = False) -> None:
    lexicon_params = params.add_argument_group('Model & Top-k')
    lexicon_params.add_argument('--model', '-m', required=True, help='Model directory containing source and target vocabularies.')
    if not is_for_block_lexicon:
        lexicon_params.add_argument('-k', type=int, default=200, help='Number of target translations to keep per source. Default: %(default)s.')

def add_lexicon_create_args(params: argparse.ArgumentParser, is_for_block_lexicon: bool = False) -> None:
    lexicon_params = params.add_argument_group('I/O')
    if is_for_block_lexicon:
        input_help = 'A text file with tokens that shall be blocked. All token must be in the model vocabulary.'
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
    logging_params.add_argument('--quiet-secondary-workers', '-qsw', default=False, action='store_true', help='Suppress console logging for secondary workers in distributed training.')
    logging_params.add_argument('--no-logfile', default=False, action='store_true', help='Suppress file logging')
    log_levels = ['INFO', 'DEBUG', 'ERROR']
    logging_params.add_argument('--loglevel', '--log-level', default='INFO', choices=log_levels, help='Log level. Default: %(default)s.')
    logging_params.add_argument('--loglevel-secondary-workers', default='INFO', choices=log_levels, help='Console log level for secondary workers. Default: %(default)s.')

def add_quantize_args(params: argparse.ArgumentParser) -> None:
    params = params.add_argument_group('Quantization')
    params.add_argument('--model', '-m', required=True, help=f'Model (directory) to quantize in place. "{C.PARAMS_BEST_NAME}" will be replaced with a quantized version and "{C.CONFIG_NAME}" will be updated with the new dtype. The original files will be backed up with suffixes indicating the starting dtype (e.g., "{C.PARAMS_BEST_NAME}.{C.DTYPE_FP32}" and "{C.CONFIG_NAME}.{C.DTYPE_FP32}").')
    params.add_argument('--dtype', default=C.DTYPE_FP16, choices=[C.DTYPE_FP32, C.DTYPE_FP16, C.DTYPE_BF16], help='Target data type for quantization. Default: %(default)s.')

def add_training_data_args(params: argparse.ArgumentParser, required: bool = False) -> None:
    params.add_argument(C.TRAINING_ARG_SOURCE, '-s', required=required, type=regular_file(), help='Source side of parallel training data.')
    params.add_argument('--source-factors', '-sf', required=False, nargs='+', type=regular_file(), default=[], help='File(s) containing additional token-parallel source-side factors. Default: %(default)s.')
    params.add_argument('--source-factors-use-source-vocab', required=False, nargs='+', type=bool_str(), default=[], help='List of bools signaling whether to use the source vocabulary for the source factors. If empty (default) each factor has its own vocabulary.')
    params.add_argument('--target-factors', '-tf', required=False, nargs='+', type=regular_file(), default=[], help='File(s) containing additional token-parallel target-side factors. Default: %(default)s.')
    params.add_argument('--target-factors-use-target-vocab', required=False, nargs='+', type=bool_str(), default=[], help='List of bools signaling whether to use the target vocabulary for the target factors. If empty (default) each factor has its own vocabulary.')
    params.add_argument(C.TRAINING_ARG_TARGET, '-t', required=required, type=regular_file(), help='Target side of parallel training data.')
    params.add_argument('--end-of-prepending-tag', type=str, default=None, help='Tag indicating the end of prepended text. Prepended tokens before this tag (inclusive) will be marked, and they will not be counted toward source length when calculating maximum output length for beam search.')

def add_validation_data_params(params: argparse.ArgumentParser) -> None:
    params.add_argument('--validation-source', '-vs', required=True, type=regular_file(), help='Source side of validation data.')
    params.add_argument('--validation-source-factors', '-vsf', required=False, nargs='+', type=regular_file(), default=[], help='File(s) containing additional token-parallel validation source side factors. Default: %(default)s.')
    params.add_argument('--validation-target', '-vt', required=True, type=regular_file(), help='Target side of validation data.')
    params.add_argument('--validation-target-factors', '-vtf', required=False, nargs='+', type=regular_file(), default=[], help='File(s) containing additional token-parallel validation target side factors. Default: %(default)s.')

def add_prepared_data_args(params: argparse.ArgumentParser) -> None:
    params.add_argument(C.TRAINING_ARG_PREPARED_DATA, '-d', type=regular_folder(), help='Prepared training data directory created through python -m sockeye.prepare_data.')

def add_training_output_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--output', '-o', required=True, help='Folder where model & training results are written to.')
    params.add_argument('--overwrite-output', action='store_true', help='Delete all contents of the model directory if it already exists.')

def add_training_io_args(params: argparse.ArgumentParser) -> None:
    params = params.add_argument_group('Data & I/O')
    add_training_data_args(params, required=False)
    add_prepared_data_args(params)
    add_validation_data_params(params)
    add_bucketing_args(params)
    add_vocab_args(params)
    add_training_output_args(params)

def add_bucketing_args(params: argparse.ArgumentParser) -> None:
    params.add_argument('--no-bucketing', action='store_true', help='Disable bucketing: always unroll the graph to --max-seq-len. Default: %(default)s.')
    params.add_argument('--bucket-width', type=int_greater_or_equal(1), default=8, help='Width of buckets in tokens. Default: %(default)s.')
    params.add_argument('--bucket-scaling', action='store_true', help='Scale source/target buckets based on length ratio to reduce padding. Default: %(default)s.')
    params.add_argument(C.TRAINING_ARG_MAX_SEQ_LEN, type=multiple_values(num_values=2, greater_or