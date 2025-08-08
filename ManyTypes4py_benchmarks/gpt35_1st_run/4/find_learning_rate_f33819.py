from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params, Tqdm
from allennlp.common import logging as common_logging
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer, Trainer
from allennlp.training.util import create_serialization_dir, data_loaders_from_params
from typing import List, Tuple

class FindLearningRate(Subcommand):

    def add_subparser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        description: str = 'Find a learning rate range where loss decreases quickly\n                         for the specified model and dataset.'
        subparser: argparse.ArgumentParser = parser.add_parser(self.name, description=description, help='Find a learning rate range.')
        subparser.add_argument('param_path', type=str, help='path to parameter file describing the model to be trained')
        subparser.add_argument('-s', '--serialization-dir', required=True, type=str, help='The directory in which to save results.')
        subparser.add_argument('-o', '--overrides', type=str, default='', help='a json(net) structure used to override the experiment configuration, e.g., \'{"iterator.batch_size": 16}\'.  Nested parameters can be specified either with nested dictionaries or with dot syntax.')
        subparser.add_argument('--start-lr', type=float, default=1e-05, help='learning rate to start the search')
        subparser.add_argument('--end-lr', type=float, default=10, help='learning rate up to which search is done')
        subparser.add_argument('--num-batches', type=int, default=100, help='number of mini-batches to run learning rate finder')
        subparser.add_argument('--stopping-factor', type=float, default=None, help='stop the search when the current loss exceeds the best loss recorded by multiple of stopping factor')
        subparser.add_argument('--linear', action='store_true', help='increase learning rate linearly instead of exponential increase')
        subparser.add_argument('-f', '--force', action='store_true', required=False, help='overwrite the output directory if it exists')
        subparser.add_argument('--file-friendly-logging', action='store_true', default=False, help='outputs tqdm status on separate lines and slows tqdm refresh rate')
        subparser.set_defaults(func=find_learning_rate_from_args)
        return subparser

def find_learning_rate_from_args(args: argparse.Namespace) -> None:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging
    params: Params = Params.from_file(args.param_path, args.overrides)
    find_learning_rate_model(params, args.serialization_dir, start_lr=args.start_lr, end_lr=args.end_lr, num_batches=args.num_batches, linear_steps=args.linear, stopping_factor=args.stopping_factor, force=args.force)

def find_learning_rate_model(params: Params, serialization_dir: str, start_lr: float = 1e-05, end_lr: float = 10, num_batches: int = 100, linear_steps: bool = False, stopping_factor: float = None, force: bool = False) -> None:
    ...

def search_learning_rate(trainer: GradientDescentTrainer, start_lr: float = 1e-05, end_lr: float = 10, num_batches: int = 100, linear_steps: bool = False, stopping_factor: float = None) -> Tuple[List[float], List[float]:
    ...

def _smooth(values: List[float], beta: float) -> List[float]:
    ...

def _save_plot(learning_rates: List[float], losses: List[float], save_path: str) -> None:
    ...
