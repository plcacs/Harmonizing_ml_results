from typing import Any

# === Internal dependency: allennlp.commands.train ===
class Train(Subcommand): ...
def train_model_from_args(args): ...
def train_model(params, serialization_dir, recover=..., force=..., node_rank=..., include_package=..., dry_run=..., file_friendly_logging=..., return_model=...): ...
class TrainModel(Registrable):
    def __init__(self, serialization_dir, model, trainer, evaluation_data_loader=..., evaluate_on_test=..., batch_weight_key=...): ...
    def run(self): ...

# === Internal dependency: allennlp.common.Params ===
from_file: Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.testing ===
def cpu_or_gpu(test_method): ...
from allennlp.common.testing.test_case import AllenNlpTestCase

# === Internal dependency: allennlp.models ===
from allennlp.models.archival import load_archive

# === Internal dependency: allennlp.models.Model ===
from_archive: Any

# === Internal dependency: allennlp.models.archival ===
CONFIG_NAME = 'config.json'

# === Internal dependency: allennlp.training ===
from allennlp.training.callbacks import TrainerCallback

# === Internal dependency: allennlp.training.learning_rate_schedulers ===
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from allennlp.training.learning_rate_schedulers.pytorch_lr_schedulers import ExponentialLearningRateScheduler

# === Internal dependency: allennlp.version ===
_MAJOR = '2'
_MINOR = '10'
_PATCH = '1'
_SUFFIX = os.environ.get(...)
VERSION = ...(...)

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, warns

# === Third-party dependency: torch ===
# Used symbols: cuda, distributed