from typing import Any

# === Internal dependency: allennlp.commands.train ===
class Train(Subcommand): ...
def train_model_from_args(args: argparse.Namespace) -> Any: ...
def train_model(params: Params, serialization_dir: Union[str, PathLike], recover: bool = ..., force: bool = ..., node_rank: int = ..., include_package: List[str] = ..., dry_run: bool = ..., file_friendly_logging: bool = ..., return_model: Optional[bool] = ...) -> Optional[Model]: ...
class TrainModel(Registrable):
    def __init__(self, serialization_dir: str, model: Model, trainer: Trainer, evaluation_data_loader: DataLoader = ..., evaluate_on_test: bool = ..., batch_weight_key: str = ...) -> None: ...
    def run(self) -> Dict[str, Any]: ...

# === Internal dependency: allennlp.common.Params ===
from_file: Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.testing ===
def cpu_or_gpu(test_method) -> Any: ...
# re-export: from allennlp.common.testing.test_case import AllenNlpTestCase

# === Internal dependency: allennlp.models ===
# re-export: from allennlp.models.archival import load_archive

# === Internal dependency: allennlp.models.Model ===
from_archive: Any

# === Internal dependency: allennlp.models.archival ===
CONFIG_NAME: str

# === Internal dependency: allennlp.training ===
# re-export: from allennlp.training.callbacks import TrainerCallback

# === Internal dependency: allennlp.training.learning_rate_schedulers ===
# re-export: from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
# re-export: from allennlp.training.learning_rate_schedulers.pytorch_lr_schedulers import ExponentialLearningRateScheduler

# === Internal dependency: allennlp.version ===
VERSION: Any

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, warns

# === Third-party dependency: torch ===
# Used symbols: cuda, distributed