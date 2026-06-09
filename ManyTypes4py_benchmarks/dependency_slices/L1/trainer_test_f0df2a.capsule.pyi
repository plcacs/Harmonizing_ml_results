from typing import Any

# === Internal dependency: allennlp.common.checks ===
class ConfigurationError(Exception): ...

# === Internal dependency: allennlp.common.params ===
class Params(MutableMapping):
    def __init__(self, params, history=...): ...

# === Internal dependency: allennlp.common.testing ===
def requires_gpu(test_method): ...
def requires_multi_gpu(test_method): ...
from allennlp.common.testing.test_case import AllenNlpTestCase

# === Internal dependency: allennlp.common.testing.confidence_check_test ===
class FakeModelForTestingNormalizationBiasVerification(Model):
    def __init__(self, use_bias=...): ...

# === Internal dependency: allennlp.data ===
from allennlp.data.tokenizers import Token

# === Internal dependency: allennlp.data.Vocabulary ===
from_instances: Any

# === Internal dependency: allennlp.data.data_loaders ===
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader

# === Internal dependency: allennlp.data.dataset_readers ===
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader

# === Internal dependency: allennlp.data.fields ===
from allennlp.data.fields.adjacency_field import AdjacencyField
from allennlp.data.fields.tensor_field import TensorField
from allennlp.data.fields.flag_field import FlagField
from allennlp.data.fields.index_field import IndexField
from allennlp.data.fields.label_field import LabelField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.fields.multilabel_field import MultiLabelField
from allennlp.data.fields.span_field import SpanField
from allennlp.data.fields.text_field import TextField

# === Internal dependency: allennlp.data.token_indexers ===
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer

# === Internal dependency: allennlp.models.model ===
class Model(Module, Registrable):
    ...

# === Internal dependency: allennlp.models.simple_tagger ===
class SimpleTagger(Model):
    def __init__(self, vocab, text_field_embedder, encoder, calculate_span_f1=..., label_encoding=..., label_namespace=..., verbose_metrics=..., initializer=..., **kwargs): ...

# === Internal dependency: allennlp.training ===
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

# === Internal dependency: allennlp.training.Checkpointer ===
_parse_model_state_path: Any
_parse_training_state_path: Any

# === Internal dependency: allennlp.training.callbacks ===
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.console_logger import ConsoleLoggerCallback
from allennlp.training.callbacks.confidence_checks import ConfidenceChecksCallback
from allennlp.training.callbacks.tensorboard import TensorBoardCallback
from allennlp.training.callbacks.track_epoch import TrackEpochCallback
from allennlp.training.callbacks.backward import OnBackwardException
from allennlp.training.callbacks.should_validate import ShouldValidateCallback

# === Internal dependency: allennlp.training.callbacks.confidence_checks ===
class ConfidenceCheckError(Exception): ...

# === Internal dependency: allennlp.training.learning_rate_schedulers ===
from allennlp.training.learning_rate_schedulers.pytorch_lr_schedulers import ExponentialLearningRateScheduler
from allennlp.training.learning_rate_schedulers.pytorch_lr_schedulers import ReduceOnPlateauLearningRateScheduler
from allennlp.training.learning_rate_schedulers.cosine import CosineWithRestarts

# === Internal dependency: allennlp.training.momentum_schedulers ===
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler

# === Internal dependency: allennlp.training.moving_average ===
class ExponentialMovingAverage(MovingAverage):
    def __init__(self, parameters, decay=..., numerator=..., denominator=...): ...

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Third-party dependency: torch ===
# Used symbols: FloatTensor, equal, nn, ones_like, optim, rand, randn

# === Third-party dependency: torch.nn.utils ===
# Used symbols: clip_grad_norm_