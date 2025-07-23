import logging
from typing import Any, Optional, TypeVar, Generic
from snorkel.classification.multitask_classifier import MultitaskClassifier
from snorkel.types import Config
from .checkpointer import Checkpointer
from .log_writer import LogWriter

T = TypeVar('T', bound=MultitaskClassifier)

class LogManagerConfig(Config):
    """Manager for checkpointing model.

    Parameters
    ----------
    counter_unit
        The unit to use when assessing when it's time to log.
        Options are ["epochs", "batches", "points"]
    evaluation_freq:
        Evaluate performance on the validation set every this many counter_units
    """
    counter_unit: str = 'epochs'
    evaluation_freq: float = 1.0

class LogManager(Generic[T]):
    """A class to manage logging during training progress.

    Parameters
    ----------
    n_batches_per_epoch
        Total number batches per epoch
    log_writer
        ``LogWriter`` for current run logs
    checkpointer
        ``Checkpointer`` for current model
    kwargs
        Settings to update in LogManagerConfig
    """

    def __init__(
        self,
        n_batches_per_epoch: int,
        log_writer: Optional[LogWriter] = None,
        checkpointer: Optional[Checkpointer] = None,
        **kwargs: Any
    ) -> None:
        self.config: LogManagerConfig = LogManagerConfig(**kwargs)
        self.n_batches_per_epoch: int = n_batches_per_epoch
        self.log_writer: Optional[LogWriter] = log_writer
        self.checkpointer: Optional[Checkpointer] = checkpointer
        self.counter_unit: str = self.config.counter_unit
        if self.counter_unit not in ['points', 'batches', 'epochs']:
            raise ValueError(f'Unrecognized counter_unit: {self.counter_unit}')
        self.evaluation_freq: float = self.config.evaluation_freq
        logging.info(f'Evaluating every {self.evaluation_freq} {self.counter_unit}.')
        self.point_count: int = 0
        self.point_total: int = 0
        self.batch_count: int = 0
        self.batch_total: int = 0
        self.epoch_count: float = 0.0
        self.epoch_total: float = 0.0
        self.unit_count: float = 0.0
        self.unit_total: float = 0.0
        self.trigger_count: int = 0

    def update(self, batch_size: int) -> None:
        """Update the count and total number."""
        self.point_count += batch_size
        self.point_total += batch_size
        self.batch_count += 1
        self.batch_total += 1
        self.epoch_count = self.batch_count / self.n_batches_per_epoch
        self.epoch_total = self.batch_total / self.n_batches_per_epoch
        if self.counter_unit == 'points':
            self.unit_count = self.point_count
            self.unit_total = self.point_total
        if self.counter_unit == 'batches':
            self.unit_count = self.batch_count
            self.unit_total = self.batch_total
        elif self.counter_unit == 'epochs':
            self.unit_count = self.epoch_count
            self.unit_total = self.epoch_total

    def trigger_evaluation(self) -> bool:
        """Check if current counts trigger evaluation."""
        satisfied: bool = self.unit_count >= self.evaluation_freq
        if satisfied:
            self.trigger_count += 1
            self.reset()
        return satisfied

    def trigger_checkpointing(self) -> bool:
        """Check if current counts trigger checkpointing."""
        if self.checkpointer is None:
            return False
        satisfied: bool = self.trigger_count >= self.checkpointer.checkpoint_factor
        if satisfied:
            self.trigger_count = 0
        return satisfied

    def reset(self) -> None:
        """Reset counters."""
        self.point_count = 0
        self.batch_count = 0
        self.epoch_count = 0
        self.unit_count = 0

    def cleanup(self, model: T) -> T:
        """Close the log writer and checkpointer if needed. Reload best model."""
        if self.log_writer is not None:
            self.log_writer.cleanup()
        if self.checkpointer is not None:
            self.checkpointer.clear()
            model = self.checkpointer.load_best_model(model)
        return model
