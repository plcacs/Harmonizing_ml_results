import copy
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import pytest

import torch
from torch.nn.utils import clip_grad_norm_

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase, requires_gpu, requires_multi_gpu
from allennlp.data import Vocabulary, Instance, Token
from allennlp.data.data_loaders import MultiProcessDataLoader, SimpleDataLoader, TensorDict
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models.model import Model
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.training import (
    GradientDescentTrainer,
    Checkpointer,
)
from allennlp.training.callbacks import (
    TrainerCallback,
    TrackEpochCallback,
    TensorBoardCallback,
    ConfidenceChecksCallback,
    ConsoleLoggerCallback,
    OnBackwardException,
    ShouldValidateCallback,
)
from allennlp.training.callbacks.confidence_checks import ConfidenceCheckError
from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from allennlp.training.learning_rate_schedulers import ExponentialLearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import ExponentialMovingAverage
from allennlp.data.fields import (
    TextField,
    IndexField,
    MetadataField,
    LabelField,
    MultiLabelField,
    SpanField,
    FlagField,
    AdjacencyField,
    TensorField,
)
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing.confidence_check_test import (
    FakeModelForTestingNormalizationBiasVerification,
)


class FakeDatasetReader(DatasetReader):
    def __init__(self, total_instances: int, batch_size: int) -> None:
        super().__init__()
        self.total_instances: int = total_instances
        self.batch_size: int = batch_size

    def _read(self, file_path: str) -> Any:
        for i in range(self.total_instances):
            yield self.text_to_instance(i, "label")

    def text_to_instance(self, index: int, field_type: str) -> Instance:  # type: ignore
        field: TextField = TextField(
            [Token(t) for t in ["The", "number", "is", str(index), "."]],
            token_indexers={"words": SingleIdTokenIndexer("words")},
        )

        return Instance(
            {
                "text": field,
                "label": LabelField(index, skip_indexing=True),
                "flag": FlagField(23),
                "index": IndexField(index % self.batch_size, field),
                "metadata": MetadataField({"some_key": "This will not be logged as a histogram."}),
                "adjacency": AdjacencyField([(0, 1), (1, 2)], field),
                "multilabel": MultiLabelField(["l1", "l2"]),
                "span": SpanField(2, 3, field),
                "tensor": TensorField(torch.randn(2, 3)),
            }
        )


class FakeModel(Model):
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.lin: torch.nn.Linear = torch.nn.Linear(1, 2)
        self.loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        out: torch.Tensor = kwargs["label"].sum().unsqueeze(-1)
        out = out.type(torch.FloatTensor)
        out = self.lin(out)
        loss: torch.Tensor = out.sum()
        return {"loss": loss}


class TrainerTestBase(AllenNlpTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_path: str = str(self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv")
        self.reader: SequenceTaggingDatasetReader = SequenceTaggingDatasetReader(max_instances=4)
        self.data_loader: MultiProcessDataLoader = MultiProcessDataLoader(self.reader, self.data_path, batch_size=2)
        self.data_loader_lazy: MultiProcessDataLoader = MultiProcessDataLoader(
            self.reader, self.data_path, batch_size=2, max_instances_in_memory=10
        )
        self.instances: List[Instance] = list(self.data_loader.iter_instances())
        self.vocab: Vocabulary = Vocabulary.from_instances(self.instances)
        self.data_loader.index_with(self.vocab)
        self.data_loader_lazy.index_with(self.vocab)
        self.model_params: Params = Params(
            {
                "text_field_embedder": {
                    "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                },
                "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
            }
        )
        self.model: SimpleTagger = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)
        self.optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
        self.validation_data_loader: MultiProcessDataLoader = MultiProcessDataLoader(
            self.reader, self.data_path, batch_size=2
        )
        self.validation_data_loader.index_with(self.vocab)


class ZeroGradientsBackwardCallback(TrainerCallback):
    """
    Zeros all gradients after backpropagation.
    """

    def on_backward(
        self,
        trainer: "GradientDescentTrainer",
        batch_outputs: Dict[str, torch.Tensor],
        backward_called: bool,
        **kwargs: Any,
    ) -> bool:
        if backward_called:
            raise OnBackwardException()
        batch_outputs["loss"].backward()
        for param in trainer.model.parameters():
            param.grad.data.zero_()
        return True


class TestTrainer(TrainerTestBase):
    def test_trainer_can_run(self) -> None:
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
        )
        metrics: Dict[str, Any] = trainer.train()
        assert "best_validation_loss" in metrics
        assert isinstance(metrics["best_validation_loss"], float)
        assert "best_validation_accuracy" in metrics
        assert isinstance(metrics["best_validation_accuracy"], float)
        assert "best_validation_accuracy3" in metrics
        assert isinstance(metrics["best_validation_accuracy3"], float)
        assert "best_epoch" in metrics
        assert isinstance(metrics["best_epoch"], int)

        # Making sure that both increasing and decreasing validation metrics work.
        trainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=2,
        )
        metrics = trainer.train()
        assert "best_validation_loss" in metrics
        assert isinstance(metrics["best_validation_loss"], float)
        assert "best_validation_accuracy" in metrics
        assert isinstance(metrics["best_validation_accuracy"], float)
        assert "best_validation_accuracy3" in metrics
        assert isinstance(metrics["best_validation_accuracy3"], float)
        assert "best_epoch" in metrics
        assert isinstance(metrics["best_epoch"], int)
        assert "peak_worker_0_memory_MB" in metrics
        assert isinstance(metrics["peak_worker_0_memory_MB"], float)
        assert metrics["peak_worker_0_memory_MB"] > 0

    def test_train_zero_gradients(self) -> None:
        weights: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.clone()

        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=2,
            validation_data_loader=self.validation_data_loader,
            callbacks=[ZeroGradientsBackwardCallback(serialization_dir=self.TEST_DIR)],
        )
        trainer.train()

        # weights should be the same
        for name, param in self.model.named_parameters():
            assert torch.equal(weights[name], param.data)

    def test_two_backward_callbacks(self) -> None:
        class SecondBackwardCallback(TrainerCallback):
            """
            Changes all gradients to 1 after backpropagation.
            """

            def on_backward(
                self,
                trainer: "GradientDescentTrainer",
                batch_outputs: Dict[str, torch.Tensor],
                backward_called: bool,
                **kwargs: Any,
            ) -> bool:
                if backward_called:
                    raise OnBackwardException()
                batch_outputs["loss"].backward()
                for param in trainer.model.parameters():
                    param.grad = torch.ones_like(param.grad, device=param.grad.device)
                return True

        with pytest.raises(OnBackwardException):
            trainer: GradientDescentTrainer = GradientDescentTrainer(
                self.model,
                self.optimizer,
                self.data_loader,
                num_epochs=2,
                validation_data_loader=self.validation_data_loader,
                callbacks=[
                    ZeroGradientsBackwardCallback(serialization_dir=self.TEST_DIR),
                    SecondBackwardCallback(serialization_dir=self.TEST_DIR),
                ],
            )
            trainer.train()

    def test_trainer_can_run_exponential_moving_average(self) -> None:
        moving_average: ExponentialMovingAverage = ExponentialMovingAverage(self.model.named_parameters(), decay=0.9999)
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
            moving_average=moving_average,
        )
        trainer.train()

    @requires_gpu
    def test_trainer_can_run_cuda(self) -> None:
        self.model.cuda()
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model, self.optimizer, self.data_loader, num_epochs=2, cuda_device=0
        )
        metrics: Dict[str, Any] = trainer.train()
        assert "peak_worker_0_memory_MB" in metrics
        assert isinstance(metrics["peak_worker_0_memory_MB"], float)
        assert metrics["peak_worker_0_memory_MB"] > 0
        assert "peak_gpu_0_memory_MB" in metrics
        assert isinstance(metrics["peak_gpu_0_memory_MB"], float)

    @requires_multi_gpu
    def test_passing_trainer_multiple_gpus_raises_error(self) -> None:
        self.model.cuda()

        with pytest.raises(ConfigurationError):
            GradientDescentTrainer(
                self.model,
                self.optimizer,
                self.data_loader,
                num_epochs=2,
                cuda_device=[0, 1],
            )

    def test_data_loader_lazy_epoch_size_correct(self) -> None:
        num_epochs: int = 3
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader_lazy,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics: Dict[str, Any] = trainer.train()
        epoch: int = metrics["epoch"]
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * 2

    def test_data_loader_lazy_epoch_size_correct_custom_epoch_size(self) -> None:
        self.data_loader_lazy.batches_per_epoch = 3
        num_epochs: int = 3
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader_lazy,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics: Dict[str, Any] = trainer.train()
        epoch: int = metrics["epoch"]
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * 3

    def test_trainer_respects_epoch_size_equals_total(self) -> None:
        batches_per_epoch: int = 4
        num_epochs: int = 3
        data_loader_equal_epoch: SimpleDataLoader = SimpleDataLoader(
            self.instances,
            2,
            batches_per_epoch=batches_per_epoch,
        )
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader_equal_epoch,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics: Dict[str, Any] = trainer.train()
        epoch: int = metrics["epoch"]
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * batches_per_epoch

    def test_trainer_respects_epoch_size_larger_tnan_total(self) -> None:
        batches_per_epoch: int = 7
        num_epochs: int = 3
        data_loader_larger_epoch: SimpleDataLoader = SimpleDataLoader(
            self.instances,
            2,
            batches_per_epoch=batches_per_epoch,
        )
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader_larger_epoch,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics: Dict[str, Any] = trainer.train()
        epoch: int = metrics["epoch"]
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * batches_per_epoch

    def test_trainer_respects_epoch_size_smaller_tnan_total(self) -> None:
        batches_per_epoch: int = 1
        num_epochs: int = 2
        data_loader_smaller_epoch: SimpleDataLoader = SimpleDataLoader(
            self.instances,
            2,
            batches_per_epoch=batches_per_epoch,
        )
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader_smaller_epoch,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics: Dict[str, Any] = trainer.train()
        epoch: int = metrics["epoch"]
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * batches_per_epoch

    def test_trainer_can_resume_training(self) -> None:
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        trainer.train()

        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        new_trainer._maybe_restore_checkpoint()

        assert new_trainer._start_after_epochs_completed == 1

        tracker: Any = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None

        new_trainer.train()

    def test_trainer_can_resume_training_for_exponential_moving_average(self) -> None:
        moving_average: ExponentialMovingAverage = ExponentialMovingAverage(self.model.named_parameters())

        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            moving_average=moving_average,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        trainer.train()

        new_moving_average: ExponentialMovingAverage = ExponentialMovingAverage(self.model.named_parameters())
        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            moving_average=new_moving_average,
            checkpointer=Checkpointer(self.TEST_DIR),
        )

        new_trainer._maybe_restore_checkpoint()
        assert new_trainer._start_after_epochs_completed == 1

        tracker: Any = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None

        new_trainer.train()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_increasing_metric(
        self,
    ) -> None:
        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+acc",
        )
        tracker: Any = new_trainer._metric_tracker

        # when it is the only metric it should be considered the best
        new_tracker: Any = copy.deepcopy(tracker)
        new_tracker.add_metrics({"acc": 1})
        assert new_tracker.is_best_so_far()

        # when it is the same as one before it it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.3]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.is_best_so_far()

        # when it is the best it is considered the best
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 13]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.is_best_so_far()

        # when it is not the the best it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.0013]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.is_best_so_far()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_decreasing_metric(
        self,
    ) -> None:
        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-acc",
        )
        tracker: Any = new_trainer._metric_tracker

        # when it is the only metric it should be considered the best
        new_tracker: Any = copy.deepcopy(tracker)
        new_tracker.add_metrics({"acc": 1})
        assert new_tracker.is_best_so_far()

        # when it is the same as one before it it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.3]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.is_best_so_far()

        # when it is the best it is considered the best
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.0013]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.is_best_so_far()

        # when it is not the the best it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 13]:
            new_tracker.add_metrics({"acc": acc})

    def test_should_stop_early_with_increasing_metric(self) -> None:
        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+acc",
        )

        tracker: Any = new_trainer._metric_tracker

        new_tracker: Any = copy.deepcopy(tracker)
        for acc in [0.5, 0.3, 0.2, 0.1, 0.4, 0.4]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.should_stop_early()

    def test_should_stop_early_with_flat_lining_metric(self) -> None:
        flatline: List[Dict[str, float]] = [{"acc": 0.2}] * 6
        tracker: Any = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+acc",
        )._metric_tracker
        for m in flatline:
            tracker.add_metrics(m)
        assert tracker.should_stop_early

        tracker = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-acc",
        )._metric_tracker
        for m in flatline:
            tracker.add_metrics(m)
        assert tracker.should_stop_early

    def test_should_stop_early_with_decreasing_metric(self) -> None:
        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-acc",
        )
        tracker: Any = new_trainer._metric_tracker

        new_tracker: Any = copy.deepcopy(tracker)
        for acc in [0.02, 0.3, 0.2, 0.1, 0.4, 0.4]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.2, 0.1, 0.4, 0.5]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        for acc in [0.1, 0.3, 0.2, 0.1, 0.4, 0.5]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.should_stop_early()

    def test_should_stop_early_with_early_stopping_disabled(self) -> None:
        # Increasing metric
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=100,
            patience=None,
            validation_metric="+acc",
        )
        tracker: Any = trainer._metric_tracker
        for m in [{"acc": float(i)} for i in reversed(range(20))]:
            tracker.add_metrics(m)
        assert not tracker.should_stop_early()

        # Decreasing metric
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=100,
            patience=None,
            validation_metric="-acc",
        )
        tracker = trainer._metric_tracker
        for m in [{"acc": float(i)} for i in range(20)]:
            tracker.add_metrics(m)
        assert not tracker.should_stop_early()

    def test_should_stop_early_with_invalid_patience(self) -> None:
        for patience in [0, -1, -2, 1.5, "None"]:
            with pytest.raises(
                ConfigurationError,
                match='.* is an invalid value for "patience": '
                "it must be a positive integer or None "
                "\\(if you want to disable early stopping\\)",
            ):
                GradientDescentTrainer(
                    self.model,
                    self.optimizer,
                    self.data_loader,
                    validation_data_loader=self.validation_data_loader,
                    num_epochs=100,
                    patience=patience,
                    validation_metric="+acc",
                )

    def test_trainer_can_run_and_resume_with_momentum_scheduler(self) -> None:
        scheduler: MomentumScheduler = MomentumScheduler.from_params(
            optimizer=self.optimizer,
            params=Params({"type": "inverted_triangular", "cool_down": 2, "warm_up": 2}),
        )
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            momentum_scheduler=scheduler,
            validation_metric="-loss",
            validation_data_loader=self.validation_data_loader,
            num_epochs=4,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        trainer.train()

        new_scheduler: MomentumScheduler = MomentumScheduler.from_params(
            optimizer=self.optimizer,
            params=Params({"type": "inverted_triangular", "cool_down": 2, "warm_up": 2}),
        )
        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            momentum_scheduler=new_scheduler,
            validation_metric="-loss",
            validation_data_loader=self.validation_data_loader,
            num_epochs=6,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        new_trainer._maybe_restore_checkpoint()
        new_trainer._start_after_epochs_completed = 4
        assert new_trainer._momentum_scheduler.last_epoch == 3
        new_trainer.train()

    def test_trainer_can_run_with_lr_scheduler(self) -> None:
        lr_scheduler: ExponentialLearningRateScheduler = ExponentialLearningRateScheduler(self.optimizer, gamma=0.5)
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            learning_rate_scheduler=lr_scheduler,
            validation_metric="-loss",
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
        )
        trainer.train()

    def test_trainer_sends_metric_to_lr_scheduler(self) -> None:
        from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler

        class RecordMetricLearningRateScheduler(ReduceOnPlateauLearningRateScheduler):
            def __init__(self, optimizer: Optimizer) -> None:
                super(RecordMetricLearningRateScheduler, self).__init__(optimizer)
                self.recordings: List[float] = []

            def step(self, metric: float = None) -> None:
                self.recordings.append(metric)
                super().step(metric)

        lr_scheduler: RecordMetricLearningRateScheduler = RecordMetricLearningRateScheduler(self.optimizer)
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            learning_rate_scheduler=lr_scheduler,
            validation_metric="-loss",
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
        )
        trainer.train()

        assert all([value != 0 for value in lr_scheduler.recordings])

    def test_trainer_can_resume_with_lr_scheduler(self) -> None:
        lr_scheduler: CosineWithRestarts = CosineWithRestarts(self.optimizer, t_initial=5)
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            learning_rate_scheduler=lr_scheduler,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        trainer.train()

        new_lr_scheduler: CosineWithRestarts = CosineWithRestarts(self.optimizer, t_initial=5)
        new_trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            learning_rate_scheduler=new_lr_scheduler,
            validation_data_loader=self.validation_data_loader,
            num_epochs=4,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        new_trainer._maybe_restore_checkpoint()
        assert new_trainer._start_after_epochs_completed == 2
        assert new_trainer._learning_rate_scheduler.last_epoch == 1
        new_trainer.train()

    def test_trainer_raises_on_model_with_no_loss_key(self) -> None:
        class FakeModel(Model):
            def forward(self, **kwargs: Any) -> Dict[str, Any]:
                return {}

        with pytest.raises(RuntimeError):
            trainer: GradientDescentTrainer = GradientDescentTrainer(
                FakeModel(None),
                self.optimizer,
                self.data_loader,
                num_epochs=2,
                serialization_dir=self.TEST_DIR,
            )
            trainer.train()

    def test_trainer_can_log_histograms(self) -> None:
        # enable activation logging
        for module in self.model.modules():
            module.should_log_activations = True

        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            callbacks=[
                TensorBoardCallback(
                    serialization_dir=self.TEST_DIR,
                    distribution_interval=2,
                )
            ],
        )
        trainer.train()

    def test_trainer_respects_num_serialized_models_to_keep(self) -> None:
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=5,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(serialization_dir=self.TEST_DIR, keep_most_recent_by_count=3),
        )
        trainer.train()

        # Now check the serialized files
        expected: List[Tuple[int, int]] = [(3, 0), (4, 0), (5, 0)]

        file_names: List[str] = glob.glob(os.path.join(self.TEST_DIR, "model_state_e*_b*"))
        epochs: List[Tuple[int, int]] = [Checkpointer._parse_model_state_path(fname) for fname in file_names]
        assert sorted(epochs) == expected

        file_names = glob.glob(os.path.join(self.TEST_DIR, "training_state_e*_b*"))
        epochs = [Checkpointer._parse_training_state_path(fname) for fname in file_names]
        assert sorted(epochs) == expected

    def test_trainer_saves_metrics_every_epoch(self) -> None:
        trainer: GradientDescentTrainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=5,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(serialization_dir=self.TEST_DIR, keep_most_recent_by_count=3),
        )
        trainer.train()

        for epoch in range(5):
            epoch_file: str = self.TEST_DIR / f"metrics_epoch_{epoch}.json"
            assert epoch_file.exists()
            metrics: Dict[str, Any] = json.load(open(epoch_file))
            assert "validation_loss" in metrics
            assert "best_validation_loss" in metrics
            assert metrics.get("epoch") == epoch

    def test_trainer_respects_