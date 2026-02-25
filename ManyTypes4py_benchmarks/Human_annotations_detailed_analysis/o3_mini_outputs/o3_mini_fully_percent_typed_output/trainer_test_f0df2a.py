#!/usr/bin/env python
import copy
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Iterator, Tuple

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

    def _read(self, file_path: str) -> Iterator[Instance]:
        for i in range(self.total_instances):
            yield self.text_to_instance(i, "label")

    def text_to_instance(self, index: int, field_type: str) -> Instance:
        field = TextField(
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
        self.lin = torch.nn.Linear(1, 2)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        out = kwargs["label"].sum().unsqueeze(-1)
        out = out.type(torch.FloatTensor)
        out = self.lin(out)
        loss = out.sum()
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
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
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
            if param.grad is not None:
                param.grad.data.zero_()
        return True


class TestTrainer(TrainerTestBase):
    def test_trainer_can_run(self) -> None:
        trainer = GradientDescentTrainer(
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
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=2,
            validation_data_loader=self.validation_data_loader,
            callbacks=[ZeroGradientsBackwardCallback(serialization_dir=self.TEST_DIR)],
        )
        trainer.train()
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
                    param.grad = torch.ones_like(param.grad, device=param.grad.device)  # type: ignore
                return True

        with pytest.raises(OnBackwardException):
            trainer = GradientDescentTrainer(
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
        trainer = GradientDescentTrainer(
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
        trainer = GradientDescentTrainer(
            self.model, self.optimizer, self.data_loader, num_epochs=2, cuda_device=0
        )
        metrics = trainer.train()
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
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader_lazy,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
        epoch: int = metrics["epoch"]
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * 2

    def test_data_loader_lazy_epoch_size_correct_custom_epoch_size(self) -> None:
        self.data_loader_lazy.batches_per_epoch = 3
        num_epochs: int = 3
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader_lazy,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
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
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader_equal_epoch,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
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
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader_larger_epoch,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
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
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader_smaller_epoch,
            validation_data_loader=self.validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
        )
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
        epoch: int = metrics["epoch"]
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * batches_per_epoch

    def test_trainer_can_resume_training(self) -> None:
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        trainer.train()
        new_trainer = GradientDescentTrainer(
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
        tracker = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None
        new_trainer.train()

    def test_trainer_can_resume_training_for_exponential_moving_average(self) -> None:
        moving_average: ExponentialMovingAverage = ExponentialMovingAverage(self.model.named_parameters())
        trainer = GradientDescentTrainer(
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
        new_trainer = GradientDescentTrainer(
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
        tracker = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None
        new_trainer.train()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_increasing_metric(
        self,
    ) -> None:
        new_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+acc",
        )
        tracker = new_trainer._metric_tracker
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics({"acc": 1})
        assert new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.3]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 13]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.0013]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.is_best_so_far()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_decreasing_metric(
        self,
    ) -> None:
        new_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-acc",
        )
        tracker = new_trainer._metric_tracker
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics({"acc": 1})
        assert new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.3]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.0013]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 13]:
            new_tracker.add_metrics({"acc": acc})

    def test_should_stop_early_with_increasing_metric(self) -> None:
        new_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+acc",
        )
        tracker = new_trainer._metric_tracker
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.5, 0.3, 0.2, 0.1, 0.4, 0.4]:
            new_tracker.add_metrics({"acc": acc})
        assert new_tracker.should_stop_early()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1]:
            new_tracker.add_metrics({"acc": acc})
        assert not new_tracker.should_stop_early()

    def test_should_stop_early_with_flat_lining_metric(self) -> None:
        flatline: List[Dict[str, float]] = [{"acc": 0.2}] * 6
        tracker = GradientDescentTrainer(
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
        new_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-acc",
        )
        tracker = new_trainer._metric_tracker
        new_tracker = copy.deepcopy(tracker)
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
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=100,
            patience=None,
            validation_metric="+acc",
        )
        tracker = trainer._metric_tracker
        for m in [{"acc": float(i)} for i in reversed(range(20))]:
            tracker.add_metrics(m)
        assert not tracker.should_stop_early()
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
        trainer = GradientDescentTrainer(
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
        new_trainer = GradientDescentTrainer(
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
        trainer = GradientDescentTrainer(
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

            def step(self, metric: Optional[float] = None) -> None:
                self.recordings.append(metric)  # type: ignore
                super().step(metric)

        lr_scheduler = RecordMetricLearningRateScheduler(self.optimizer)
        trainer = GradientDescentTrainer(
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
        trainer = GradientDescentTrainer(
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
        new_trainer = GradientDescentTrainer(
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
        class FakeModelNoLoss(Model):
            def forward(self, **kwargs: Any) -> Dict[str, Any]:
                return {}

        with pytest.raises(RuntimeError):
            trainer = GradientDescentTrainer(
                FakeModelNoLoss(None),
                self.optimizer,
                self.data_loader,
                num_epochs=2,
                serialization_dir=self.TEST_DIR,
            )
            trainer.train()

    def test_trainer_can_log_histograms(self) -> None:
        for module in self.model.modules():
            module.should_log_activations = True
        trainer = GradientDescentTrainer(
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
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=5,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(serialization_dir=self.TEST_DIR, keep_most_recent_by_count=3),
        )
        trainer.train()
        expected: List[Tuple[int, int]] = [(3, 0), (4, 0), (5, 0)]
        file_names = glob.glob(os.path.join(self.TEST_DIR, "model_state_e*_b*"))
        epochs: List[Tuple[int, int]] = [Checkpointer._parse_model_state_path(fname) for fname in file_names]
        assert sorted(epochs) == expected
        file_names = glob.glob(os.path.join(self.TEST_DIR, "training_state_e*_b*"))
        epochs = [Checkpointer._parse_training_state_path(fname) for fname in file_names]
        assert sorted(epochs) == expected

    def test_trainer_saves_metrics_every_epoch(self) -> None:
        trainer = GradientDescentTrainer(
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
            epoch_file = self.TEST_DIR / f"metrics_epoch_{epoch}.json"
            assert epoch_file.exists()
            metrics = json.load(open(epoch_file))
            assert "validation_loss" in metrics
            assert "best_validation_loss" in metrics
            assert metrics.get("epoch") == epoch

    def test_trainer_respects_keep_serialized_model_every_num_seconds(self) -> None:
        class SlowDataLoader:
            data_loader: SimpleDataLoader = SimpleDataLoader(self.instances, batch_size=2)
            def __iter__(self) -> Iterator[Any]:
                time.sleep(2.5)
                return iter(self.data_loader)
            def __len__(self) -> int:
                return len(self.data_loader)
            def set_target_device(self, _: Any) -> None:
                pass

        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            SlowDataLoader(),
            num_epochs=6,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(
                save_completed_epochs=False,
                serialization_dir=self.TEST_DIR,
                keep_most_recent_by_count=4,
                save_every_num_seconds=5,
            ),
        )
        trainer.train()
        expected: List[Tuple[int, int]] = [(1, 1), (3, 1), (5, 1)]
        file_names = glob.glob(os.path.join(self.TEST_DIR, "model_state_e*_b*"))
        epochs = [Checkpointer._parse_model_state_path(fname) for fname in file_names]
        assert sorted(epochs) == expected
        file_names = glob.glob(os.path.join(self.TEST_DIR, "training_state_e*_b*"))
        epochs = [Checkpointer._parse_training_state_path(fname) for fname in file_names]
        assert sorted(epochs) == expected

    def test_trainer_can_log_learning_rates_tensorboard(self) -> None:
        data_loader: SimpleDataLoader = SimpleDataLoader(self.instances, 4)
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader,
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            callbacks=[
                TensorBoardCallback(
                    serialization_dir=self.TEST_DIR,
                    summary_interval=2,
                    should_log_learning_rate=True,
                )
            ],
        )
        trainer.train()

    def test_confidence_check_callback(self) -> None:
        model_with_bias: FakeModelForTestingNormalizationBiasVerification = FakeModelForTestingNormalizationBiasVerification(use_bias=True)
        inst: Instance = Instance({"x": TensorField(torch.rand(3, 1, 4))})
        data_loader: SimpleDataLoader = SimpleDataLoader([inst, inst], 2)
        trainer = GradientDescentTrainer(
            model_with_bias,
            self.optimizer,
            data_loader,
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            callbacks=[ConfidenceChecksCallback(serialization_dir=self.TEST_DIR)],
        )
        with pytest.raises(ConfidenceCheckError):
            trainer.train()

    def test_confidence_check_default(self) -> None:
        model_with_bias: FakeModelForTestingNormalizationBiasVerification = FakeModelForTestingNormalizationBiasVerification(use_bias=True)
        inst: Instance = Instance({"x": TensorField(torch.rand(3, 1, 4))})
        data_loader: SimpleDataLoader = SimpleDataLoader([inst, inst], 2)
        trainer = GradientDescentTrainer.from_partial_objects(
            model_with_bias,
            serialization_dir=self.TEST_DIR,
            data_loader=data_loader,
            num_epochs=1,
        )
        with pytest.raises(ConfidenceCheckError):
            trainer.train()
        trainer = GradientDescentTrainer.from_partial_objects(
            model_with_bias,
            serialization_dir=self.TEST_DIR,
            data_loader=data_loader,
            num_epochs=1,
            run_confidence_checks=False,
        )
        trainer.train()

    @pytest.mark.parametrize("checkpoint_to_keep", range(20))
    def test_trainer_restores_and_makes_same_results(self, checkpoint_to_keep: int) -> None:
        batch_size: int = 2
        data_loader: SimpleDataLoader = SimpleDataLoader(self.instances, batch_size)
        num_epochs: int = 10
        num_batches: int = len(self.instances) // batch_size
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            data_loader,
            validation_data_loader=data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(
                serialization_dir=self.TEST_DIR,
                save_every_num_seconds=0.0001,
                keep_most_recent_by_count=20,
            ),
        )
        original_metrics: Dict[str, Any] = trainer.train()
        file_names = glob.glob(os.path.join(self.TEST_DIR, "model_state_e*_b*"))
        checkpoints: List[Tuple[int, int]] = [Checkpointer._parse_model_state_path(fname) for fname in file_names]
        checkpoints.sort()
        expected: List[Tuple[int, int]] = [(e, b) for e in range(num_epochs) for b in range(num_batches + 1)]
        del expected[0]
        expected.append((num_epochs, 0))
        expected = expected[-20:]
        assert checkpoints == expected
        for i, checkpoint in enumerate(checkpoints):
            if i != checkpoint_to_keep:
                os.remove(trainer._checkpointer._model_state_path(*checkpoint))
                os.remove(trainer._checkpointer._training_state_path(*checkpoint))
        os.remove(os.path.join(self.TEST_DIR, "best.th"))
        restored_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=data_loader,
            num_epochs=num_epochs,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(
                serialization_dir=self.TEST_DIR,
                save_every_num_seconds=0.0001,
                keep_most_recent_by_count=10,
            ),
        )
        restored_metrics: Dict[str, Any] = restored_trainer.train()
        assert original_metrics["best_validation_loss"] == restored_metrics["best_validation_loss"]

    def test_trainer_saves_and_loads_best_validation_metrics_correctly_1(self) -> None:
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="-loss",
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        trainer.train()
        _ = trainer._maybe_restore_checkpoint()
        best_epoch_1: int = trainer._metric_tracker.best_epoch  # type: ignore
        best_validation_metrics_epoch_1: Dict[str, Any] = trainer._metric_tracker.best_epoch_metrics  # type: ignore
        assert isinstance(best_validation_metrics_epoch_1, dict)
        assert "loss" in best_validation_metrics_epoch_1
        restore_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="-loss",
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        restore_trainer.train()
        _ = restore_trainer._maybe_restore_checkpoint()
        best_epoch_2: int = restore_trainer._metric_tracker.best_epoch  # type: ignore
        best_validation_metrics_epoch_2: Dict[str, Any] = restore_trainer._metric_tracker.best_epoch_metrics  # type: ignore
        assert best_epoch_1 == 0 and best_epoch_2 == 1
        assert best_validation_metrics_epoch_2 != best_validation_metrics_epoch_1

    def test_trainer_saves_and_loads_best_validation_metrics_correctly_2(self) -> None:
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        trainer.train()
        _ = trainer._maybe_restore_checkpoint()
        best_epoch_1: int = trainer._metric_tracker.best_epoch  # type: ignore
        best_validation_metrics_epoch_1: Dict[str, Any] = trainer._metric_tracker.best_epoch_metrics  # type: ignore
        assert isinstance(best_validation_metrics_epoch_1, dict)
        assert "loss" in best_validation_metrics_epoch_1
        restore_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        restore_trainer.train()
        _ = restore_trainer._maybe_restore_checkpoint()
        best_epoch_2: int = restore_trainer._metric_tracker.best_epoch  # type: ignore
        best_validation_metrics_epoch_2: Dict[str, Any] = restore_trainer._metric_tracker.best_epoch_metrics  # type: ignore
        assert best_epoch_1 == best_epoch_2 == 0
        assert best_validation_metrics_epoch_2 == best_validation_metrics_epoch_1

    def test_restored_training_returns_best_epoch_metrics_even_if_no_better_epoch_is_found_after_restoring(self) -> None:
        original_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        training_metrics: Dict[str, Any] = original_trainer.train()
        restored_trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            checkpointer=Checkpointer(self.TEST_DIR),
        )
        restored_metrics: Dict[str, Any] = restored_trainer.train()
        assert "best_validation_loss" in restored_metrics
        assert "best_validation_accuracy" in restored_metrics
        assert "best_validation_accuracy3" in restored_metrics
        assert "best_epoch" in restored_metrics
        assert training_metrics["best_validation_loss"] == restored_metrics["best_validation_loss"]
        assert training_metrics["best_epoch"] == 0
        assert training_metrics["validation_loss"] > restored_metrics["validation_loss"]

    def test_trainer_can_run_gradient_accumulation(self) -> None:
        instances: List[Instance] = list(self.instances)
        steps_to_accumulate: int = 2
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
            num_gradient_accumulation_steps=steps_to_accumulate,
        )
        assert trainer._num_gradient_accumulation_steps == steps_to_accumulate
        trainer.train()
        num_batches_trained_per_epoch: int = trainer._total_batches_completed // trainer._epochs_completed
        num_batches_expected: int = math.ceil(
            math.ceil(len(instances) / self.data_loader.batch_size) / steps_to_accumulate
        )
        assert num_batches_trained_per_epoch == num_batches_expected

    def test_track_epoch_callback(self) -> None:
        num_epochs: int = 4
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=num_epochs,
            validation_data_loader=self.validation_data_loader,
            callbacks=[TrackEpochCallback(serialization_dir=self.TEST_DIR)],
        )
        trainer.train()
        assert trainer.model.epoch == num_epochs

    def test_trainer_callback_is_called_everywhere(self) -> None:
        class FakeTrainerCallback(TrainerCallback):
            def on_start(
                self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs: Any
            ) -> None:
                if not hasattr(trainer, "start_callback_is_fired_first"):
                    trainer.start_callback_is_fired_first = True

            def on_batch(
                self,
                trainer: "GradientDescentTrainer",
                batch_inputs: List[TensorDict],
                batch_outputs: List[Dict[str, Any]],
                batch_metrics: Dict[str, Any],
                epoch: int,
                batch_number: int,
                is_training: bool,
                is_primary: bool = True,
                batch_grad_norm: Optional[float] = None,
                **kwargs: Any,
            ) -> None:
                if not hasattr(trainer, "start_callback_is_fired_first"):
                    trainer.start_callback_is_fired_first = False
                if not hasattr(trainer, "batch_callback_calls"):
                    trainer.batch_callback_calls = []
                trainer.batch_callback_calls.append((epoch, batch_number, is_training))

            def on_epoch(
                self,
                trainer: "GradientDescentTrainer",
                metrics: Dict[str, Any],
                epoch: int,
                is_primary: bool = True,
                **kwargs: Any,
            ) -> None:
                if not hasattr(trainer, "start_callback_is_fired_first"):
                    trainer.start_callback_is_fired_first = False
                if not hasattr(trainer, "epoch_callback_calls"):
                    trainer.epoch_callback_calls = []
                trainer.epoch_callback_calls.append(epoch)

            def on_end(
                self,
                trainer: "GradientDescentTrainer",
                metrics: Optional[Dict[str, Any]] = None,
                epoch: Optional[int] = None,
                is_primary: bool = True,
                **kwargs: Any,
            ) -> None:
                if not hasattr(trainer, "start_callback_is_fired_first"):
                    trainer.start_callback_is_fired_first = False
                if not hasattr(trainer, "end_callback_calls"):
                    trainer.end_callback_calls = []
                trainer.end_callback_calls.append(epoch)
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=2,
            validation_data_loader=self.validation_data_loader,
            callbacks=[FakeTrainerCallback(serialization_dir=self.TEST_DIR)],
        )
        trainer.train()
        expected_batch_calls: List[Tuple[int, int, bool]] = [
            (epoch, batch_number + 1, is_train)
            for epoch in range(2)
            for is_train in (True, False)
            for batch_number in range(len(self.instances) // 2)
        ]
        expected_epoch_calls: List[int] = [epoch for epoch in range(0, 2)]
        expected_end_calls: List[Optional[int]] = [1]
        assert trainer.start_callback_is_fired_first
        assert trainer.batch_callback_calls == expected_batch_calls
        assert trainer.epoch_callback_calls == expected_epoch_calls
        assert trainer.end_callback_calls == expected_end_calls

    def test_total_loss_is_average_of_batch_loss(self) -> None:
        batches_per_epoch: int = 3
        self.data_loader_lazy.batches_per_epoch = batches_per_epoch

        class FakeOnBatchCallback(TrainerCallback):
            def on_batch(
                self,
                trainer: "GradientDescentTrainer",
                batch_inputs: List[TensorDict],
                batch_outputs: List[Dict[str, Any]],
                batch_metrics: Dict[str, Any],
                epoch: int,
                batch_number: int,
                is_training: bool,
                is_primary: bool = True,
                batch_grad_norm: Optional[float] = None,
                **kwargs: Any,
            ) -> None:
                if not hasattr(trainer, "batch_losses"):
                    trainer.batch_losses = []
                trainer.batch_losses.append(batch_outputs[0]["loss"].item())
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader_lazy,
            num_epochs=1,
            callbacks=[FakeOnBatchCallback(serialization_dir=self.TEST_DIR)],
        )
        metrics = trainer.train()
        assert metrics["training_loss"] == float(sum(trainer.batch_losses) / batches_per_epoch)

    def test_trainer_can_log_batch_inputs(self) -> None:
        total_instances: int = 1000
        batch_size: int = 25
        reader: FakeDatasetReader = FakeDatasetReader(total_instances, batch_size)
        data_loader: SimpleDataLoader = SimpleDataLoader.from_dataset_reader(
            reader, "fake_path", batch_size=batch_size
        )
        instances: List[Instance] = list(data_loader.iter_instances())
        vocab: Vocabulary = Vocabulary.from_instances(instances)
        data_loader.index_with(vocab)
        model: FakeModel = FakeModel(vocab)
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)
        trainer = GradientDescentTrainer(
            model,
            optimizer,
            data_loader,
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            callbacks=[
                TensorBoardCallback(
                    serialization_dir=self.TEST_DIR,
                    distribution_interval=2,
                )
            ],
        )
        trainer.train()

    def test_console_log_callback(self) -> None:
        total_instances: int = 1000
        batch_size: int = 25
        reader: FakeDatasetReader = FakeDatasetReader(total_instances, batch_size)
        data_loader: SimpleDataLoader = SimpleDataLoader.from_dataset_reader(
            reader, "fake_path", batch_size=batch_size
        )
        instances: List[Instance] = list(data_loader.iter_instances())
        vocab: Vocabulary = Vocabulary.from_instances(instances)
        data_loader.index_with(vocab)
        model: FakeModel = FakeModel(vocab)
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)
        trainer = GradientDescentTrainer(
            model,
            optimizer,
            data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            callbacks=[
                ConsoleLoggerCallback.from_params(
                    Params({"should_log_inputs": True}),
                    serialization_dir=self.TEST_DIR,
                )
            ],
        )
        trainer.train()

    def test_should_validate_callback(self) -> None:
        total_instances: int = 1000
        batch_size: int = 25
        reader: FakeDatasetReader = FakeDatasetReader(total_instances, batch_size)
        data_loader: SimpleDataLoader = SimpleDataLoader.from_dataset_reader(
            reader, "fake_path", batch_size=batch_size
        )
        instances: List[Instance] = list(data_loader.iter_instances())
        vocab: Vocabulary = Vocabulary.from_instances(instances)
        data_loader.index_with(vocab)
        model: FakeModel = FakeModel(vocab)
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)
        callback: ShouldValidateCallback = ShouldValidateCallback.from_params(
            Params({"validation_start": 4, "validation_interval": 2}),
            serialization_dir=self.TEST_DIR,
        )
        trainer = GradientDescentTrainer(
            model,
            optimizer,
            data_loader,
            num_epochs=6,
            serialization_dir=self.TEST_DIR,
            callbacks=[callback],
        )
        trainer.train()
        callback.on_start(trainer)
        assert not trainer._should_validate_this_epoch
        callback.on_epoch(trainer, metrics={}, epoch=1)
        assert not trainer._should_validate_this_epoch
        callback.on_epoch(trainer, metrics={}, epoch=2)
        assert not trainer._should_validate_this_epoch
        callback.on_epoch(trainer, metrics={}, epoch=3)
        assert trainer._should_validate_this_epoch
        callback.on_end(trainer)
        assert trainer._should_validate_this_epoch


@requires_gpu
class TestAmpTrainer(TrainerTestBase):
    @pytest.mark.parametrize(
        "grad_norm, num_gradient_accumulation_steps", [(None, 1), (1.0, 1), (1.0, 2)]
    )
    def test_trainer_can_run_amp(self, grad_norm: Optional[float], num_gradient_accumulation_steps: int) -> None:
        self.model.cuda()
        trainer = GradientDescentTrainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=2,
            cuda_device=0,
            use_amp=True,
            grad_norm=True,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
        )
        _ = trainer.train()


class TestSparseClipGrad(AllenNlpTestCase):
    def test_sparse_clip_grad(self) -> None:
        embedding: torch.nn.Embedding = torch.nn.Embedding(100, 16, sparse=True)
        embedding.zero_grad()
        ids = (torch.rand(17) * 100).long()
        ids[:5] = 5
        loss = embedding(ids).sum()
        loss.backward()
        assert embedding.weight.grad.is_sparse
        _ = clip_grad_norm_([embedding.weight], 1.5)
        grad = embedding.weight.grad.coalesce()
        assert grad._values().norm(2.0).item() == pytest.approx(1.5, rel=1e-4)