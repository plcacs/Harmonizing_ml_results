import copy
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Set, TypeVar, Callable, Iterable, Sequence
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
from allennlp.training import GradientDescentTrainer, Checkpointer
from allennlp.training.callbacks import TrainerCallback, TrackEpochCallback, TensorBoardCallback, ConfidenceChecksCallback, ConsoleLoggerCallback, OnBackwardException, ShouldValidateCallback
from allennlp.training.callbacks.confidence_checks import ConfidenceCheckError
from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from allennlp.training.learning_rate_schedulers import ExponentialLearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import ExponentialMovingAverage
from allennlp.data.fields import TextField, IndexField, MetadataField, LabelField, MultiLabelField, SpanField, FlagField, AdjacencyField, TensorField
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing.confidence_check_test import FakeModelForTestingNormalizationBiasVerification

class FakeDatasetReader(DatasetReader):
    def __init__(self, total_instances: int, batch_size: int) -> None:
        super().__init__()
        self.total_instances = total_instances
        self.batch_size = batch_size

    def _read(self, file_path: str) -> Iterable[Instance]:
        for i in range(self.total_instances):
            yield self.text_to_instance(i, 'label')

    def text_to_instance(self, index: int, field_type: str) -> Instance:
        field = TextField([Token(t) for t in ['The', 'number', 'is', str(index), '.']], token_indexers={'words': SingleIdTokenIndexer('words')})
        return Instance({'text': field, 'label': LabelField(index, skip_indexing=True), 'flag': FlagField(23), 'index': IndexField(index % self.batch_size, field), 'metadata': MetadataField({'some_key': 'This will not be logged as a histogram.'}), 'adjacency': AdjacencyField([(0, 1), (1, 2)], field), 'multilabel': MultiLabelField(['l1', 'l2']), 'span': SpanField(2, 3, field), 'tensor': TensorField(torch.randn(2, 3))})

class FakeModel(Model):
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.lin = torch.nn.Linear(1, 2)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        out = kwargs['label'].sum().unsqueeze(-1)
        out = out.type(torch.FloatTensor)
        out = self.lin(out)
        loss = out.sum()
        return {'loss': loss}

class TrainerTestBase(AllenNlpTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_path = str(self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv')
        self.reader = SequenceTaggingDatasetReader(max_instances=4)
        self.data_loader = MultiProcessDataLoader(self.reader, self.data_path, batch_size=2)
        self.data_loader_lazy = MultiProcessDataLoader(self.reader, self.data_path, batch_size=2, max_instances_in_memory=10)
        self.instances = list(self.data_loader.iter_instances())
        self.vocab = Vocabulary.from_instances(self.instances)
        self.data_loader.index_with(self.vocab)
        self.data_loader_lazy.index_with(self.vocab)
        self.model_params = Params({'text_field_embedder': {'token_embedders': {'tokens': {'type': 'embedding', 'embedding_dim': 5}}}, 'encoder': {'type': 'lstm', 'input_size': 5, 'hidden_size': 7, 'num_layers': 2}})
        self.model = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
        self.validation_data_loader = MultiProcessDataLoader(self.reader, self.data_path, batch_size=2)
        self.validation_data_loader.index_with(self.vocab)

class ZeroGradientsBackwardCallback(TrainerCallback):
    def on_backward(self, trainer: GradientDescentTrainer, batch_outputs: Dict[str, torch.Tensor], backward_called: bool, **kwargs: Any) -> bool:
        if backward_called:
            raise OnBackwardException()
        batch_outputs['loss'].backward()
        for param in trainer.model.parameters():
            param.grad.data.zero_()
        return True

class TestTrainer(TrainerTestBase):
    def test_trainer_can_run(self) -> None:
        trainer = GradientDescentTrainer(model=self.model, optimizer=self.optimizer, data_loader=self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=2)
        metrics = trainer.train()
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)
        trainer = GradientDescentTrainer(model=self.model, optimizer=self.optimizer, data_loader=self.data_loader, validation_data_loader=self.validation_data_loader, validation_metric='+loss', num_epochs=2)
        metrics = trainer.train()
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)
        assert 'peak_worker_0_memory_MB' in metrics
        assert isinstance(metrics['peak_worker_0_memory_MB'], float)
        assert metrics['peak_worker_0_memory_MB'] > 0

    def test_train_zero_gradients(self) -> None:
        weights: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.clone()
        trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, num_epochs=2, validation_data_loader=self.validation_data_loader, callbacks=[ZeroGradientsBackwardCallback(serialization_dir=self.TEST_DIR)])
        trainer.train()
        for name, param in self.model.named_parameters():
            assert torch.equal(weights[name], param.data)

    def test_two_backward_callbacks(self) -> None:
        class SecondBackwardCallback(TrainerCallback):
            def on_backward(self, trainer: GradientDescentTrainer, batch_outputs: Dict[str, torch.Tensor], backward_called: bool, **kwargs: Any) -> bool:
                if backward_called:
                    raise OnBackwardException()
                batch_outputs['loss'].backward()
                for param in trainer.model.parameters():
                    param.grad = torch.ones_like(param.grad, device=param.grad.device)
                return True
        with pytest.raises(OnBackwardException):
            trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, num_epochs=2, validation_data_loader=self.validation_data_loader, callbacks=[ZeroGradientsBackwardCallback(serialization_dir=self.TEST_DIR), SecondBackwardCallback(serialization_dir=self.TEST_DIR)])
            trainer.train()

    def test_trainer_can_run_exponential_moving_average(self) -> None:
        moving_average = ExponentialMovingAverage(self.model.named_parameters(), decay=0.9999)
        trainer = GradientDescentTrainer(model=self.model, optimizer=self.optimizer, data_loader=self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=2, moving_average=moving_average)
        trainer.train()

    @requires_gpu
    def test_trainer_can_run_cuda(self) -> None:
        self.model.cuda()
        trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, num_epochs=2, cuda_device=0)
        metrics = trainer.train()
        assert 'peak_worker_0_memory_MB' in metrics
        assert isinstance(metrics['peak_worker_0_memory_MB'], float)
        assert metrics['peak_worker_0_memory_MB'] > 0
        assert 'peak_gpu_0_memory_MB' in metrics
        assert isinstance(metrics['peak_gpu_0_memory_MB'], float)

    @requires_multi_gpu
    def test_passing_trainer_multiple_gpus_raises_error(self) -> None:
        self.model.cuda()
        with pytest.raises(ConfigurationError):
            GradientDescentTrainer(self.model, self.optimizer, self.data_loader, num_epochs=2, cuda_device=[0, 1])

    def test_data_loader_lazy_epoch_size_correct(self) -> None:
        num_epochs = 3
        trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader_lazy, validation_data_loader=self.validation_data_loader, num_epochs=num_epochs, serialization_dir=self.TEST_DIR)
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
        epoch = metrics['epoch']
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * 2

    def test_data_loader_lazy_epoch_size_correct_custom_epoch_size(self) -> None:
        self.data_loader_lazy.batches_per_epoch = 3
        num_epochs = 3
        trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader_lazy, validation_data_loader=self.validation_data_loader, num_epochs=num_epochs, serialization_dir=self.TEST_DIR)
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
        epoch = metrics['epoch']
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * 3

    def test_trainer_respects_epoch_size_equals_total(self) -> None:
        batches_per_epoch = 4
        num_epochs = 3
        data_loader_equal_epoch = SimpleDataLoader(self.instances, 2, batches_per_epoch=batches_per_epoch)
        trainer = GradientDescentTrainer(self.model, self.optimizer, data_loader_equal_epoch, validation_data_loader=self.validation_data_loader, num_epochs=num_epochs, serialization_dir=self.TEST_DIR)
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
        epoch = metrics['epoch']
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * batches_per_epoch

    def test_trainer_respects_epoch_size_larger_tnan_total(self) -> None:
        batches_per_epoch = 7
        num_epochs = 3
        data_loader_larger_epoch = SimpleDataLoader(self.instances, 2, batches_per_epoch=batches_per_epoch)
        trainer = GradientDescentTrainer(self.model, self.optimizer, data_loader_larger_epoch, validation_data_loader=self.validation_data_loader, num_epochs=num_epochs, serialization_dir=self.TEST_DIR)
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
        epoch = metrics['epoch']
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * batches_per_epoch

    def test_trainer_respects_epoch_size_smaller_tnan_total(self) -> None:
        batches_per_epoch = 1
        num_epochs = 2
        data_loader_smaller_epoch = SimpleDataLoader(self.instances, 2, batches_per_epoch=batches_per_epoch)
        trainer = GradientDescentTrainer(self.model, self.optimizer, data_loader_smaller_epoch, validation_data_loader=self.validation_data_loader, num_epochs=num_epochs, serialization_dir=self.TEST_DIR)
        assert trainer._total_batches_completed == 0
        metrics = trainer.train()
        epoch = metrics['epoch']
        assert epoch == num_epochs - 1
        assert trainer._total_batches_completed == num_epochs * batches_per_epoch

    def test_trainer_can_resume_training(self) -> None:
        trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=1, serialization_dir=self.TEST_DIR, checkpointer=Checkpointer(self.TEST_DIR))
        trainer.train()
        new_trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=3, serialization_dir=self.TEST_DIR, checkpointer=Checkpointer(self.TEST_DIR))
        new_trainer._maybe_restore_checkpoint()
        assert new_trainer._start_after_epochs_completed == 1
        tracker = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None
        new_trainer.train()

    def test_trainer_can_resume_training_for_exponential_moving_average(self) -> None:
        moving_average = ExponentialMovingAverage(self.model.named_parameters())
        trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=1, serialization_dir=self.TEST_DIR, moving_average=moving_average, checkpointer=Checkpointer(self.TEST_DIR))
        trainer.train()
        new_moving_average = ExponentialMovingAverage(self.model.named_parameters())
        new_trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=3, serialization_dir=self.TEST_DIR, moving_average=new_moving_average, checkpointer=Checkpointer(self.TEST_DIR))
        new_trainer._maybe_restore_checkpoint()
        assert new_trainer._start_after_epochs_completed == 1
        tracker = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None
        new_trainer.train()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_increasing_metric(self) -> None:
        new_trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=3, serialization_dir=self.TEST_DIR, patience=5, validation_metric='+acc')
        tracker = new_trainer._metric_tracker
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics({'acc': 1})
        assert new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.3]:
            new_tracker.add_metrics({'acc': acc})
        assert not new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 13]:
            new_tracker.add_metrics({'acc': acc})
        assert new_tracker.is_best_so_far()
        new_tracker = copy.deepcopy(tracker)
        for acc in [0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.0013]:
            new_tracker.add_metrics({'acc': acc})
        assert not new_tracker.is_best_so_far()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_decreasing_metric(self) -> None:
        new_trainer = GradientDescentTrainer(self.model, self.optimizer, self.data_loader, validation_data_loader=self.validation_data_loader, num_epochs=3, serialization_dir=self.TEST_DIR, patience=5, validation_metric='-acc')
        tracker = new_trainer._metric_tracker
        new_tracker = copy