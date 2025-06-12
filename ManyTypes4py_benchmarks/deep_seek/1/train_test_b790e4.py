import argparse
import copy
import json
import logging
import math
import os
import re
import shutil
from collections import OrderedDict, Counter
from typing import Optional, List, Dict, Any, Set, Callable, Pattern, Match, Union, cast
import pytest
import torch
from allennlp.version import VERSION
from allennlp.commands.train import Train, train_model, train_model_from_args, TrainModel
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, cpu_or_gpu
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import TensorDict
from allennlp.models import load_archive, Model
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import TrainerCallback, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import ExponentialLearningRateScheduler, LearningRateScheduler

SEQUENCE_TAGGING_DATA_PATH: str = str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv')
SEQUENCE_TAGGING_SHARDS_PATH: str = str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'shards' / '*')

@TrainerCallback.register('training_data_logger')
class TrainingDataLoggerOnBatchCallback(TrainerCallback):
    def on_batch(self, 
                trainer: GradientDescentTrainer, 
                batch_inputs: List[Dict[str, Any]], 
                batch_outputs: Dict[str, torch.Tensor], 
                batch_metrics: Dict[str, float], 
                epoch: int, 
                batch_number: int, 
                is_training: bool, 
                is_primary: bool = True, 
                **kwargs: Any) -> None:
        if is_training:
            logger = logging.getLogger(__name__)
            for batch in batch_inputs:
                for metadata in batch['metadata']:
                    logger.info(f"First word from training data: '{metadata['words'][0]}'")

_seen_training_devices: Set[torch.device] = set()

@TrainerCallback.register('training_device_logger')
class TrainingDeviceLoggerOnBatchCallback(TrainerCallback):
    def on_batch(self, 
                trainer: GradientDescentTrainer, 
                batch_inputs: List[Dict[str, Any]], 
                batch_outputs: Dict[str, torch.Tensor], 
                batch_metrics: Dict[str, float], 
                epoch: int, 
                batch_number: int, 
                is_training: bool, 
                is_primary: bool = True, 
                **kwargs: Any) -> None:
        global _seen_training_devices
        for tensor in trainer.model.parameters():
            _seen_training_devices.add(tensor.device)

@TrainerCallback.register('training_primary_check')
class TrainingPrimaryCheckCallback(TrainerCallback):
    def on_start(self, 
                 trainer: GradientDescentTrainer, 
                 is_primary: bool = True, 
                 **kwargs: Any) -> None:
        super().on_start(trainer, is_primary=is_primary, **kwargs)
        if is_primary:
            assert torch.distributed.get_rank() == 0

class TestTrain(AllenNlpTestCase):
    DEFAULT_PARAMS: Params = Params({
        'model': {
            'type': 'simple_tagger',
            'text_field_embedder': {
                'token_embedders': {
                    'tokens': {
                        'type': 'embedding',
                        'embedding_dim': 5
                    }
                }
            },
            'encoder': {
                'type': 'lstm',
                'input_size': 5,
                'hidden_size': 7,
                'num_layers': 2
            }
        },
        'dataset_reader': {'type': 'sequence_tagging'},
        'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
        'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
        'data_loader': {'batch_size': 2},
        'trainer': {
            'num_epochs': 2,
            'optimizer': 'adam'
        }
    })

    def test_train_model(self) -> None:
        params: Callable[[], Params] = lambda: copy.deepcopy(self.DEFAULT_PARAMS)
        serialization_dir: str = os.path.join(self.TEST_DIR, 'test_train_model')
        train_model(params(), serialization_dir=serialization_dir)
        archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'))
        assert archive.meta is not None
        assert archive.meta.version == VERSION
        serialization_dir2: str = os.path.join(self.TEST_DIR, 'empty_directory')
        assert not os.path.exists(serialization_dir2)
        os.makedirs(serialization_dir2)
        train_model(params(), serialization_dir=serialization_dir2)
        serialization_dir3: str = os.path.join(self.TEST_DIR, 'non_empty_directory')
        assert not os.path.exists(serialization_dir3)
        os.makedirs(serialization_dir3)
        with open(os.path.join(serialization_dir3, 'README.md'), 'w') as f:
            f.write('TEST')
        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=serialization_dir3)
        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))
        train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'), recover=True)
        train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'), force=True)
        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'), force=True, recover=True)

    @cpu_or_gpu
    def test_detect_gpu(self) -> None:
        params: Params = copy.deepcopy(self.DEFAULT_PARAMS)
        params['trainer']['callbacks'] = ['training_device_logger']
        global _seen_training_devices
        _seen_training_devices.clear()
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_detect_gpu'))
        assert len(_seen_training_devices) == 1
        seen_training_device: torch.device = next(iter(_seen_training_devices))
        if torch.cuda.device_count() == 0:
            assert seen_training_device.type == 'cpu'
        else:
            assert seen_training_device.type == 'cuda'

    @cpu_or_gpu
    def test_force_gpu(self) -> None:
        params: Params = copy.deepcopy(self.DEFAULT_PARAMS)
        params['trainer']['callbacks'] = ['training_device_logger']
        params['trainer']['cuda_device'] = 0
        global _seen_training_devices
        _seen_training_devices.clear()
        if torch.cuda.device_count() == 0:
            with pytest.raises(ConfigurationError):
                train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_force_gpu'))
        else:
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_force_gpu'))
            assert len(_seen_training_devices) == 1
            seen_training_device: torch.device = next(iter(_seen_training_devices))
            assert seen_training_device.type == 'cuda'

    @cpu_or_gpu
    def test_force_cpu(self) -> None:
        params: Params = copy.deepcopy(self.DEFAULT_PARAMS)
        params['trainer']['callbacks'] = ['training_device_logger']
        params['trainer']['cuda_device'] = -1
        global _seen_training_devices
        _seen_training_devices.clear()
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_force_cpu'))
        assert len(_seen_training_devices) == 1
        seen_training_device: torch.device = next(iter(_seen_training_devices))
        assert seen_training_device.type == 'cpu'

    @cpu_or_gpu
    def test_train_model_distributed(self) -> None:
        devices: List[int] = [0, 1] if torch.cuda.device_count() >= 2 else [-1, -1]
        params: Callable[[], Params] = lambda: Params({
            'model': {
                'type': 'simple_tagger',
                'text_field_embedder': {
                    'token_embedders': {
                        'tokens': {
                            'type': 'embedding',
                            'embedding_dim': 5
                        }
                    }
                },
                'encoder': {
                    'type': 'lstm',
                    'input_size': 5,
                    'hidden_size': 7,
                    'num_layers': 2
                }
            },
            'dataset_reader': {'type': 'sequence_tagging'},
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {'batch_size': 2},
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam',
                'callbacks': ['tests.commands.train_test.TrainingPrimaryCheckCallback']
            },
            'distributed': {'cuda_devices': devices}
        })
        out_dir: str = os.path.join(self.TEST_DIR, 'test_distributed_train')
        train_model(params(), serialization_dir=out_dir)
        serialized_files: List[str] = os.listdir(out_dir)
        assert 'out_worker0.log' in serialized_files
        assert 'out_worker1.log' in serialized_files
        assert 'model.tar.gz' in serialized_files
        assert 'metrics.json' in serialized_files
        with open(os.path.join(out_dir, 'metrics.json')) as f:
            metrics: Dict[str, Any] = json.load(f)
            assert metrics['peak_worker_0_memory_MB'] > 0
            assert metrics['peak_worker_1_memory_MB'] > 0
            if torch.cuda.device_count() >= 2:
                assert metrics['peak_gpu_0_memory_MB'] > 0
                assert metrics['peak_gpu_1_memory_MB'] > 0
        assert load_archive(out_dir).model

    @pytest.mark.parametrize('max_instances', [1, 2, 3, 4, None])
    @pytest.mark.parametrize('grad_acc', [None, 2])
    @pytest.mark.parametrize('batch_size', [1, 2, 3])
    def test_train_model_distributed_with_gradient_accumulation(self, 
                                                              max_instances: Optional[int], 
                                                              grad_acc: Optional[int], 
                                                              batch_size: int) -> None:
        devices: List[int] = [0, 1] if torch.cuda.device_count() >= 2 else [-1, -1]
        params: Callable[[], Params] = lambda: Params({
            'model': {
                'type': 'simple_tagger',
                'text_field_embedder': {
                    'token_embedders': {
                        'tokens': {
                            'type': 'embedding',
                            'embedding_dim': 5
                        }
                    }
                },
                'encoder': {
                    'type': 'lstm',
                    'input_size': 5,
                    'hidden_size': 7,
                    'num_layers': 2
                }
            },
            'dataset_reader': {
                'type': 'sequence_tagging',
                'max_instances': max_instances
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {'batch_size': batch_size},
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam',
                'num_gradient_accumulation_steps': grad_acc
            },
            'distributed': {'cuda_devices': devices}
        })
        out_dir: str = os.path.join(self.TEST_DIR, 'test_distributed_train_with_grad_acc')
        train_model(params(), serialization_dir=out_dir)
        serialized_files: List[str] = os.listdir(out_dir)
        assert 'out_worker0.log' in serialized_files
        assert 'out_worker1.log' in serialized_files
        assert 'model.tar.gz' in serialized_files
        assert 'metrics.json' in serialized_files
        with open(os.path.join(out_dir, 'metrics.json')) as f:
            metrics: Dict[str, Any] = json.load(f)
            assert metrics['peak_worker_0_memory_MB'] > 0
            assert metrics['peak_worker_1_memory_MB'] > 0
            if torch.cuda.device_count() >= 2:
                assert metrics['peak_gpu_0_memory_MB'] > 0
                assert metrics['peak_gpu_1_memory_MB'] > 0
        assert load_archive(out_dir).model

    @cpu_or_gpu
    @pytest.mark.parametrize('max_instances_in_memory', [None, 10])
    def test_train_model_distributed_with_sharded_reader(self, max_instances_in_memory: Optional[int]) -> None:
        devices: List[int] = [0, 1] if torch.cuda.device_count() >= 2 else [-1, -1]
        params: Callable[[], Params] = lambda: Params({
            'model': {
                'type': 'simple_tagger',
                'text_field_embedder': {
                    'token_embedders': {
                        'tokens': {
                            'type': 'embedding',
                            'embedding_dim': 5
                        }
                    }
                },
                'encoder': {
                    'type': 'lstm',
                    'input_size': 5,
                    'hidden_size': 7,
                    'num_layers': 2
                }
            },
            'dataset_reader': {
                'type': 'sharded',
                'base_reader': {'type': 'sequence_tagging'}
            },
            'train_data_path': SEQUENCE_TAGGING_SHARDS_PATH,
            'validation_data_path': SEQUENCE_TAGGING_SHARDS_PATH,
            'data_loader': {
                'batch_size': 1,
                'max_instances_in_memory': max_instances_in_memory
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            },
            'distributed': {'cuda_devices': devices}
        })
        out_dir: str = os.path.join(self.TEST_DIR, 'test_distributed_train')
        train_model(params(), serialization_dir=out_dir)
        serialized_files: List[str] = os.listdir(out_dir)
        assert 'out_worker0.log' in serialized_files
        assert 'out_worker1.log' in serialized_files
        assert 'model.tar.gz' in serialized_files
        archive = load_archive(out_dir)
        assert archive.model
        tokens: List[str] = list(archive.model.vocab._token_to_index['tokens'].keys())
        assert tokens == ['@@PADDING@@', '@@UNKNOWN@@', 'are', '.', 'animals', 'plants', 'vehicles', 'cats', 'dogs', 'snakes', 'birds', 'ferns', 'trees', 'flowers', 'vegetables', 'cars', 'buses', 'planes', 'rockets']
        train_early: str = 'finishing training early!'
        validation_early: str = 'finishing validation early!'
        train_complete: str = 'completed its entire epoch (training).'
        validation_complete: str = 'completed its entire epoch (validation).'
        with open(os.path.join(out_dir, 'out_worker0.log')) as f:
            worker0_log: str = f.read()
            assert train_early in worker0_log
            assert validation_early in worker0_log
            assert train_complete not in worker0_log
            assert validation_complete not in worker0_log
        with open(os.path.join(out_dir, 'out_worker1.log')) as f:
            worker1_log: str = f.read()
            assert train_early not in worker1_log
            assert validation_early not in worker1_log
            assert train_complete in worker1_log
            assert validation_complete in worker1_log

    @cpu_or_gpu
    @pytest.mark.parametrize('max_instances_in_memory', [None, 10])
    def test_train_model_distributed_without_sharded_reader(self, max_instances_in_memory: Optional[int]) -> None:
        devices: List[int] = [0, 1] if torch.cuda.device_count() >= 2 else [-1, -1]
        num_epochs: int = 2
        params: Callable[[], Params] = lambda: Params({
            'model': {
                'type': 'simple_tagger',
                'text_field_embedder': {
                    'token_embedders': {
                        'tokens': {
                            'type': 'embedding',
                            'embedding_dim': 5
                        }
                    }
                },
                'encoder': {
                    'type': 'lstm',
                    'input_size': 5,
                    'hidden_size': 7,
                    'num_layers': 2
                }
            },
            'dataset_reader': {
                'type': 'sequence_tagging',
                'max_instances': 4
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {
                'batch_size': 1,
                'max_instances_in_memory': max_instances_in_memory
            },
            'trainer': {
                'num_epochs': num_epochs,
                'optimizer': 'adam',
                'callbacks': ['tests.commands.train_test.TrainingDataLoggerOnBatchCallback']
            },
            'distributed': {'cuda_devices': devices}
        })
        out_dir: str = os.path.join(self.TEST_DIR, '