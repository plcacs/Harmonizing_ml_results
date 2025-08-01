import argparse
import copy
import json
import logging
import math
import os
import re
import shutil
from collections import OrderedDict, Counter
from typing import Optional, List, Dict, Any, Callable
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

SEQUENCE_TAGGING_DATA_PATH: str = str(
    AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'
)
SEQUENCE_TAGGING_SHARDS_PATH: str = str(
    AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'shards' / '*'
)

TrainerCallbackRegister = Callable[..., None]

@TrainerCallback.register('training_data_logger')
class TrainingDataLoggerOnBatchCallback(TrainerCallback):
    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[Dict[str, Any]],
        batch_outputs: Dict[str, Any],
        batch_metrics: Dict[str, float],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        **kwargs: Any
    ) -> None:
        if is_training:
            logger: logging.Logger = logging.getLogger(__name__)
            for batch in batch_inputs:
                for metadata in batch.get('metadata', []):
                    first_word: str = metadata['words'][0]
                    logger.info(f"First word from training data: '{first_word}'")

_seen_training_devices: set = set()

@TrainerCallback.register('training_device_logger')
class TrainingDeviceLoggerOnBatchCallback(TrainerCallback):
    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[Dict[str, Any]],
        batch_outputs: Dict[str, Any],
        batch_metrics: Dict[str, float],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        **kwargs: Any
    ) -> None:
        global _seen_training_devices
        for tensor in trainer.model.parameters():
            _seen_training_devices.add(tensor.device)

@TrainerCallback.register('training_primary_check')
class TrainingPrimaryCheckCallback(TrainerCallback):
    """
    Makes sure there is only one primary worker.
    """

    def on_start(
        self,
        trainer: GradientDescentTrainer,
        is_primary: bool = True,
        **kwargs: Any
    ) -> None:
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
        'dataset_reader': {
            'type': 'sequence_tagging'
        },
        'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
        'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
        'data_loader': {
            'batch_size': 2
        },
        'trainer': {
            'num_epochs': 2,
            'optimizer': 'adam'
        }
    })

    def test_train_model(self) -> None:
        params: Params = copy.deepcopy(self.DEFAULT_PARAMS)
        serialization_dir: str = os.path.join(self.TEST_DIR, 'test_train_model')
        train_model(params, serialization_dir=serialization_dir)
        archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'))
        assert archive.meta is not None
        assert archive.meta.version == VERSION
        serialization_dir2: str = os.path.join(self.TEST_DIR, 'empty_directory')
        assert not os.path.exists(serialization_dir2)
        os.makedirs(serialization_dir2)
        train_model(params, serialization_dir=serialization_dir2)
        serialization_dir3: str = os.path.join(self.TEST_DIR, 'non_empty_directory')
        assert not os.path.exists(serialization_dir3)
        os.makedirs(serialization_dir3)
        with open(os.path.join(serialization_dir3, 'README.md'), 'w') as f:
            f.write('TEST')
        with pytest.raises(ConfigurationError):
            train_model(params, serialization_dir=serialization_dir3)
        with pytest.raises(ConfigurationError):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'), recover=True)
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'), force=True)
        with pytest.raises(ConfigurationError):
            train_model(
                params,
                serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'),
                force=True,
                recover=True
            )

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
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
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
                'type': 'sequence_tagging'
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam',
                'callbacks': ['tests.commands.train_test.TrainingPrimaryCheckCallback']
            },
            'distributed': {
                'cuda_devices': devices
            }
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
    def test_train_model_distributed_with_gradient_accumulation(
        self,
        max_instances: Optional[int],
        grad_acc: Optional[int],
        batch_size: int
    ) -> None:
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
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
            'data_loader': {
                'batch_size': batch_size
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam',
                'num_gradient_accumulation_steps': grad_acc
            },
            'distributed': {
                'cuda_devices': devices
            }
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
    def test_train_model_distributed_with_sharded_reader(
        self,
        max_instances_in_memory: Optional[int]
    ) -> None:
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
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
                'base_reader': {
                    'type': 'sequence_tagging'
                }
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
            'distributed': {
                'cuda_devices': devices
            }
        })
        out_dir: str = os.path.join(self.TEST_DIR, 'test_distributed_train')
        train_model(params(), serialization_dir=out_dir)
        serialized_files: List[str] = os.listdir(out_dir)
        assert 'out_worker0.log' in serialized_files
        assert 'out_worker1.log' in serialized_files
        assert 'model.tar.gz' in serialized_files
        assert 'metrics.json' in serialized_files
        archive = load_archive(out_dir)
        assert archive.model
        tokens: set = set(archive.model.vocab._token_to_index['tokens'].keys())
        assert tokens == {
            '@@PADDING@@', '@@UNKNOWN@@', 'are', '.', 'animals', 'plants',
            'vehicles', 'cats', 'dogs', 'snakes', 'birds', 'ferns',
            'trees', 'flowers', 'vegetables', 'cars', 'buses', 'planes', 'rockets'
        }
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
    def test_train_model_distributed_without_sharded_reader(
        self,
        max_instances_in_memory: Optional[int]
    ) -> None:
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
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
            'distributed': {
                'cuda_devices': devices
            }
        })
        out_dir: str = os.path.join(self.TEST_DIR, 'test_distributed_train')
        train_model(params(), serialization_dir=out_dir)
        serialized_files: List[str] = os.listdir(out_dir)
        assert 'out_worker0.log' in serialized_files
        assert 'out_worker1.log' in serialized_files
        assert 'model.tar.gz' in serialized_files
        archive = load_archive(out_dir)
        assert archive.model
        tokens: set = set(archive.model.vocab._token_to_index['tokens'].keys())
        assert tokens == {'@@PADDING@@', '@@UNKNOWN@@', 'are', '.', 'animals', 'cats', 'dogs', 'snakes', 'birds'}
        train_complete: str = 'completed its entire epoch (training).'
        validation_complete: str = 'completed its entire epoch (validation).'
        import re
        pattern: re.Pattern = re.compile("First word from training data: '([^']*)'")
        first_word_counts: Counter = Counter()
        with open(os.path.join(out_dir, 'out_worker0.log')) as f:
            worker0_log: str = f.read()
            assert train_complete in worker0_log
            assert validation_complete in worker0_log
            for first_word in pattern.findall(worker0_log):
                first_word_counts[first_word] += 1
        with open(os.path.join(out_dir, 'out_worker1.log')) as f:
            worker1_log: str = f.read()
            assert train_complete in worker1_log
            assert validation_complete in worker1_log
            for first_word in pattern.findall(worker1_log):
                first_word_counts[first_word] += 1
        assert first_word_counts == {
            'cats': num_epochs,
            'dogs': num_epochs,
            'snakes': num_epochs,
            'birds': num_epochs
        }

    def test_distributed_raises_error_with_no_gpus(self) -> None:
        params: Params = Params({
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
                'type': 'sequence_tagging'
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            },
            'distributed': {}
        })
        with pytest.raises(ConfigurationError):
            train_model(
                params,
                serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model')
            )

    def test_train_saves_all_keys_in_config(self) -> None:
        params: Params = Params({
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
            'pytorch_seed': 42,
            'numpy_seed': 42,
            'random_seed': 42,
            'dataset_reader': {
                'type': 'sequence_tagging'
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            }
        })
        serialization_dir: str = os.path.join(self.TEST_DIR, 'test_train_model')
        params_as_dict: OrderedDict = params.as_ordered_dict()
        train_model(params, serialization_dir=serialization_dir)
        config_path: str = os.path.join(serialization_dir, CONFIG_NAME)
        with open(config_path) as config:
            saved_config_as_dict: OrderedDict = OrderedDict(json.load(config))
        assert params_as_dict == saved_config_as_dict

    def test_error_is_throw_when_cuda_device_is_not_available(self) -> None:
        params: Params = Params({
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
                'type': 'sequence_tagging'
            },
            'train_data_path': 'test_fixtures/data/sequence_tagging.tsv',
            'validation_data_path': 'test_fixtures/data/sequence_tagging.tsv',
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': 2,
                'cuda_device': torch.cuda.device_count(),
                'optimizer': 'adam'
            }
        })
        with pytest.raises(ConfigurationError, match='Experiment specified'):
            train_model(
                params,
                serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model')
            )

    def test_train_with_test_set(self) -> None:
        params: Params = Params({
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
                'type': 'sequence_tagging'
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'test_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'evaluate_on_test': True,
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            }
        })
        train_model(
            params,
            serialization_dir=os.path.join(self.TEST_DIR, 'train_with_test_set')
        )

    def test_train_number_of_steps(self) -> None:
        number_of_epochs: int = 2
        last_num_steps_per_epoch: Optional[int] = None

        @LearningRateScheduler.register('mock')
        class MockLRScheduler(ExponentialLearningRateScheduler):
            def __init__(
                self,
                optimizer: torch.optim.Optimizer,
                num_steps_per_epoch: int
            ) -> None:
                super().__init__(optimizer)
                nonlocal last_num_steps_per_epoch
                last_num_steps_per_epoch = num_steps_per_epoch

        batch_callback_counter: int = 0

        @TrainerCallback.register('counter')
        class CounterOnBatchCallback(TrainerCallback):
            def on_batch(
                self,
                trainer: GradientDescentTrainer,
                batch_inputs: List[Dict[str, Any]],
                batch_outputs: Dict[str, Any],
                batch_metrics: Dict[str, float],
                epoch: int,
                batch_number: int,
                is_training: bool,
                is_primary: bool = True,
                batch_grad_norm: Optional[float] = None,
                **kwargs: Any
            ) -> None:
                nonlocal batch_callback_counter
                if is_training:
                    batch_callback_counter += 1

        params: Params = Params({
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
                'type': 'sequence_tagging'
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'test_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'evaluate_on_test': True,
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': number_of_epochs,
                'optimizer': 'adam',
                'learning_rate_scheduler': {
                    'type': 'mock'
                },
                'callbacks': ['counter']
            }
        })
        train_model(
            params.duplicate(),
            serialization_dir=os.path.join(self.TEST_DIR, 'train_normal')
        )
        assert batch_callback_counter == (last_num_steps_per_epoch or 0) * number_of_epochs
        batch_callback_counter = 0
        normal_steps_per_epoch: Optional[int] = last_num_steps_per_epoch
        original_batch_size: int = params['data_loader']['batch_size']
        params['data_loader']['batch_size'] = 1
        train_model(
            params.duplicate(),
            serialization_dir=os.path.join(self.TEST_DIR, 'train_with_bs1')
        )
        assert batch_callback_counter == (last_num_steps_per_epoch or 0) * number_of_epochs
        batch_callback_counter = 0
        assert normal_steps_per_epoch == math.ceil((last_num_steps_per_epoch or 0) / original_batch_size)
        params['data_loader']['batch_size'] = original_batch_size
        params['trainer']['num_gradient_accumulation_steps'] = 3
        train_model(
            params,
            serialization_dir=os.path.join(self.TEST_DIR, 'train_with_ga')
        )
        assert batch_callback_counter == (last_num_steps_per_epoch or 0) * number_of_epochs
        batch_callback_counter = 0
        assert math.ceil((normal_steps_per_epoch or 0) / 3) == (last_num_steps_per_epoch or 0)

    def test_train_args(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Testing')
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Train().add_subparser(subparsers)
        for serialization_arg in ['-s', '--serialization-dir']:
            raw_args: List[str] = ['train', 'path/to/params', serialization_arg, 'serialization_dir']
            args = parser.parse_args(raw_args)
            assert args.func == train_model_from_args
            assert args.param_path == 'path/to/params'
            assert args.serialization_dir == 'serialization_dir'
        with pytest.raises(SystemExit) as cm:
            parser.parse_args(['train', '-s', 'serialization_dir'])
            assert cm.value.code == 2
        with pytest.raises(SystemExit) as cm:
            parser.parse_args(['train', 'path/to/params'])
            assert cm.value.code == 2

    def test_train_model_can_instantiate_from_params(self) -> None:
        params: Params = Params.from_file(self.FIXTURES_ROOT / 'simple_tagger' / 'experiment.json')
        TrainModel.from_params(
            params=params,
            serialization_dir=self.TEST_DIR,
            local_rank=0,
            batch_weight_key=''
        )

    def test_train_can_fine_tune_model_from_archive(self) -> None:
        params: Params = Params.from_file(
            self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_from_archive.jsonnet'
        )
        train_loop: TrainModel = TrainModel.from_params(
            params=params,
            serialization_dir=self.TEST_DIR,
            local_rank=0,
            batch_weight_key=''
        )
        train_loop.run()
        model: Model = Model.from_archive(
            self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz'
        )
        assert train_loop.model.vocab.get_vocab_size() > model.vocab.get_vocab_size()

    def test_train_nograd_regex(self) -> None:
        params_get: Callable[[], Params] = lambda: Params({
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
                'type': 'sequence_tagging'
            },
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            }
        })
        serialization_dir: str = os.path.join(self.TEST_DIR, 'test_train_nograd')
        regex_lists: List[List[str]] = [
            [],
            ['.*text_field_embedder.*'],
            ['.*text_field_embedder.*', '.*encoder.*']
        ]
        for regex_list in regex_lists:
            params: Params = params_get()
            params['trainer']['no_grad'] = regex_list
            shutil.rmtree(serialization_dir, ignore_errors=True)
            model: Model = train_model(params, serialization_dir=serialization_dir)
            for name, parameter in model.named_parameters():
                if any((re.search(regex, name) for regex in regex_list)):
                    assert not parameter.requires_grad
                else:
                    assert parameter.requires_grad
        params: Params = params_get()
        params['trainer']['no_grad'] = ['*']
        shutil.rmtree(serialization_dir, ignore_errors=True)
        with pytest.raises(Exception):
            train_model(params, serialization_dir=serialization_dir)

class TestDryRun(AllenNlpTestCase):

    def setup_method(self, method: Any) -> None:
        super().setup_method(method)
        self.params: Params = Params({
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
                'type': 'sequence_tagging'
            },
            'train_data_path': str(self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'),
            'validation_data_path': str(self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'),
            'data_loader': {
                'batch_size': 2
            },
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            }
        })

    def test_dry_run_doesnt_overwrite_vocab(self) -> None:
        vocab_path: str = str(self.TEST_DIR / 'vocabulary')
        os.mkdir(vocab_path)
        with open(os.path.join(vocab_path, 'test.txt'), 'a+') as open_file:
            open_file.write('test')
        with pytest.raises(ConfigurationError):
            train_model(self.params, self.TEST_DIR, dry_run=True)

    def test_dry_run_makes_vocab(self) -> None:
        vocab_path: str = str(self.TEST_DIR / 'vocabulary')
        train_model(self.params, self.TEST_DIR, dry_run=True)
        vocab_files: List[str] = os.listdir(vocab_path)
        assert set(vocab_files) == {
            '.lock',
            'labels.txt',
            'non_padded_namespaces.txt',
            'tokens.txt'
        }
        with open(os.path.join(vocab_path, 'tokens.txt')) as f:
            tokens: List[str] = [line.strip() for line in f]
        tokens.sort()
        assert tokens == ['.', '@@UNKNOWN@@', 'animals', 'are', 'birds', 'cats', 'dogs', 'horses', 'snakes']
        with open(os.path.join(vocab_path, 'labels.txt')) as f:
            labels: List[str] = [line.strip() for line in f]
        labels.sort()
        assert labels == ['N', 'V']

    def test_dry_run_with_extension(self) -> None:
        existing_serialization_dir: str = str(self.TEST_DIR / 'existing')
        extended_serialization_dir: str = str(self.TEST_DIR / 'extended')
        existing_vocab_path: str = os.path.join(existing_serialization_dir, 'vocabulary')
        extended_vocab_path: str = os.path.join(extended_serialization_dir, 'vocabulary')
        vocab: Vocabulary = Vocabulary()
        vocab.add_token_to_namespace('some_weird_token_1', namespace='tokens')
        vocab.add_token_to_namespace('some_weird_token_2', namespace='tokens')
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)
        self.params['vocabulary'] = {}
        self.params['vocabulary']['type'] = 'extend'
        self.params['vocabulary']['directory'] = str(existing_vocab_path)
        self.params['vocabulary']['min_count'] = {'tokens': 3}
        train_model(self.params, extended_serialization_dir, dry_run=True)
        vocab_files: List[str] = os.listdir(extended_vocab_path)
        assert set(vocab_files) == {
            '.lock',
            'labels.txt',
            'non_padded_namespaces.txt',
            'tokens.txt'
        }
        with open(os.path.join(extended_vocab_path, 'tokens.txt')) as f:
            tokens: List[str] = [line.strip() for line in f]
        assert tokens[0] == '@@UNKNOWN@@'
        assert tokens[1] == 'some_weird_token_1'
        assert tokens[2] == 'some_weird_token_2'
        tokens_sorted: List[str] = sorted(tokens)
        assert tokens_sorted == [
            '.', '@@UNKNOWN@@', 'animals', 'are',
            'some_weird_token_1', 'some_weird_token_2'
        ]
        with open(os.path.join(extended_vocab_path, 'labels.txt')) as f:
            labels: List[str] = [line.strip() for line in f]
        labels_sorted: List[str] = sorted(labels)
        assert labels_sorted == ['N', 'V']

    def test_dry_run_without_extension(self) -> None:
        existing_serialization_dir: str = str(self.TEST_DIR / 'existing')
        extended_serialization_dir: str = str(self.TEST_DIR / 'extended')
        existing_vocab_path: str = os.path.join(existing_serialization_dir, 'vocabulary')
        extended_vocab_path: str = os.path.join(extended_serialization_dir, 'vocabulary')
        vocab: Vocabulary = Vocabulary()
        vocab.add_token_to_namespace('some_weird_token_1', namespace='tokens')
        vocab.add_token_to_namespace('some_weird_token_2', namespace='tokens')
        vocab.add_token_to_namespace('N', namespace='labels')
        vocab.add_token_to_namespace('V', namespace='labels')
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)
        self.params['vocabulary'] = {}
        self.params['vocabulary']['type'] = 'from_files'
        self.params['vocabulary']['directory'] = str(existing_vocab_path)
        train_model(self.params, extended_serialization_dir, dry_run=True)
        with open(os.path.join(extended_vocab_path, 'tokens.txt')) as f:
            tokens: List[str] = [line.strip() for line in f]
        assert tokens[0] == '@@UNKNOWN@@'
        assert tokens[1] == 'some_weird_token_1'
        assert tokens[2] == 'some_weird_token_2'
        assert len(tokens) == 3

    def test_make_vocab_args(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Testing')
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Train().add_subparser(subparsers)
        for serialization_arg in ['-s', '--serialization-dir']:
            raw_args: List[str] = ['train', 'path/to/params', serialization_arg, 'serialization_dir', '--dry-run']
            args = parser.parse_args(raw_args)
            assert args.func == train_model_from_args
            assert args.param_path == 'path/to/params'
            assert args.serialization_dir == 'serialization_dir'
            assert getattr(args, 'dry_run', False)
    
    def test_warn_validation_loader_batches_per_epoch(self) -> None:
        self.params['data_loader']['batches_per_epoch'] = 3
        with pytest.warns(UserWarning, match='batches_per_epoch'):
            train_model(self.params, self.TEST_DIR, dry_run=True)
