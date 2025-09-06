import argparse
import copy
import json
import logging
import math
import os
import re
import shutil
from collections import OrderedDict, Counter
from typing import Optional, List, Dict, Any
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

    def func_l58u91x2(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[TensorDict],
        batch_outputs: Any,
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        **kwargs: Any
    ) -> None:
        if is_training:
            logger = logging.getLogger(__name__)
            for batch in batch_inputs:
                for metadata in batch['metadata']:
                    logger.info(
                        f"First word from training data: '{metadata['words'][0]}'"
                    )


_seen_training_devices: set = set()


@TrainerCallback.register('training_device_logger')
class TrainingDeviceLoggerOnBatchCallback(TrainerCallback):

    def func_l58u91x2(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[TensorDict],
        batch_outputs: Any,
        batch_metrics: Dict[str, Any],
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

    def func_zjplb8th(
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

    def func_vsyn2f6z(self) -> None:
        params: Params = lambda: copy.deepcopy(self.DEFAULT_PARAMS)
        serialization_dir: str = os.path.join(self.TEST_DIR, 'test_train_model')
        train_model(params(), serialization_dir=serialization_dir)
        archive: Any = load_archive(os.path.join(serialization_dir, 'model.tar.gz'))
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
            train_model(
                params(),
                serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'),
                force=True,
                recover=True
            )

    @cpu_or_gpu
    def func_9foizr6l(self) -> None:
        import copy
        params: Params = copy.deepcopy(self.DEFAULT_PARAMS)
        params['trainer']['callbacks'] = ['training_device_logger']
        global _seen_training_devices
        _seen_training_devices.clear()
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_detect_gpu'))
        assert len(_seen_training_devices) == 1
        seen_training_device = next(iter(_seen_training_devices))
        if torch.cuda.device_count() == 0:
            assert seen_training_device.type == 'cpu'
        else:
            assert seen_training_device.type == 'cuda'

    @cpu_or_gpu
    def func_t7q4xw79(self) -> None:
        import copy
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
            seen_training_device = next(iter(_seen_training_devices))
            assert seen_training_device.type == 'cuda'

    @cpu_or_gpu
    def func_ie29kz3e(self) -> None:
        import copy
        params: Params = copy.deepcopy(self.DEFAULT_PARAMS)
        params['trainer']['callbacks'] = ['training_device_logger']
        params['trainer']['cuda_device'] = -1
        global _seen_training_devices
        _seen_training_devices.clear()
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_force_cpu'))
        assert len(_seen_training_devices) == 1
        seen_training_device = next(iter(_seen_training_devices))
        assert seen_training_device.type == 'cpu'

    @cpu_or_gpu
    def func_wkesmv6q(self) -> None:
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
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
            'dataset_reader': {'type': 'sequence_tagging'},
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {'batch_size': 2},
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam',
                'callbacks': [
                    'tests.commands.train_test.TrainingPrimaryCheckCallback'
                ]
            },
            'distributed': {'cuda_devices': devices}
        })
        out_dir: str = os.path.join(self.TEST_DIR, 'test_distributed_train')
        train_model(params, serialization_dir=out_dir)
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
    def func_zln321mh(self, max_instances: Optional[int], grad_acc: Optional[int], batch_size: int) -> None:
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
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
            'dataset_reader': {'type': 'sequence_tagging', 'max_instances': max_instances},
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
        train_model(params, serialization_dir=out_dir)
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
    def func_9bk93k9a(self, max_instances_in_memory: Optional[int]) -> None:
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
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
            'dataset_reader': {'type': 'sharded', 'base_reader': {'type': 'sequence_tagging'}},
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
        train_model(params, serialization_dir=out_dir)
        serialized_files: List[str] = os.listdir(out_dir)
        assert 'out_worker0.log' in serialized_files
        assert 'out_worker1.log' in serialized_files
        assert 'model.tar.gz' in serialized_files
        archive: Any = load_archive(out_dir)
        assert archive.model
        tokens: set = archive.model.vocab._token_to_index['tokens'].keys()
        assert tokens == {
            '@@PADDING@@', '@@UNKNOWN@@', 'are', '.', 'animals',
            'plants', 'vehicles', 'cats', 'dogs', 'snakes',
            'birds', 'ferns', 'trees', 'flowers', 'vegetables',
            'cars', 'buses', 'planes', 'rockets'
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
    def func_pmontpni(self, max_instances_in_memory: Optional[int]) -> None:
        if torch.cuda.device_count() >= 2:
            devices: List[int] = [0, 1]
        else:
            devices = [-1, -1]
        num_epochs: int = 2
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
            'dataset_reader': {'type': 'sequence_tagging', 'max_instances': 4},
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {
                'batch_size': 1,
                'max_instances_in_memory': max_instances_in_memory
            },
            'trainer': {
                'num_epochs': num_epochs,
                'optimizer': 'adam',
                'callbacks': [
                    'tests.commands.train_test.TrainingDataLoggerOnBatchCallback'
                ]
            },
            'distributed': {'cuda_devices': devices}
        })
        out_dir: str = os.path.join(self.TEST_DIR, 'test_distributed_train')
        train_model(params, serialization_dir=out_dir)
        serialized_files: List[str] = os.listdir(out_dir)
        assert 'out_worker0.log' in serialized_files
        assert 'out_worker1.log' in serialized_files
        assert 'model.tar.gz' in serialized_files
        archive: Any = load_archive(out_dir)
        assert archive.model
        tokens: set = set(archive.model.vocab._token_to_index['tokens'].keys())
        assert tokens == {
            '@@PADDING@@', '@@UNKNOWN@@', 'are', '.', 'animals',
            'cats', 'dogs', 'snakes', 'birds'
        }
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

    def func_n8hot5d6(self) -> None:
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
            'dataset_reader': {'type': 'sequence_tagging'},
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {'batch_size': 2},
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            },
            'distributed': {}
        })
        with pytest.raises(ConfigurationError):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))

    def func_gkxvgnkg(self) -> None:
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
            'dataset_reader': {'type': 'sequence_tagging'},
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'data_loader': {'batch_size': 2},
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            }
        })
        serialization_dir: str = os.path.join(self.TEST_DIR, 'test_train_model')
        params_as_dict: Dict[str, Any] = params.as_ordered_dict()
        train_model(params, serialization_dir=serialization_dir)
        config_path: str = os.path.join(serialization_dir, CONFIG_NAME)
        with open(config_path) as config:
            saved_config_as_dict: OrderedDict = OrderedDict(json.load(config))
        assert params_as_dict == saved_config_as_dict

    def func_6b4ztz6l(self) -> None:
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
            'dataset_reader': {'type': 'sequence_tagging'},
            'train_data_path': 'test_fixtures/data/sequence_tagging.tsv',
            'validation_data_path': 'test_fixtures/data/sequence_tagging.tsv',
            'data_loader': {'batch_size': 2},
            'trainer': {
                'num_epochs': 2,
                'cuda_device': torch.cuda.device_count(),
                'optimizer': 'adam'
            }
        })
        with pytest.raises(ConfigurationError, match='Experiment specified'):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'test_train_model'))

    def func_6y7zzte6(self) -> None:
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
            'dataset_reader': {'type': 'sequence_tagging'},
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'test_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'evaluate_on_test': True,
            'data_loader': {'batch_size': 2},
            'trainer': {
                'num_epochs': 2,
                'optimizer': 'adam'
            }
        })
        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, 'train_with_test_set'))

    def func_usblky1p(self) -> None:
        number_of_epochs: int = 2
        last_num_steps_per_epoch: Optional[int] = None

        @LearningRateScheduler.register('mock')
        class MockLRScheduler(ExponentialLearningRateScheduler):

            def __init__(self, optimizer: torch.optim.Optimizer, num_steps_per_epoch: int) -> None:
                super().__init__(optimizer)
                nonlocal last_num_steps_per_epoch
                last_num_steps_per_epoch = num_steps_per_epoch

        batch_callback_counter: int = 0

        @TrainerCallback.register('counter')
        class CounterOnBatchCallback(TrainerCallback):

            def func_l58u91x2(
                self,
                trainer: GradientDescentTrainer,
                batch_inputs: List[TensorDict],
                batch_outputs: Any,
                batch_metrics: Dict[str, Any],
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
            'dataset_reader': {'type': 'sequence_tagging'},
            'train_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'test_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'validation_data_path': SEQUENCE_TAGGING_DATA_PATH,
            'evaluate_on_test': True,
            'data_loader': {'batch_size': 2},
            'trainer': {
                'num_epochs': number_of_epochs,
                'optimizer': 'adam',
                'learning_rate_scheduler': {'type': 'mock'},
                'callbacks': ['counter']
            }
        })
        train_model(params.duplicate(), serialization_dir=os.path.join(self.TEST_DIR, 'train_normal'))
        assert batch_callback_counter == (last_num_steps_per_epoch or 0) * number_of_epochs
        batch_callback_counter = 0
        normal_steps_per_epoch: int = last_num_steps_per_epoch or 0
        original_batch_size: int = params['data_loader']['batch_size']
        params['data_loader']['batch_size'] = 1
        train_model(params.duplicate(), serialization_dir=os.path.join(self.TEST_DIR, 'train_with_bs1'))
        assert batch_callback_counter == normal_steps_per_epoch * number_of_epochs
        batch_callback_counter = 0
        assert normal_steps_per_epoch == math.ceil((last_num_steps_per_epoch or 0) / original_batch_size)
        params['data_loader']['batch_size'] = original_batch_size
        params['trainer']['num_gradient_accumulation_steps'] = 3
        train_model(
            params,
            serialization_dir=os.path.join(self.TEST_DIR, 'train_with_ga')
        )
        assert batch_callback_counter == last_num_steps_per_epoch * number_of_epochs
        batch_callback_counter = 0
        assert math.ceil((normal_steps_per_epoch or 0) / 3) == (last_num_steps_per_epoch or 0)

    def func_lhqa32s5(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Testing')
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        Train().add_subparser(subparsers)
        for serialization_arg in ['-s', '--serialization-dir']:
            raw_args: List[str] = ['train', 'path/to/params', serialization_arg, 'serialization_dir', '--dry-run']
            args: argparse.Namespace = parser.parse_args(raw_args)
            assert args.func == train_model_from_args
            assert args.param_path == 'path/to/params'
            assert args.serialization_dir == 'serialization_dir'
            assert args.dry_run

    def func_pnawm4mb(self) -> None:
        self.params['data_loader']['batches_per_epoch'] = 3
        with pytest.warns(UserWarning, match='batches_per_epoch'):
            train_model(self.params, self.TEST_DIR, dry_run=True)
