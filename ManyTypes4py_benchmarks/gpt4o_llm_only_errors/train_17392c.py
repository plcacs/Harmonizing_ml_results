import argparse
import logging
import os
from os import PathLike
from typing import Any, Dict, List, Optional, Union
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.meta import Meta, META_NAME
from allennlp.common import logging as common_logging
from allennlp.common import util as common_util
from allennlp.common.plugins import import_plugins
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.models.archival import archive_model, CONFIG_NAME, verify_include_in_archive
from allennlp.models.model import Model
from allennlp.nn.parallel import DdpAccelerator
from allennlp.training.trainer import Trainer
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)

@Subcommand.register('train')
class Train(Subcommand):

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = 'Train the specified model on the specified dataset.'
        subparser = parser.add_parser(self.name, description=description, help='Train a model.')
        subparser.add_argument('param_path', type=str, help='path to parameter file describing the model to be trained')
        subparser.add_argument('-s', '--serialization-dir', required=True, type=str, help='directory in which to save the model and its logs')
        subparser.add_argument('-r', '--recover', action='store_true', default=False, help='recover training from the state in serialization_dir')
        subparser.add_argument('-f', '--force', action='store_true', required=False, help='overwrite the output directory if it exists')
        subparser.add_argument('-o', '--overrides', type=str, default='', help='a json(net) structure used to override the experiment configuration, e.g., \'{"iterator.batch_size": 16}\'.  Nested parameters can be specified either with nested dictionaries or with dot syntax.')
        subparser.add_argument('--node-rank', type=int, default=0, help='rank of this node in the distributed setup')
        subparser.add_argument('--dry-run', action='store_true', help='do not train a model, but create a vocabulary, show dataset statistics and other training information')
        subparser.add_argument('--file-friendly-logging', action='store_true', default=False, help='outputs tqdm status on separate lines and slows tqdm refresh rate')
        subparser.set_defaults(func=train_model_from_args)
        return subparser

def train_model_from_args(args: argparse.Namespace) -> None:
    train_model_from_file(
        parameter_filename=args.param_path,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
        recover=args.recover,
        force=args.force,
        node_rank=args.node_rank,
        include_package=args.include_package,
        dry_run=args.dry_run,
        file_friendly_logging=args.file_friendly_logging
    )

def train_model_from_file(
    parameter_filename: str,
    serialization_dir: str,
    overrides: Union[str, Dict[str, Any]] = '',
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: Optional[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    return_model: Optional[bool] = None
) -> Optional[Union[str, Model]]:
    params = Params.from_file(parameter_filename, overrides)
    return train_model(
        params=params,
        serialization_dir=serialization_dir,
        recover=recover,
        force=force,
        node_rank=node_rank,
        include_package=include_package,
        dry_run=dry_run,
        file_friendly_logging=file_friendly_logging,
        return_model=return_model
    )

def train_model(
    params: Params,
    serialization_dir: str,
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: Optional[List[str]] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    return_model: Optional[bool] = None
) -> Optional[Model]:
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging
    training_util.create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))
    params.pop('evaluation', None)
    meta = Meta.new()
    meta.to_file(os.path.join(serialization_dir, META_NAME))
    include_in_archive = params.pop('include_in_archive', None)
    verify_include_in_archive(include_in_archive)
    model = None
    distributed_params = params.params.pop('distributed', None)
    if distributed_params is None:
        model = _train_worker(
            process_rank=0,
            params=params,
            serialization_dir=serialization_dir,
            include_package=include_package,
            dry_run=dry_run,
            file_friendly_logging=file_friendly_logging
        )
    else:
        common_logging.prepare_global_logging(serialization_dir, rank=0, world_size=1)
        device_ids = distributed_params.pop('cuda_devices', None)
        multi_device = isinstance(device_ids, list) and len(device_ids) > 1
        num_nodes = distributed_params.pop('num_nodes', 1)
        if not (multi_device or num_nodes > 1):
            raise ConfigurationError('Multiple cuda devices/nodes need to be configured to run distributed training.')
        check_for_gpu(device_ids)
        primary_addr = distributed_params.pop('primary_address', '127.0.0.1')
        if primary_addr in ('127.0.0.1', '0.0.0.0', 'localhost'):
            primary_port = distributed_params.pop('primary_port', None) or common_util.find_open_port()
        else:
            primary_port = distributed_params.pop('primary_port')
        num_procs = len(device_ids)
        world_size = num_nodes * num_procs
        vocab_dir = os.path.join(serialization_dir, 'vocabulary')
        if recover:
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = training_util.make_vocab_from_params(params.duplicate(), serialization_dir, print_statistics=dry_run)
        params['vocabulary'] = {'type': 'from_files', 'directory': vocab_dir, 'padding_token': vocab._padding_token, 'oov_token': vocab._oov_token}
        logging.info(f'Switching to distributed training mode since multiple GPUs are configured | Primary is at: {primary_addr}:{primary_port} | Rank of this node: {node_rank} | Number of workers in this node: {num_procs} | Number of nodes: {num_nodes} | World size: {world_size}')
        mp.spawn(
            _train_worker,
            args=(
                params.duplicate(),
                serialization_dir,
                include_package,
                dry_run,
                node_rank,
                primary_addr,
                primary_port,
                world_size,
                device_ids,
                file_friendly_logging,
                include_in_archive,
                Params(distributed_params)
            ),
            nprocs=num_procs
        )
    if not dry_run:
        archive_model(serialization_dir, include_in_archive=include_in_archive)
    else:
        return None
    if return_model is None:
        return model
    elif return_model is True:
        return model if model is not None else Model.load(params, serialization_dir)
    else:
        return None

def _train_worker(
    process_rank: int,
    params: Params,
    serialization_dir: str,
    include_package: Optional[List[str]] = None,
    dry_run: bool = False,
    node_rank: int = 0,
    primary_addr: str = '127.0.0.1',
    primary_port: int = 29500,
    world_size: int = 1,
    distributed_device_ids: Optional[List[str]] = None,
    file_friendly_logging: bool = False,
    include_in_archive: Optional[List[str]] = None,
    distributed_params: Optional[Params] = None
) -> Optional[Model]:
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging
    common_logging.prepare_global_logging(serialization_dir, rank=process_rank, world_size=world_size)
    common_util.prepare_environment(params)
    distributed = world_size > 1
    primary = process_rank == 0
    include_package = include_package or []
    ddp_accelerator = None
    if distributed:
        assert distributed_device_ids is not None
        assert distributed_params is not None
        import_plugins()
        for package_name in include_package:
            common_util.import_module_and_submodules(package_name)
        num_procs_per_node = len(distributed_device_ids)
        global_rank = node_rank * num_procs_per_node + process_rank
        os.environ['ALLENNLP_PROCS_PER_NODE'] = str(num_procs_per_node)
        gpu_id = int(distributed_device_ids[process_rank])
        params['trainer']['local_rank'] = process_rank
        params['trainer']['cuda_device'] = gpu_id
        params['trainer']['world_size'] = world_size
        params['trainer']['distributed'] = True
        if gpu_id >= 0:
            torch.cuda.set_device(gpu_id)
            dist.init_process_group(backend='nccl', init_method=f'tcp://{primary_addr}:{primary_port}', world_size=world_size, rank=global_rank)
        else:
            dist.init_process_group(backend='gloo', init_method=f'tcp://{primary_addr}:{primary_port}', world_size=world_size, rank=global_rank)
        if 'ddp_accelerator' in distributed_params:
            ddp_accelerator_params = distributed_params.pop('ddp_accelerator')
            ddp_accelerator = DdpAccelerator.from_params(ddp_accelerator_params, local_rank=process_rank, world_size=world_size, cuda_device=gpu_id)
        logging.info(f'Process group of world size {world_size} initialized for distributed training in worker {global_rank}')
    train_loop = TrainModel.from_params(params=params, serialization_dir=serialization_dir, local_rank=process_rank, ddp_accelerator=ddp_accelerator)
    if dry_run:
        return None
    try:
        if distributed:
            dist.barrier()
        metrics = train_loop.run()
    except (KeyboardInterrupt, common_util.SigTermReceived):
        if primary:
            best_weights_path = train_loop.trainer.get_best_weights_path()
            if best_weights_path is None:
                logging.info('Training interrupted by the user, and no best model has been saved. No model archive created.')
            else:
                logging.info('Training interrupted by the user. Attempting to create a model archive using the current best epoch weights.')
                archive_model(serialization_dir, weights=best_weights_path, include_in_archive=include_in_archive)
        raise
    if primary:
        train_loop.finish(metrics)
    if not distributed:
        return train_loop.model
    return None

class TrainModel(Registrable):
    default_implementation = 'default'

    def __init__(
        self,
        serialization_dir: str,
        model: Model,
        trainer: Trainer,
        evaluation_data_loader: Optional[DataLoader] = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = ''
    ) -> None:
        self.serialization_dir = serialization_dir
        self.model = model
        self.trainer = trainer
        self.evaluation_data_loader = evaluation_data_loader
        self.evaluate_on_test = evaluate_on_test
        self.batch_weight_key = batch_weight_key

    def run(self) -> Dict[str, Any]:
        return self.trainer.train()

    def finish(self, metrics: Dict[str, Any]) -> None:
        if self.evaluation_data_loader is not None and self.evaluate_on_test:
            logger.info('The model will be evaluated using the best epoch weights.')
            test_metrics = training_util.evaluate(self.model, self.evaluation_data_loader, cuda_device=self.trainer.cuda_device, batch_weight_key=self.batch_weight_key)
            for key, value in test_metrics.items():
                metrics['test_' + key] = value
        elif self.evaluation_data_loader is not None:
            logger.info("To evaluate on the test set after training, pass the 'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")
        common_util.dump_metrics(os.path.join(self.serialization_dir, 'metrics.json'), metrics, log=True)

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        local_rank: int,
        dataset_reader: DatasetReader,
        train_data_path: str,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary),
        datasets_for_vocab_creation: Optional[List[str]] = None,
        validation_dataset_reader: Optional[DatasetReader] = None,
        validation_data_path: Optional[str] = None,
        validation_data_loader: Optional[Lazy[DataLoader]] = None,
        test_data_path: Optional[str] = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = '',
        ddp_accelerator: Optional[DdpAccelerator] = None
    ) -> 'TrainModel':
        data_loaders = {'train': data_loader.construct(reader=dataset_reader, data_path=train_data_path)}
        if validation_data_path is not None:
            validation_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders['validation'] = validation_data_loader.construct(reader=validation_dataset_reader, data_path=validation_data_path)
            else:
                data_loaders['validation'] = data_loader.construct(reader=validation_dataset_reader, data_path=validation_data_path)
                if getattr(data_loaders['validation'], 'batches_per_epoch', None) is not None:
                    warnings.warn("Using 'data_loader' params to construct validation data loader since 'validation_data_loader' params not specified, but you have 'data_loader.batches_per_epoch' set which may result in different validation datasets for each epoch.", UserWarning)
        if test_data_path is not None:
            test_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders['test'] = validation_data_loader.construct(reader=test_dataset_reader, data_path=test_data_path)
            else:
                data_loaders['test'] = data_loader.construct(reader=test_dataset_reader, data_path=test_data_path)
        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in data_loaders:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")
            logger.info('From dataset instances, %s will be considered for vocabulary creation.', ', '.join(datasets_for_vocab_creation))
        instance_generator = (instance for key, data_loader in data_loaders.items() if datasets_for_vocab_creation is None or key in datasets_for_vocab_creation for instance in data_loader.iter_instances())
        vocabulary_ = vocabulary.construct(instances=instance_generator)
        model_ = model.construct(vocab=vocabulary_, serialization_dir=serialization_dir, ddp_accelerator=ddp_accelerator)
        if local_rank == 0:
            vocabulary_path = os.path.join(serialization_dir, 'vocabulary')
            vocabulary_.save_to_files(vocabulary_path)
        for data_loader_ in data_loaders.values():
            data_loader_.index_with(model_.vocab)
        trainer_ = trainer.construct(serialization_dir=serialization_dir, model=model_, data_loader=data_loaders['train'], validation_data_loader=data_loaders.get('validation'), local_rank=local_rank, ddp_accelerator=ddp_accelerator)
        assert trainer_ is not None
        return cls(serialization_dir=serialization_dir, model=model_, trainer=trainer_, evaluation_data_loader=data_loaders.get('test'), evaluate_on_test=evaluate_on_test, batch_weight_key=batch_weight_key)

TrainModel.register('default', constructor='from_partial_objects')(TrainModel)
