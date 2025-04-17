```python
"""
The `train` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.
"""

import argparse
import logging
import os
from os import PathLike
from typing import Any, Dict, List, Optional, Union, cast
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


@Subcommand.register("train")
class Train(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Train a model.")

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace) -> None:
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """
    train_model_from_file(
        parameter_filename=args.param_path,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
        recover=args.recover,
        force=args.force,
        node_rank=args.node_rank,
        include_package=args.include_package,
        dry_run=args.dry_run,
        file_friendly_logging=args.file_friendly_logging,
    )


def train_model_from_file(
    parameter_filename: Union[str, PathLike],
    serialization_dir: Union[str, PathLike],
    overrides: Union[str, Dict[str, Any]] = "",
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: Optional[List[str]] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    return_model: Optional[bool] = None,
) -> Optional[Model]:
    """
    A wrapper around [`train_model`](#train_model) which loads the params from a file.

    # Parameters

    parameter_filename : `str`
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : `str`
        The directory in which to save results and logs. We just pass this along to
        [`train_model`](#train_model).
    overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
        A JSON string or a dict that we will use to override values in the input parameter file.
    recover : `bool`, optional (default=`False`)
        If `True`, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see `Model.from_archive`.
    force : `bool`, optional (default=`False`)
        If `True`, we will overwrite the serialization directory if it already exists.
    node_rank : `int`, optional
        Rank of the current node in distributed training
    include_package : `List[str]`, optional
        In distributed mode, extra packages mentioned will be imported in trainer workers.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    return_model : `Optional[bool]`, optional (default = `None`)
        Whether or not to return the final model. If not specified, this defaults to `False` for
        distributed training and `True` otherwise.

    # Returns

    best_model : `Optional[str]`
        The path to the archived model with the best weights or `None` if in dry run.
    best_model : `Optional[Model]`
        The model with the best epoch weights or `None`, depending on the value of `return_model` and `dry_run`.
    """
    # Load the experiment config from a file and pass it to `train_model`.
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
        return_model=return_model,
    )


def train_model(
    params: Params,
    serialization_dir: Union[str, PathLike],
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: Optional[List[str]] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    return_model: Optional[bool] = None,
) -> Optional[Model]:
    """
    Trains the model specified in the given [`Params`](../common/params.md#params) object, using the data
    and training parameters also specified in that object, and saves the results in `serialization_dir`.

    # Parameters

    params : `Params`
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : `str`
        The directory in which to save results and logs.
    recover : `bool`, optional (default=`False`)
        If `True`, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see `Model.from_archive`.
    force : `bool`, optional (default=`False`)
        If `True`, we will overwrite the serialization directory if it already exists.
    node_rank : `int`, optional
        Rank of the current node in distributed training
    include_package : `List[str]`, optional
        In distributed mode, extra packages mentioned will be imported in trainer workers.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    return_model : `Optional[bool]`, optional (default = `None`)
        Whether or not to return the final model. If not specified, this defaults to `False` for
        distributed training and `True` otherwise.

    # Returns

    best_model : `Optional[Model]`
        The model with the best epoch weights or `None`, depending on the value of `return_model` and `dry_run`.
    """
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    training_util.create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    # Change Author: Gabe Orlanski
    # Placeholder for the time being to make sure no errors are raised b/c of
    # the evaluator.
    params.pop("evaluation", None)

    meta = Meta.new()
    meta.to_file(os.path.join(serialization_dir, META_NAME))

    include_in_archive = params.pop("include_in_archive", None)
    verify_include_in_archive(include_in_archive)

    model: Optional[Model] = None

    distributed_params = params.params.pop("distributed", None)
    # If distributed isn't in the config and the config contains strictly
    # one cuda device, we just run a single training process.
    if distributed_params is None:
        model = _train_worker(
            process_rank=0,
            params=params,
            serialization_dir=serialization_dir,
            include_package=include_package,
            dry_run=dry_run,
            file_friendly_logging=file_friendly_logging,
        )
    # Otherwise, we are running multiple processes for training.
    else:
        common_logging.prepare_global_logging(
            serialization_dir,
            rank=0,
            world_size=1,
        )

        # We are careful here so that we can raise a good error if someone
        # passed the wrong thing - cuda_devices are required.
        device_ids = distributed_params.pop("cuda_devices", None)
        multi_device = isinstance(device_ids, list) and len(device_ids) > 1
        num_nodes = distributed_params.pop("num_nodes", 1)

        if not (multi_device or num_nodes > 1):
            raise ConfigurationError(
                "Multiple cuda devices/nodes need to be configured to run distributed training."
            )
        check_for_gpu(device_ids)

        primary_addr = distributed_params.pop("primary_address", "127.0.0.1")
        if primary_addr in ("127.0.0.1", "0.0.0.0", "localhost"):
            # If running locally, we can automatically find an open port if one is not specified.
            primary_port = (
                distributed_params.pop("primary_port", None) or common_util.find_open_port()
            )
        else:
            # Otherwise we require that the port be specified.
            primary_port = distributed_params.pop("primary_port")

        num_procs = len(device_ids)
        world_size = num_nodes * num_procs

        # Creating `Vocabulary` objects from workers could be problematic since
        # the data loaders in each worker will yield only `rank` specific
        # instances. Hence it is safe to construct the vocabulary and write it
        # to disk before initializing the distributed context. The workers will
        # load the vocabulary from the path specified.
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        if recover:
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = training_util.make_vocab_from_params(
                params.duplicate(), serialization_dir, print_statistics=dry_run
            )
        params["vocabulary"] = {
            "type": "from_files",
            "directory": vocab_dir,
            "padding_token": vocab._padding_token,
            "oov_token": vocab._oov_token,
        }

        logging.info(
            "Switching to distributed training mode since multiple GPUs are configured | "
            f"Primary is at: {primary_addr}:{primary_port} | Rank of this node: {node_rank} | "
            f"Number of workers in this node: {num_procs} | Number of nodes: {num_nodes} | "
            f"World size: {world_size}"
        )

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
                Params(distributed_params),
            ),
            nprocs=num_procs,
        )

    if not dry_run:
        archive_model(serialization_dir, include_in_archive=include_in_archive)
    else:
        return None

    if return_model is None:
        return model  # model may or may not be `None`.
    elif return_model is True:
        return model if model is not None else Model.load(params, serialization_dir)
    else:
        return None


def _train_worker(
    process_rank: int,
    params: Params,
    serialization_dir: Union[str, PathLike],
    include_package: Optional[List[str]] = None,
    dry_run: bool = False,
    node_rank: int = 0,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
    world_size: int = 1,
    distributed_device_ids: Optional[List[int]] = None,
    file_friendly_logging: bool = False,
    include_in_archive: Optional[List[str]] = None,
    distributed_params: Optional[Params] = None,
) -> Optional[Model]:
    """
    Helper to train the configured model/experiment. In distributed mode, this is spawned as a
    worker process. In a single GPU experiment, this returns the `Model` object and in distributed
    training, nothing is returned.

    # Parameters

    process_rank : `int`
        The process index that is initialized using the GPU device id.
    params : `Params`
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : `str`
        The directory in which to save results and logs.
    include_package : `List[str]`, optional
        In distributed mode, since this function would have been spawned as a separate process,
        the extra imports need to be done again. NOTE: This does not have any effect in single
        GPU training.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    node_rank : `int`, optional
        Rank of the node.
    primary_addr : `str`, optional (default=`"127.0.0.1"`)
        Address of the primary node for distributed training.
    primary_port : `str`, optional (default=`"29500"`)
        Port of the primary node for distributed training.
    world_size : `int`, optional
        The number of processes involved in distributed training.
    distributed_device_ids: `List[str]`, optional
        IDs of the devices used involved in distributed training.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    include_in_archive : `List[str]`, optional
        Paths relative to `serialization_dir` that should be archived in addition to the default ones.
    distributed_params : `Optional[Params]`, optional
        Additional distributed params.

    # Returns

    best_model : `Optional[Model]`
        The model with the best epoch weights or `None` if in distributed training or in dry run.
    """
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    common_logging.prepare_global_logging(
        serialization_dir,
        rank=process_rank,
        world_size=world_size,
    )
    common_util.prepare_environment(params)

    distributed = world_size > 1

    primary = process_rank == 0

    include_package = include_package or []

    ddp_accelerator: Optional[DdpAccelerator] = None

    if distributed:
        assert distributed_device_ids is not None
        assert distributed_params is not None

        # Since the worker is spawned and not forked, the extra imports need to be done again.
        # Both the ones from the plugins and the ones from `include_package`.
        import_plugins()
        for package_name in include_package:
            common_util.import_module_and_submodules(package_name)

        num_procs_per_node = len(distributed_device_ids)
        # The Unique identifier of the worker process among all the processes in the
        # distributed training group is computed here. This is used while initializing
        # the process group using `init_process_group`
        global_rank = node_rank * num_procs_per_node + process_rank

        # Number of processes per node is useful to know if a process
        # is a primary in the local node(node in which it is running)
        os.environ["ALLENNLP_PROCS_PER_NODE"] = str(num_procs_per_node)

        # In distributed training, the configured device is always going to be a list.
        # The corresponding gpu id for the particular worker is obtained by picking the id
        # from the device list with the rank as index
        gpu_id = int(distributed_device_ids[process_rank])  # type: ignore

        # Till now,