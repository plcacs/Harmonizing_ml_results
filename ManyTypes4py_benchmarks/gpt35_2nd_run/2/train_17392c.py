from typing import Any, Dict, List, Optional, Union

def train_model_from_args(args: argparse.Namespace) -> None:
    ...

def train_model_from_file(parameter_filename: str, serialization_dir: str, overrides: Union[str, Dict[str, Any]] = '', recover: bool = False, force: bool = False, node_rank: int = 0, include_package: Optional[str] = None, dry_run: bool = False, file_friendly_logging: bool = False, return_model: Optional[bool] = None) -> Union[str, Model]:
    ...

def train_model(params: Params, serialization_dir: str, recover: bool = False, force: bool = False, node_rank: int = 0, include_package: Optional[str] = None, dry_run: bool = False, file_friendly_logging: bool = False, return_model: Optional[bool] = None) -> Optional[Model]:
    ...

def _train_worker(process_rank: int, params: Params, serialization_dir: str, include_package: Optional[List[str]] = None, dry_run: bool = False, node_rank: int = 0, primary_addr: str = '127.0.0.1', primary_port: str = '29500', world_size: int = 1, distributed_device_ids: Optional[List[str]] = None, file_friendly_logging: bool = False, include_in_archive: Optional[List[str]] = None, distributed_params: Optional[Params] = None) -> Optional[Model]:
    ...

@classmethod
def from_partial_objects(cls, serialization_dir: str, local_rank: int, dataset_reader: DatasetReader, train_data_path: str, model: Lazy[Model], data_loader: Lazy[DataLoader], trainer: Lazy[Trainer], vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary), datasets_for_vocab_creation: Optional[List[str]] = None, validation_dataset_reader: Optional[DatasetReader] = None, validation_data_path: Optional[str] = None, validation_data_loader: Optional[Lazy[DataLoader]] = None, test_data_path: Optional[str] = None, evaluate_on_test: bool = False, batch_weight_key: str = '', ddp_accelerator: Optional[DdpAccelerator] = None) -> TrainModel:
    ...
