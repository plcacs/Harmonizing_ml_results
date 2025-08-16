from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from .utils import list_to_tensor

XDict = Dict[str, Any]
YDict = Dict[str, Tensor]
Batch = Tuple[XDict, YDict]
DEFAULT_INPUT_DATA_KEY: str = 'input_data'
DEFAULT_DATASET_NAME: str = 'SnorkelDataset'
DEFAULT_TASK_NAME: str = 'task'

class DictDataset(Dataset):
    def __init__(self, name: str, split: str, X_dict: XDict, Y_dict: YDict) -> None:
    
    def __getitem__(self, index: int) -> Batch:
    
    def __len__(self) -> int:
    
    def __repr__(self) -> str:
    
    @classmethod
    def from_tensors(cls, X_tensor: Tensor, Y_tensor: Tensor, split: str, input_data_key: str = DEFAULT_INPUT_DATA_KEY, task_name: str = DEFAULT_TASK_NAME, dataset_name: str = DEFAULT_DATASET_NAME) -> 'DictDataset':
    
def collate_dicts(batch: List[Batch]) -> Batch:

class DictDataLoader(DataLoader):
    def __init__(self, dataset: DictDataset, collate_fn: Callable = collate_dicts, **kwargs: Any) -> None:
