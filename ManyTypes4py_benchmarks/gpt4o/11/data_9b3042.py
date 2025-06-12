from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from .utils import list_to_tensor

XDict = Dict[str, Any]
YDict = Dict[str, Tensor]
Batch = Tuple[XDict, YDict]
DEFAULT_INPUT_DATA_KEY = 'input_data'
DEFAULT_DATASET_NAME = 'SnorkelDataset'
DEFAULT_TASK_NAME = 'task'

class DictDataset(Dataset):
    def __init__(self, name: str, split: str, X_dict: XDict, Y_dict: YDict) -> None:
        self.name = name
        self.split = split
        self.X_dict = X_dict
        self.Y_dict = Y_dict
        for name, label in self.Y_dict.items():
            if not isinstance(label, Tensor):
                raise ValueError(f'Label {name} should be torch.Tensor, not {type(label)}.')

    def __getitem__(self, index: int) -> Batch:
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return (x_dict, y_dict)

    def __len__(self) -> int:
        try:
            return len(next(iter(self.Y_dict.values())))
        except StopIteration:
            return 0

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name}, X_keys={list(self.X_dict.keys())}, Y_keys={list(self.Y_dict.keys())})'

    @classmethod
    def from_tensors(cls, X_tensor: Tensor, Y_tensor: Tensor, split: str, input_data_key: str = DEFAULT_INPUT_DATA_KEY, task_name: str = DEFAULT_TASK_NAME, dataset_name: str = DEFAULT_DATASET_NAME) -> 'DictDataset':
        return cls(name=dataset_name, split=split, X_dict={input_data_key: X_tensor}, Y_dict={task_name: Y_tensor})

def collate_dicts(batch: List[Batch]) -> Batch:
    X_batch = defaultdict(list)
    Y_batch = defaultdict(list)
    for x_dict, y_dict in batch:
        for field_name, value in x_dict.items():
            X_batch[field_name].append(value)
        for label_name, value in y_dict.items():
            Y_batch[label_name].append(value)
    for field_name, values in X_batch.items():
        if isinstance(values[0], Tensor):
            X_batch[field_name] = list_to_tensor(values)
    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(values)
    return (dict(X_batch), dict(Y_batch))

class DictDataLoader(DataLoader):
    def __init__(self, dataset: DictDataset, collate_fn: Callable[[List[Batch]], Batch] = collate_dicts, **kwargs: Any) -> None:
        assert isinstance(dataset, DictDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
