from typing import List, Dict
import torch
import torch.nn as nn
from snorkel.analysis import Scorer
from snorkel.classification import DictDataset
from snorkel.slicing import SFApplier, SliceAwareClassifier, slicing_function

def create_dataset(X: torch.Tensor, Y: torch.Tensor, split: str, dataset_name: str, input_name: str, task_name: str) -> DictDataset:

    def f(x: SimpleNamespace) -> bool:
        return x.num > 42

    def g(x: SimpleNamespace) -> bool:
        return x.num > 10

    sfs: List[Callable[[SimpleNamespace], bool]] = [f, g]
    DATA: List[int] = [3, 43, 12, 9, 3]

    data_points = [SimpleNamespace(num=num) for num in DATA]
    applier = SFApplier(sfs)
    S = applier.apply(data_points, progress_bar=False)
    hidden_dim: int = 10
    mlp = nn.Sequential(nn.Linear(2, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
    data_name: str = 'test_data'
    task_name: str = 'test_task'
    X = torch.FloatTensor([(x, x) for x in DATA])
    Y = torch.LongTensor([int(i % 2 == 0) for i in range(len(DATA))])
    dataset_name: str = 'test_dataset'
    splits: List[str] = ['train', 'valid']
    datasets: List[DictDataset] = [create_dataset(X, Y, split, dataset_name, data_name, task_name) for split in splits]
    slice_model = SliceAwareClassifier(base_architecture=mlp, head_dim=hidden_dim, slice_names=[sf.__name__ for sf in sfs], input_data_key=data_name, task_name=task_name, scorer=Scorer(metrics=['f1']))

    expected_tasks: set = {'test_task', 'test_task_slice:base_pred', 'test_task_slice:base_ind', 'test_task_slice:f_pred', 'test_task_slice:f_ind', 'test_task_slice:g_pred', 'test_task_slice:g_ind'}
    assert slice_model.task_names == expected_tasks

    dataloader = slice_model.make_slice_dataloader(dataset=datasets[0], S=S)
    Y_dict = dataloader.dataset.Y_dict
    assert len(Y_dict) == 7
    assert 'test_task' in Y_dict
    assert 'test_task_slice:base_pred' in Y_dict
    assert 'test_task_slice:base_ind' in Y_dict
    assert 'test_task_slice:f_pred' in Y_dict
    assert 'test_task_slice:f_ind' in Y_dict
    assert 'test_task_slice:g_pred' in Y_dict
    assert 'test_task_slice:g_ind' in Y_dict

    bad_data_dataset = DictDataset(name='test_data', split='train', X_dict={data_name: X}, Y_dict={'bad_labels': Y})
    with self.assertRaisesRegex(ValueError, 'labels missing'):
        slice_model.make_slice_dataloader(dataset=bad_data_dataset, S=S)

    valid_dl = slice_model.make_slice_dataloader(dataset=datasets[1], S=S, batch_size=4)
    scores = slice_model.score([valid_dl])
    assert 'test_task/test_dataset/valid/f1' in scores
    assert 'test_task_slice:f_pred/test_dataset/valid/f1' in scores
    assert 'test_task_slice:f_pred/test_dataset/valid/f1' in scores
    assert 'test_task_slice:g_ind/test_dataset/valid/f1' in scores
    assert 'test_task_slice:g_ind/test_dataset/valid/f1' in scores
    slice_scores = slice_model.score_slices([valid_dl])
    assert 'test_task/test_dataset/valid/f1' in slice_scores
    assert 'test_task_slice:f_pred/test_dataset/valid/f1' in slice_scores
    assert 'test_task_slice:g_pred/test_dataset/valid/f1' in slice_scores
    assert 'test_task_slice:f_ind/test_dataset/valid/f1' not in slice_scores
    assert 'test_task_slice:g_ind/test_dataset/valid/f1' not in slice_scores
