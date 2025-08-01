#!/usr/bin/env python3
from typing import Any, List
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn
from snorkel.analysis import Scorer
from snorkel.classification import DictDataset
from snorkel.slicing import SFApplier, SliceAwareClassifier, slicing_function

@slicing_function()
def f(x: SimpleNamespace) -> bool:
    return x.num > 42

@slicing_function()
def g(x: SimpleNamespace) -> bool:
    return x.num > 10

sfs: List[Any] = [f, g]
DATA: List[int] = [3, 43, 12, 9, 3]

def create_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
    split: str,
    dataset_name: str,
    input_name: str,
    task_name: str
) -> DictDataset:
    return DictDataset(
        name=dataset_name, 
        split=split, 
        X_dict={input_name: X}, 
        Y_dict={task_name: Y}
    )

class SliceCombinerTest(unittest.TestCase):
    def setUp(self) -> None:
        data_points: List[SimpleNamespace] = [SimpleNamespace(num=num) for num in DATA]
        applier: SFApplier = SFApplier([f, g])
        self.S: Any = applier.apply(data_points, progress_bar=False)
        self.hidden_dim: int = 10
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(2, self.hidden_dim), 
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU()
        )
        self.data_name: str = 'test_data'
        self.task_name: str = 'test_task'
        self.X: torch.Tensor = torch.FloatTensor([(x, x) for x in DATA])
        self.Y: torch.Tensor = torch.LongTensor([int(i % 2 == 0) for i in range(len(DATA))])
        dataset_name: str = 'test_dataset'
        splits: List[str] = ['train', 'valid']
        self.datasets: List[DictDataset] = [
            create_dataset(self.X, self.Y, split, dataset_name, self.data_name, self.task_name)
            for split in splits
        ]
        self.slice_model: SliceAwareClassifier = SliceAwareClassifier(
            base_architecture=self.mlp, 
            head_dim=self.hidden_dim, 
            slice_names=[sf.name for sf in sfs], 
            input_data_key=self.data_name, 
            task_name=self.task_name, 
            scorer=Scorer(metrics=['f1'])
        )

    def test_slice_tasks(self) -> None:
        """Ensure that all the desired slice tasks are initialized."""
        expected_tasks: set = {
            'test_task', 
            'test_task_slice:base_pred', 
            'test_task_slice:base_ind', 
            'test_task_slice:f_pred', 
            'test_task_slice:f_ind', 
            'test_task_slice:g_pred', 
            'test_task_slice:g_ind'
        }
        self.assertEqual(self.slice_model.task_names, expected_tasks)

    def test_make_slice_dataloader(self) -> None:
        dataloader: Any = self.slice_model.make_slice_dataloader(dataset=self.datasets[0], S=self.S)
        Y_dict: dict = dataloader.dataset.Y_dict
        self.assertEqual(len(Y_dict), 7)
        self.assertIn('test_task', Y_dict)
        self.assertIn('test_task_slice:base_pred', Y_dict)
        self.assertIn('test_task_slice:base_ind', Y_dict)
        self.assertIn('test_task_slice:f_pred', Y_dict)
        self.assertIn('test_task_slice:f_ind', Y_dict)
        self.assertIn('test_task_slice:g_pred', Y_dict)
        self.assertIn('test_task_slice:g_ind', Y_dict)
        bad_data_dataset: DictDataset = DictDataset(
            name='test_data', 
            split='train', 
            X_dict={self.data_name: self.X}, 
            Y_dict={'bad_labels': self.Y}
        )
        with self.assertRaisesRegex(ValueError, 'labels missing'):
            self.slice_model.make_slice_dataloader(dataset=bad_data_dataset, S=self.S)

    def test_scores_pipeline(self) -> None:
        """Ensure that the appropriate scores are returned with .score and .score_slices."""
        valid_dl: Any = self.slice_model.make_slice_dataloader(dataset=self.datasets[1], S=self.S, batch_size=4)
        scores: dict = self.slice_model.score([valid_dl])
        self.assertIn('test_task/test_dataset/valid/f1', scores)
        self.assertIn('test_task_slice:f_pred/test_dataset/valid/f1', scores)
        self.assertIn('test_task_slice:f_pred/test_dataset/valid/f1', scores)
        self.assertIn('test_task_slice:g_ind/test_dataset/valid/f1', scores)
        self.assertIn('test_task_slice:g_ind/test_dataset/valid/f1', scores)
        slice_scores: dict = self.slice_model.score_slices([valid_dl])
        self.assertIn('test_task/test_dataset/valid/f1', slice_scores)
        self.assertIn('test_task_slice:f_pred/test_dataset/valid/f1', slice_scores)
        self.assertIn('test_task_slice:g_pred/test_dataset/valid/f1', slice_scores)
        self.assertNotIn('test_task_slice:f_ind/test_dataset/valid/f1', slice_scores)
        self.assertNotIn('test_task_slice:g_ind/test_dataset/valid/f1', slice_scores)
