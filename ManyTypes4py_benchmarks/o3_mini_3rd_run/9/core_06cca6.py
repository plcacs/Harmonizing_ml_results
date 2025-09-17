from itertools import chain
from typing import DefaultDict, Dict, List, NamedTuple, Tuple, Union, Any
import numpy as np
from tqdm import tqdm
from snorkel.labeling.lf import LabelingFunction
from snorkel.types import DataPoint, DataPoints
from snorkel.utils.data_operators import check_unique_names

RowData = List[Tuple[int, int, int]]


class ApplierMetadata(NamedTuple):
    fault_counts: DefaultDict[str, int]


class _FunctionCaller:
    def __init__(self, fault_tolerant: bool) -> None:
        self.fault_tolerant: bool = fault_tolerant
        self.fault_counts: DefaultDict[str, int] = DefaultDict(int)

    def __call__(self, f: LabelingFunction, x: DataPoint) -> int:
        if not self.fault_tolerant:
            return f(x)
        try:
            return f(x)
        except Exception:
            self.fault_counts[f.name] += 1
            return -1


class BaseLFApplier:
    _use_recarray: bool = False

    def __init__(self, lfs: List[LabelingFunction]) -> None:
        self._lfs: List[LabelingFunction] = lfs
        self._lf_names: List[str] = [lf.name for lf in lfs]
        check_unique_names(self._lf_names)

    def _numpy_from_row_data(self, labels: List[RowData]) -> np.ndarray:
        L: np.ndarray = np.zeros((len(labels), len(self._lfs)), dtype=int) - 1
        if any(map(len, labels)):
            row, col, data = zip(*chain.from_iterable(labels))
            L[row, col] = data
        if self._use_recarray:
            n_rows, _ = L.shape
            dtype = [(name, np.int64) for name in self._lf_names]
            recarray = np.recarray(n_rows, dtype=dtype)
            for idx, name in enumerate(self._lf_names):
                recarray[name] = L[:, idx]
            return recarray
        else:
            return L

    def __repr__(self) -> str:
        return f'{type(self).__name__}, LFs: {self._lf_names}'


def apply_lfs_to_data_point(
    x: DataPoint, index: int, lfs: List[LabelingFunction], f_caller: _FunctionCaller
) -> RowData:
    labels: RowData = []
    for j, lf in enumerate(lfs):
        y: int = f_caller(lf, x)
        if y >= 0:
            labels.append((index, j, y))
    return labels


class LFApplier(BaseLFApplier):
    def apply(
        self,
        data_points: DataPoints,
        progress_bar: bool = True,
        fault_tolerant: bool = False,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ApplierMetadata]]:
        labels: List[RowData] = []
        f_caller: _FunctionCaller = _FunctionCaller(fault_tolerant)
        for i, x in tqdm(enumerate(data_points), disable=not progress_bar):
            labels.append(apply_lfs_to_data_point(x, i, self._lfs, f_caller))
        L: np.ndarray = self._numpy_from_row_data(labels)
        if return_meta:
            return (L, ApplierMetadata(f_caller.fault_counts))
        return L