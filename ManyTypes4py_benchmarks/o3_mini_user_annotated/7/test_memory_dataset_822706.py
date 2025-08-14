import re
from typing import Any, Union

import numpy as np
import pandas as pd
import pytest
from kedro_datasets.pandas import CSVDataset

from kedro.io import DatasetError, MemoryDataset
from kedro.io.memory_dataset import (
    _copy_with_mode,
    _infer_copy_mode,
    _is_memory_dataset,
)


def _update_data(
    data: Union[pd.DataFrame, np.ndarray, Any], idx: int, jdx: int, value: Any
) -> Union[pd.DataFrame, np.ndarray, Any]:
    if isinstance(data, pd.DataFrame):
        data.iloc[idx, jdx] = value
        return data
    if isinstance(data, np.ndarray):
        data[idx, jdx] = value
        return data
    return data  # pragma: no cover


def _check_equals(data1: Any, data2: Any) -> bool:
    if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
        return data1.equals(data2)
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        return np.array_equal(data1, data2)
    return False  # pragma: no cover


@pytest.fixture
def memory_dataset(input_data: Any) -> MemoryDataset:
    return MemoryDataset(data=input_data)


@pytest.fixture
def mocked_infer_mode(mocker: Any) -> Any:
    return mocker.patch("kedro.io.memory_dataset._infer_copy_mode")


@pytest.fixture
def mocked_copy_with_mode(mocker: Any) -> Any:
    return mocker.patch("kedro.io.memory_dataset._copy_with_mode")


class TestMemoryDataset:
    def test_load(self, memory_dataset: MemoryDataset, input_data: Any) -> None:
        """Test basic load"""
        loaded_data = memory_dataset.load()
        assert _check_equals(loaded_data, input_data)

    def test_load_none(self) -> None:
        loaded_data = MemoryDataset(None).load()
        assert loaded_data is None

    def test_ephemeral_attribute(self, memory_dataset: MemoryDataset) -> None:
        assert memory_dataset._EPHEMERAL is True

    def test_load_infer_mode(
        self,
        memory_dataset: MemoryDataset,
        input_data: Any,
        mocked_infer_mode: Any,
        mocked_copy_with_mode: Any,
    ) -> None:
        """Test load calls infer_mode and copy_mode_with"""
        memory_dataset.load()
        assert mocked_infer_mode.call_count == 1
        assert mocked_copy_with_mode.call_count == 1

        assert mocked_infer_mode.call_args
        assert mocked_infer_mode.call_args[0]
        assert _check_equals(mocked_infer_mode.call_args[0][0], input_data)
        assert mocked_copy_with_mode.call_args
        assert mocked_copy_with_mode.call_args[0]
        assert _check_equals(mocked_copy_with_mode.call_args[0][0], input_data)

    def test_save(self, memory_dataset: MemoryDataset, input_data: Any, new_data: Any) -> None:
        """Test overriding the dataset"""
        memory_dataset.save(data=new_data)
        reloaded = memory_dataset.load()
        assert not _check_equals(reloaded, input_data)
        assert _check_equals(reloaded, new_data)

    def test_save_infer_mode(
        self,
        memory_dataset: MemoryDataset,
        new_data: Any,
        mocked_infer_mode: Any,
        mocked_copy_with_mode: Any,
    ) -> None:
        """Test save calls infer_mode and copy_mode_with"""
        memory_dataset.save(data=new_data)
        assert mocked_infer_mode.call_count == 1
        assert mocked_copy_with_mode.call_count == 1

        assert mocked_infer_mode.call_args
        assert mocked_infer_mode.call_args[0]
        assert _check_equals(mocked_infer_mode.call_args[0][0], new_data)
        assert mocked_copy_with_mode.call_args
        assert mocked_copy_with_mode.call_args[0]
        assert _check_equals(mocked_copy_with_mode.call_args[0][0], new_data)

    def test_load_modify_original_data(self, memory_dataset: MemoryDataset, input_data: Any) -> None:
        """Check that the dataset object is not updated when the original
        object is changed."""
        input_data = _update_data(input_data, 1, 1, -5)
        assert not _check_equals(memory_dataset.load(), input_data)

    def test_save_modify_original_data(self, memory_dataset: MemoryDataset, new_data: Any) -> None:
        """Check that the dataset object is not updated when the original
        object is changed."""
        memory_dataset.save(new_data)
        new_data = _update_data(new_data, 1, 1, "new value")
        assert not _check_equals(memory_dataset.load(), new_data)

    @pytest.mark.parametrize(
        "input_data", ["dummy_dataframe", "dummy_numpy_array"], indirect=True
    )
    def test_load_returns_new_object(self, memory_dataset: MemoryDataset, input_data: Any) -> None:
        """Test that consecutive loads point to different objects in case of a
        pandas DataFrame and numpy array"""
        loaded_data = memory_dataset.load()
        reloaded_data = memory_dataset.load()
        assert _check_equals(loaded_data, input_data)
        assert _check_equals(reloaded_data, input_data)
        assert loaded_data is not reloaded_data

    def test_create_without_data(self) -> None:
        """Test instantiation without data"""
        assert MemoryDataset() is not None

    def test_loading_none(self) -> None:
        """Check the error when attempting to load the dataset that doesn't
        contain any data"""
        pattern = r"Data for MemoryDataset has not been saved yet\."
        with pytest.raises(DatasetError, match=pattern):
            MemoryDataset().load()

    def test_saving_none(self) -> None:
        """Check the error when attempting to save the dataset without
        providing the data"""
        pattern = r"Saving 'None' to a 'Dataset' is not allowed"
        with pytest.raises(DatasetError, match=pattern):
            MemoryDataset().save(None)

    @pytest.mark.parametrize(
        "input_data,expected",
        [
            (
                "dummy_dataframe",
                "MemoryDataset(data=<DataFrame>)",
            ),
            (
                "dummy_numpy_array",
                "MemoryDataset(data=<ndarray>)",
            ),
        ],
        indirect=["input_data"],
    )
    def test_str_representation(self, memory_dataset: MemoryDataset, input_data: Any, expected: str) -> None:
        """Test string representation of the dataset"""
        assert expected in str(memory_dataset)

    @pytest.mark.parametrize(
        "input_data,expected",
        [
            (
                "dummy_dataframe",
                "kedro.io.memory_dataset.MemoryDataset(data='<DataFrame>')",
            ),
            (
                "dummy_numpy_array",
                "kedro.io.memory_dataset.MemoryDataset(data='<ndarray>')",
            ),
        ],
        indirect=["input_data"],
    )
    def test_repr_representation(self, memory_dataset: MemoryDataset, input_data: Any, expected: str) -> None:
        """Test string representation of the dataset"""
        assert expected in repr(memory_dataset)

    def test_exists(self, new_data: Any) -> None:
        """Test `exists` method invocation"""
        dataset = MemoryDataset()
        assert not dataset.exists()

        dataset.save(new_data)
        assert dataset.exists()


@pytest.mark.parametrize("data", [["a", "b"], [{"a": "b"}, {"c": "d"}]])
def test_copy_mode_assign(data: Any) -> None:
    """Test _copy_with_mode with assign"""
    copied_data = _copy_with_mode(data, copy_mode="assign")
    assert copied_data is data


@pytest.mark.parametrize("data", [[{"a": "b"}], [["a"]]])
def test_copy_mode_copy(data: Any) -> None:
    """Test _copy_with_mode with copy"""
    copied_data = _copy_with_mode(data, copy_mode="copy")
    assert copied_data is not data
    assert copied_data == data
    assert copied_data[0] is data[0]


@pytest.mark.parametrize("data", [[{"a": "b"}], [["a"]]])
def test_copy_mode_deepcopy(data: Any) -> None:
    """Test _copy_with_mode with deepcopy"""
    copied_data = _copy_with_mode(data, copy_mode="deepcopy")
    assert copied_data is not data
    assert copied_data == data
    assert copied_data[0] is not data[0]


def test_copy_mode_invalid_string() -> None:
    """Test _copy_with_mode with invalid string"""
    pattern = "Invalid copy mode: alice. Possible values are: deepcopy, copy, assign."
    with pytest.raises(DatasetError, match=re.escape(pattern)):
        _copy_with_mode(None, copy_mode="alice")


def test_infer_mode_copy(input_data: Any) -> None:
    copy_mode = _infer_copy_mode(input_data)
    assert copy_mode == "copy"


@pytest.mark.parametrize("data", [["a", "b"], [["a", "b"]], {"a": "b"}, [{"a": "b"}]])
def test_infer_mode_deepcopy(data: Any) -> None:
    copy_mode = _infer_copy_mode(data)
    assert copy_mode == "deepcopy"


def test_infer_mode_assign() -> None:
    class DataFrame:
        pass

    data = DataFrame()
    copy_mode = _infer_copy_mode(data)
    assert copy_mode == "assign"


@pytest.mark.parametrize(
    "ds_or_type,expected_result",
    [
        ("MemoryDataset", True),
        ("kedro.io.memory_dataset.MemoryDataset", True),
        ("NotMemoryDataset", False),
        (MemoryDataset(data=""), True),
        (CSVDataset(filepath="abc.csv"), False),
    ],
)
def test_is_memory_dataset(ds_or_type: Any, expected_result: bool) -> None:
    assert _is_memory_dataset(ds_or_type) == expected_result
