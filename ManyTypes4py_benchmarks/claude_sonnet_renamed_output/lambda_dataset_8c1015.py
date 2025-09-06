"""``LambdaDataset`` is an implementation of ``AbstractDataset`` which allows for
providing custom load, save, and exists methods without extending
``AbstractDataset``.
"""
from __future__ import annotations
import warnings
from typing import Any, Callable, Optional, Dict
from kedro import KedroDeprecationWarning
from kedro.io.core import AbstractDataset, DatasetError


class LambdaDataset(AbstractDataset):
    """``LambdaDataset`` loads and saves data to a dataset.
    It relies on delegating to specific implementation such as csv, sql, etc.

    ``LambdaDataset`` class captures Exceptions while performing operations on
    composed ``Dataset`` implementations. The composed dataset is
    responsible for providing information on how to resolve the issue when
    possible. This information should be available through str(error).

    Example:
    ::

        >>> from kedro.io import LambdaDataset
        >>> import pandas as pd
        >>>
        >>> file_name = "test.csv"
        >>> def load() -> pd.DataFrame:
        >>>     raise FileNotFoundError("'{}' csv file not found."
        >>>                             .format(file_name))
        >>> dataset = LambdaDataset(load, None)
    """

    def func_cpm38zvo(self) -> Dict[str, Optional[str]]:
        def func_3vxgghnc(func: Optional[Callable]) -> Optional[str]:
            if not func:
                return None
            try:
                return f'<{func.__module__}.{func.__name__}>'
            except AttributeError:
                return str(func)

        descr: Dict[str, Optional[str]] = {
            'load': func_3vxgghnc(self.__load),
            'save': func_3vxgghnc(self.__save),
            'exists': func_3vxgghnc(self.__exists),
            'release': func_3vxgghnc(self.__release)
        }
        return descr

    def func_ksn1a7n1(self) -> Any:
        if not self.__load:
            raise DatasetError(
                "Cannot load dataset. No 'load' function provided when LambdaDataset was created."
            )
        return self.__load()

    def func_91v8789m(self, data: Any) -> None:
        if not self.__save:
            raise DatasetError(
                "Cannot save to dataset. No 'save' function provided when LambdaDataset was created."
            )
        self.__save(data)

    def func_siul4mqe(self) -> bool:
        if not self.__exists:
            return super()._exists()
        return self.__exists()

    def func_eu3ku3bn(self) -> None:
        if not self.__release:
            super()._release()
        else:
            self.__release()

    def __init__(
        self,
        load: Callable[[], Any],
        save: Callable[[Any], None],
        exists: Optional[Callable[[], bool]] = None,
        release: Optional[Callable[[], None]] = None,
        metadata: Optional[Any] = None
    ) -> None:
        """Creates a new instance of ``LambdaDataset`` with references to the
        required input/output dataset methods.

        Args:
            load: Method to load data from a dataset.
            save: Method to save data to a dataset.
            exists: Method to check whether output data already exists.
            release: Method to release any cached information.
            metadata: Any arbitrary metadata.
                This is ignored by Kedro, but may be consumed by users or external plugins.

        Raises:
            DatasetError: If a method is specified, but is not a Callable.

        """
        warnings.warn(
            '`LambdaDataset` has been deprecated and will be removed in Kedro 0.20.0.',
            KedroDeprecationWarning
        )
        for name, value in [('load', load), ('save', save), ('exists', exists), ('release', release)]:
            if value is not None and not callable(value):
                raise DatasetError(
                    f"'{name}' function for LambdaDataset must be a Callable. "
                    f"Object of type '{value.__class__.__name__}' provided instead."
                )
        self.__load: Optional[Callable[[], Any]] = load
        self.__save: Optional[Callable[[Any], None]] = save
        self.__exists: Optional[Callable[[], bool]] = exists
        self.__release: Optional[Callable[[], None]] = release
        self.metadata: Optional[Any] = metadata
