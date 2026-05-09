from __future__ import annotations
import warnings
from typing import Any, Callable, Optional
from kedro import KedroDeprecationWarning
from kedro.io.core import AbstractDataset, DatasetError

class LambdaDataset(AbstractDataset):
    def _describe(self) -> dict[str, Optional[str]]:
        def _to_str(func: Callable) -> str:
            if not func:
                return None
            try:
                return f'<{func.__module__}.{func.__name__}>'
            except AttributeError:
                return str(func)
        descr = {'load': _to_str(self.__load), 'save': _to_str(self.__save), 'exists': _to_str(self.__exists), 'release': _to_str(self.__release)}
        return descr

    def _load(self) -> Any:
        if not self.__load:
            raise DatasetError("Cannot load dataset. No 'load' function provided when LambdaDataset was created.")
        return self.__load()

    def _save(self, data: Any) -> None:
        if not self.__save:
            raise DatasetError("Cannot save to dataset. No 'save' function provided when LambdaDataset was created.")
        self.__save(data)

    def _exists(self) -> bool:
        if not self.__exists:
            return super()._exists()
        return self.__exists()

    def _release(self) -> None:
        if not self.__release:
            super()._release()
        else:
            self.__release()

    def __init__(self, load: Callable, save: Callable, exists: Optional[Callable] = None, release: Optional[Callable] = None, metadata: Optional[Any] = None) -> None:
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
        warnings.warn('`LambdaDataset` has been deprecated and will be removed in Kedro 0.20.0.', KedroDeprecationWarning)
        for name, value in [('load', load), ('save', save), ('exists', exists), ('release', release)]:
            if value is not None and (not callable(value)):
                raise DatasetError(f"'{name}' function for LambdaDataset must be a Callable. Object of type '{value.__class__.__name__}' provided instead.")
        self.__load = load
        self.__save = save
        self.__exists = exists
        self.__release = release
        self.metadata = metadata