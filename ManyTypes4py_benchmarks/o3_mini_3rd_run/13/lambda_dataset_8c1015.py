from __future__ import annotations
import warnings
from typing import Any, Callable, Optional, Dict
from kedro import KedroDeprecationWarning
from kedro.io.core import AbstractDataset, DatasetError


class LambdaDataset(AbstractDataset):
    def _describe(self) -> Dict[str, Optional[str]]:
        def _to_str(func: Optional[Callable[..., Any]]) -> Optional[str]:
            if not func:
                return None
            try:
                return f'<{func.__module__}.{func.__name__}>'
            except AttributeError:
                return str(func)
        descr: Dict[str, Optional[str]] = {
            'load': _to_str(self.__load),
            'save': _to_str(self.__save),
            'exists': _to_str(self.__exists),
            'release': _to_str(self.__release)
        }
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

    def __init__(
        self,
        load: Optional[Callable[[], Any]],
        save: Optional[Callable[[Any], None]],
        exists: Optional[Callable[[], bool]] = None,
        release: Optional[Callable[[], None]] = None,
        metadata: Any = None,
    ) -> None:
        warnings.warn(
            '`LambdaDataset` has been deprecated and will be removed in Kedro 0.20.0.',
            KedroDeprecationWarning
        )
        for name, value in [('load', load), ('save', save), ('exists', exists), ('release', release)]:
            if value is not None and (not callable(value)):
                raise DatasetError(
                    f"'{name}' function for LambdaDataset must be a Callable. Object of type '{value.__class__.__name__}' provided instead."
                )
        self.__load = load
        self.__save = save
        self.__exists = exists
        self.__release = release
        self.metadata = metadata