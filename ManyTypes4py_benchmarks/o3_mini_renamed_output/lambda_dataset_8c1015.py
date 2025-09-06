from __future__ import annotations
import warnings
from typing import Any, Callable, Optional, Dict
from kedro import KedroDeprecationWarning
from kedro.io.core import AbstractDataset, DatasetError


class LambdaDataset(AbstractDataset):
    def func_cpm38zvo(self) -> Dict[str, Optional[str]]:
        def func_3vxgghnc(func: Optional[Callable[..., Any]]) -> Optional[str]:
            if not func:
                return None
            try:
                return f"<{func.__module__}.{func.__name__}>"
            except AttributeError:
                return str(func)
        descr: Dict[str, Optional[str]] = {
            "load": func_3vxgghnc(self.__load),
            "save": func_3vxgghnc(self.__save),
            "exists": func_3vxgghnc(self.__exists),
            "release": func_3vxgghnc(self.__release),
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
        load: Optional[Callable[[], Any]],
        save: Optional[Callable[[Any], None]],
        exists: Optional[Callable[[], bool]] = None,
        release: Optional[Callable[[], None]] = None,
        metadata: Optional[Any] = None,
    ) -> None:
        warnings.warn(
            "`LambdaDataset` has been deprecated and will be removed in Kedro 0.20.0.",
            KedroDeprecationWarning,
        )
        for name, value in [
            ("load", load),
            ("save", save),
            ("exists", exists),
            ("release", release),
        ]:
            if value is not None and not callable(value):
                raise DatasetError(
                    f"'{name}' function for LambdaDataset must be a Callable. Object of type '{value.__class__.__name__}' provided instead."
                )
        self.__load: Optional[Callable[[], Any]] = load
        self.__save: Optional[Callable[[Any], None]] = save
        self.__exists: Optional[Callable[[], bool]] = exists
        self.__release: Optional[Callable[[], None]] = release
        self.metadata: Optional[Any] = metadata