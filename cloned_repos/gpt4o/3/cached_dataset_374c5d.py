from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Union
from kedro.io.core import VERSIONED_FLAG_KEY, AbstractDataset, Version
from kedro.io.memory_dataset import MemoryDataset

class CachedDataset(AbstractDataset):
    _SINGLE_PROCESS = True

    def __init__(
        self,
        dataset: Union[AbstractDataset, Dict[str, Any]],
        version: Optional[Version] = None,
        copy_mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self._EPHEMERAL = True
        if isinstance(dataset, dict):
            self._dataset = self._from_config(dataset, version)
        elif isinstance(dataset, AbstractDataset):
            self._dataset = dataset
        else:
            raise ValueError("The argument type of 'dataset' should be either a dict/YAML representation of the dataset, or the actual dataset object.")
        self._cache = MemoryDataset(copy_mode=copy_mode)
        self.metadata = metadata

    def _release(self) -> None:
        self._cache.release()
        self._dataset.release()

    @staticmethod
    def _from_config(config: Dict[str, Any], version: Optional[Version]) -> AbstractDataset:
        if VERSIONED_FLAG_KEY in config:
            raise ValueError("Cached datasets should specify that they are versioned in the 'CachedDataset', not in the wrapped dataset.")
        if version:
            config[VERSIONED_FLAG_KEY] = True
            return AbstractDataset.from_config('_cached', config, version.load, version.save)
        return AbstractDataset.from_config('_cached', config)

    def _describe(self) -> Dict[str, Any]:
        return {'dataset': self._dataset._describe(), 'cache': self._cache._describe()}

    def __repr__(self) -> str:
        object_description = {'dataset': self._dataset._pretty_repr(self._dataset._describe()), 'cache': self._dataset._pretty_repr(self._cache._describe())}
        return self._pretty_repr(object_description)

    def load(self) -> Any:
        data = self._cache.load() if self._cache.exists() else self._dataset.load()
        if not self._cache.exists():
            self._cache.save(data)
        return data

    def save(self, data: Any) -> None:
        self._dataset.save(data)
        self._cache.save(data)

    def _exists(self) -> bool:
        return self._cache.exists() or self._dataset.exists()

    def __getstate__(self) -> Dict[str, Any]:
        logging.getLogger(__name__).warning('%s: clearing cache to pickle.', str(self))
        self._cache.release()
        return self.__dict__
