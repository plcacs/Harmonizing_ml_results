import contextlib
import json
import operator
import typing as t
from functools import reduce
from mimesis import random as _random
from mimesis.constants import DATADIR, LOCALE_SEP
from mimesis.exceptions import NonEnumerableError
from mimesis.locales import Locale, validate_locale
from mimesis.types import JSON, MissingSeed, Seed

__all__: t.List[str] = ['BaseDataProvider', 'BaseProvider']

class BaseProvider:
    random: _random.Random
    seed: Seed

    class Meta:
        pass

    def __init__(self, *, seed: Seed = MissingSeed, random: t.Optional[_random.Random] = None) -> None:
        ...

    def reseed(self, seed: Seed = MissingSeed) -> None:
        ...

    def validate_enum(self, item: t.Any, enum: t.Type[t.Any]) -> t.Any:
        ...

    def _read_global_file(self, file_name: str) -> JSON:
        ...

    def _has_seed(self) -> bool:
        ...

    def __str__(self) -> str:
        ...

class BaseDataProvider(BaseProvider):
    locale: str
    _dataset: JSON

    def __init__(self, locale: str = Locale.DEFAULT, seed: Seed = MissingSeed, *args: t.Any, **kwargs: t.Any) -> None:
        ...

    def _setup_locale(self, locale: str = Locale.DEFAULT) -> None:
        ...

    def _extract(self, keys: t.List[str], default: t.Any = None) -> t.Any:
        ...

    def _update_dict(self, initial: dict, other: dict) -> dict:
        ...

    def _load_dataset(self) -> None:
        ...

    def update_dataset(self, data: dict) -> None:
        ...

    def get_current_locale(self) -> str:
        ...

    def _override_locale(self, locale: str = Locale.DEFAULT) -> None:
        ...

    @contextlib.contextmanager
    def override_locale(self, locale: str) -> t.Iterator['BaseDataProvider']:
        ...

    def __str__(self) -> str:
        ...
