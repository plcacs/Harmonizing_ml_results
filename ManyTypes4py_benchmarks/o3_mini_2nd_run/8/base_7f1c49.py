"""Base data provider."""
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

__all__ = ['BaseDataProvider', 'BaseProvider']


class BaseProvider:
    """This is a base class for all providers.

    :attr random: An instance of :class:`mimesis.random.Random`.
    :attr seed: Seed for random.
    """

    class Meta:
        pass

    def __init__(self, *, seed: t.Union[Seed, type(MissingSeed)] = MissingSeed,
                 random: t.Optional[_random.Random] = None) -> None:
        """Initialize attributes.

        Keep in mind that locale-independent data providers will work
        only with keyword-only arguments.

        :param seed: Seed for random.
            When set to `None` the current system time is used.
        :param random: Custom random.
            See https://github.com/lk-geimfari/mimesis/issues/1313 for details.
        """
        if random is not None:
            if not isinstance(random, _random.Random):
                raise TypeError('The random must be an instance of mimesis.random.Random')
            self.random: _random.Random = random
        else:
            self.random = _random.Random()
        self.seed: t.Union[Seed, type(MissingSeed)] = seed
        self.reseed(seed)

    def reseed(self, seed: t.Union[Seed, type(MissingSeed)] = MissingSeed) -> None:
        """Reseeds the internal random generator.

        In case we use the default seed, we need to create a per instance
        random generator. In this case, two providers with the same seed
        will always return the same values.

        :param seed: Seed for random.
            When set to `None` the current system time is used.
        """
        self.seed = seed
        if seed is MissingSeed:
            if _random.global_seed is not MissingSeed:
                self.random.seed(t.cast(t.Any, _random.global_seed))
        else:
            self.random.seed(t.cast(t.Any, seed))

    def validate_enum(self, item: t.Any, enum: t.Any) -> t.Any:
        """Validates various enum objects that are used as arguments for methods.

        :param item: Item of an enum object.
        :param enum: Enum object.
        :return: Value of item.
        :raises NonEnumerableError: If enums has not such an item.
        """
        if item is None:
            result = self.random.choice_enum_item(enum)
        elif item and isinstance(item, enum):
            result = item
        else:
            raise NonEnumerableError(enum)
        return result.value

    def _read_global_file(self, file_name: str) -> JSON:
        """Reads JSON file and return dict.

        Reads JSON file from mimesis/data/global/ directory.

        :param file_name: Path to file.
        :raises FileNotFoundError: If the file was not found.
        :return: JSON data.
        """
        with open(DATADIR.joinpath('global', file_name), encoding='utf8') as f:
            data: JSON = json.load(f)
        return data

    def _has_seed(self) -> bool:
        """Internal API to check if seed is set."""
        return (self.seed is not None and self.seed is not MissingSeed or
                (_random.global_seed is not None and _random.global_seed is not MissingSeed))

    def __str__(self) -> str:
        """Human-readable representation of locale."""
        return self.__class__.__name__


class BaseDataProvider(BaseProvider):
    """This is a base class for all data providers."""

    def __init__(
        self,
        locale: str = Locale.DEFAULT,
        seed: t.Union[Seed, type(MissingSeed)] = MissingSeed,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> None:
        """Initialize attributes for data providers.

        :param locale: Current locale.
        :param seed: Seed to all the random functions.
        """
        super().__init__(*args, seed=seed, **kwargs)
        self._dataset: JSON = {}
        self._setup_locale(locale)
        self._load_dataset()

    def _setup_locale(self, locale: str = Locale.DEFAULT) -> None:
        """Set up locale after pre-check.

        :param locale: Locale
        :raises UnsupportedLocale: When locale not supported.
        :return: Nothing.
        """
        locale_obj = validate_locale(locale)
        self.locale: str = locale_obj.value

    def _extract(self, keys: t.Sequence[t.Hashable], default: t.Any = None) -> t.Any:
        """Extracts nested values from JSON file by list of keys.

        :param keys: List of keys (order extremely matters).
        :param default: Default value.
        :return: Data.
        """
        if not keys:
            raise ValueError('The list of keys to extract cannot be empty.')
        try:
            return reduce(operator.getitem, keys, self._dataset)
        except (TypeError, KeyError):
            return default

    def _update_dict(self, initial: t.Dict[t.Any, t.Any], other: t.Dict[t.Any, t.Any]) -> t.Dict[t.Any, t.Any]:
        """Recursively updates a dictionary.

        :param initial: Dict to update.
        :param other: Dict to update from.
        :return: Updated dict.
        """
        for k, v in other.items():
            if isinstance(v, dict):
                initial[k] = self._update_dict(initial.get(k, {}), v)
            else:
                initial[k] = other[k]
        return initial

    def _load_dataset(self) -> None:
        """Loads the content from the JSON dataset.

        :return: The content of the file.
        :raises UnsupportedLocale: Raises if locale is unsupported.
        """
        locale: str = self.locale
        datafile: str = getattr(self.Meta, 'datafile', '')
        datadir = getattr(self.Meta, 'datadir', DATADIR)
        if not datafile:
            return

        def read_file(locale_name: str) -> JSON:
            file_path = datadir / locale_name / datafile
            with open(file_path, encoding='utf8') as f:
                return json.load(f)

        master_locale: str = locale.split(LOCALE_SEP).pop(0)
        data: JSON = read_file(master_locale)
        if LOCALE_SEP in locale:
            data = self._update_dict(data, read_file(locale))
        self._dataset = data

    def update_dataset(self, data: t.Dict[str, t.Any]) -> None:
        """Updates dataset merging a given dict into default data.

        This method may be useful when you need to override data
        for a given key in JSON file.
        """
        if not isinstance(data, dict):
            raise TypeError('The data must be a dict.')
        self._dataset |= data

    def get_current_locale(self) -> str:
        """Returns current locale.

        If locale is not defined, then this method will always return ``en``,
        because ``en`` is default locale for all providers, excluding builtins.

        :return: Current locale.
        """
        return self.locale

    def _override_locale(self, locale: str = Locale.DEFAULT) -> None:
        """Overrides current locale with passed and pull data for new locale.

        :param locale: Locale
        :return: Nothing.
        """
        self._setup_locale(locale)
        self._load_dataset()

    @contextlib.contextmanager
    def override_locale(self, locale: str) -> t.Iterator["BaseDataProvider"]:
        """Context manager that allows overriding current locale.

        Temporarily overrides current locale for
        locale-dependent providers.

        :param locale: Locale.
        :return: Provider with overridden locale.
        """
        try:
            origin_locale = Locale(self.locale)
            self._override_locale(locale)
            try:
                yield self
            finally:
                self._override_locale(origin_locale)
        except AttributeError:
            raise ValueError(f'«{self.__class__.__name__}» has not locale dependent')

    def __str__(self) -> str:
        """Human-readable representation of locale."""
        locale_obj = Locale(getattr(self, 'locale', Locale.DEFAULT))
        return f'{self.__class__.__name__} <{locale_obj}>'
