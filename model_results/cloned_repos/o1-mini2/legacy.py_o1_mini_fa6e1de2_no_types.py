import inspect
import os
from functools import cache
from typing import Any, Dict, Optional, Set, Type, get_args
from pydantic import AliasChoices
from pydantic_settings import BaseSettings
from typing_extensions import Self
from prefect.settings.base import PrefectBaseSettings
from prefect.settings.constants import _SECRET_TYPES
from prefect.settings.context import get_current_settings
from prefect.settings.models.root import Settings


class Setting:
    """Mimics the old Setting object for compatibility with existing code."""
    _name: str
    _default: Any
    _type: Type[Any]
    accessor: str

    def __init__(self, name, default, type_, accessor=None):
        self._name = name
        self._default = default
        self._type = type_
        if accessor is None:
            self.accessor = _env_var_to_accessor(name)
        else:
            self.accessor = accessor

    @property
    def name(self):
        return self._name

    @property
    def is_secret(self):
        if self._type in _SECRET_TYPES:
            return True
        for secret_type in _SECRET_TYPES:
            if secret_type in get_args(self._type):
                return True
        return False

    def default(self):
        return self._default

    def value(self):
        if (self.name == 'PREFECT_TEST_SETTING' or self.name ==
            'PREFECT_TESTING_TEST_SETTING'):
            if ('PREFECT_TEST_MODE' in os.environ or 
                'PREFECT_TESTING_TEST_MODE' in os.environ):
                return get_current_settings().testing.test_setting
            else:
                return None
        return self.value_from(get_current_settings())

    def value_from(self, settings):
        path: list[str] = self.accessor.split('.')
        current_value: Any = settings
        for key in path:
            current_value = getattr(current_value, key, None)
        if isinstance(current_value, _SECRET_TYPES):
            return current_value.get_secret_value()
        return current_value

    def __bool__(self):
        return bool(self.value())

    def __str__(self):
        return str(self.value())

    def __repr__(self):
        return f'<{self.name}: {self._type!r}>'

    def __eq__(self, __o):
        if isinstance(__o, Setting):
            return self.value() == __o.value()
        return self.value() == __o

    def __hash__(self):
        return hash((type(self), self.name))


def _env_var_to_accessor(env_var):
    """
    Convert an environment variable name to a settings accessor.
    """
    fields: Dict[str, Any] = _get_settings_fields(Settings)
    if (field := fields.get(env_var)) is not None:
        return field.accessor
    return env_var.replace('PREFECT_', '').lower()


@cache
def _get_valid_setting_names(cls):
    """
    A set of valid setting names, e.g. "PREFECT_API_URL" or "PREFECT_API_KEY".
    """
    settings_fields: Set[str] = set()
    for field_name, field in cls.model_fields.items():
        if inspect.isclass(field.annotation) and issubclass(field.
            annotation, PrefectBaseSettings):
            settings_fields.update(_get_valid_setting_names(field.annotation))
        elif field.validation_alias and isinstance(field.validation_alias,
            AliasChoices):
            for alias in field.validation_alias.choices:
                if not isinstance(alias, str):
                    continue
                settings_fields.add(alias.upper())
        else:
            env_prefix: str = cls.model_config.get('env_prefix', '')
            settings_fields.add(f'{env_prefix}{field_name.upper()}')
    return settings_fields


@cache
def _get_settings_fields(settings, accessor_prefix=None):
    """Get the settings fields for the settings object"""
    settings_fields: Dict[str, Setting] = {}
    for field_name, field in settings.model_fields.items():
        if inspect.isclass(field.annotation) and issubclass(field.
            annotation, PrefectBaseSettings):
            accessor: str = (field_name if accessor_prefix is None else
                f'{accessor_prefix}.{field_name}')
            nested_fields: Dict[str, Setting] = _get_settings_fields(field.
                annotation, accessor)
            settings_fields.update(nested_fields)
        else:
            accessor: str = (field_name if accessor_prefix is None else
                f'{accessor_prefix}.{field_name}')
            if field.validation_alias and isinstance(field.validation_alias,
                AliasChoices):
                for alias in field.validation_alias.choices:
                    if not isinstance(alias, str):
                        continue
                    setting = Setting(name=alias.upper(), default=field.
                        default, type_=field.annotation, accessor=accessor)
                    settings_fields[setting.name] = setting
                    settings_fields[setting.accessor] = setting
            else:
                env_prefix: str = settings.model_config.get('env_prefix', '')
                setting_name: str = f'{env_prefix}{field_name.upper()}'
                setting = Setting(name=setting_name, default=field.default,
                    type_=field.annotation, accessor=accessor)
                settings_fields[setting.name] = setting
                settings_fields[setting.accessor] = setting
    return settings_fields
