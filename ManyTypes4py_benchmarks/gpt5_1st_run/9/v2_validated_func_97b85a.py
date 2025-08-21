from __future__ import annotations

"""
This module contains an implementation of pydantic v1's ValidateFunction
modified to validate function arguments and return a pydantic v2 model.

Specifically it allows for us to validate v2 models used as flow/task
arguments.
"""
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union
from pydantic import BaseModel, ConfigDict, create_model, field_validator
from pydantic.v1.decorator import ValidatedFunction
from pydantic.v1.errors import ConfigError
from pydantic.v1.utils import to_camel
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    ConfigType: TypeAlias = Union[None, type[Any], dict[str, Any]]

V_POSITIONAL_ONLY_NAME: str = 'v__positional_only'
V_DUPLICATE_KWARGS: str = 'v__duplicate_kwargs'


class V2ValidatedFunction(ValidatedFunction):
    model: Optional[type[BaseModel]]

    def create_model(
        self,
        fields: dict[str, tuple[Any, Any]],
        takes_args: bool,
        takes_kwargs: bool,
        config: ConfigType,
    ) -> None:
        pos_args = len(self.arg_mapping)

        if config is None:
            cfg: dict[str, Any] = {}
        else:
            if not isinstance(config, dict):
                raise TypeError(f'config must be None or a dict, got {type(config)}')
            cfg = dict(config)

        if cfg.get('fields') or cfg.get('alias_generator'):
            raise ConfigError(
                'Setting the "fields" and "alias_generator" property on custom Config for @validate_arguments is not yet supported, please remove.'
            )
        if 'extra' not in cfg:
            cfg['extra'] = 'forbid'

        class DecoratorBaseModel(BaseModel):
            model_config: ClassVar[ConfigDict] = ConfigDict(**cfg)

            @field_validator(self.v_args_name, check_fields=False)
            @classmethod
            def check_args(cls, v: Optional[tuple[Any, ...]]) -> Optional[tuple[Any, ...]]:
                if takes_args or v is None:
                    return v
                raise TypeError(f'{pos_args} positional arguments expected but {pos_args + len(v)} given')

            @field_validator(self.v_kwargs_name, check_fields=False)
            @classmethod
            def check_kwargs(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
                if takes_kwargs or v is None:
                    return v
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v.keys()))
                raise TypeError(f'unexpected keyword argument{plural}: {keys}')

            @field_validator(V_POSITIONAL_ONLY_NAME, check_fields=False)
            @classmethod
            def check_positional_only(
                cls, v: Optional[Union[list[str], tuple[str, ...]]]
            ) -> Optional[Union[list[str], tuple[str, ...]]]:
                if v is None:
                    return v
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v))
                raise TypeError(f'positional-only argument{plural} passed as keyword argument{plural}: {keys}')

            @field_validator(V_DUPLICATE_KWARGS, check_fields=False)
            @classmethod
            def check_duplicate_kwargs(
                cls, v: Optional[Union[list[str], tuple[str, ...]]]
            ) -> Optional[Union[list[str], tuple[str, ...]]]:
                if v is None:
                    return v
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v))
                raise TypeError(f'multiple values for argument{plural}: {keys}')

        self.model = create_model(
            to_camel(self.raw_function.__name__),
            __base__=DecoratorBaseModel,
            **fields,
        )