import datetime
import os
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Type, TypeVar, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field
from pydantic.config import JsonDict
from typing_extensions import Self
from prefect.types._datetime import DateTime, human_friendly_diff

if TYPE_CHECKING:
    from pydantic.main import IncEx
    from rich.repr import RichReprResult

T = TypeVar('T')
B = TypeVar('B', bound=BaseModel)

def get_class_fields_only(model: Type[BaseModel]) -> Set[str]:
    """
    Gets all the field names defined on the model class but not any parent classes.
    Any fields that are on the parent but redefined on the subclass are included.
    """
    return set(model.__annotations__)

class PrefectDescriptorBase(ABC):
    """A base class for descriptor objects used with PrefectBaseModel"""

    @abstractmethod
    def __get__(self, __instance: Optional[Any], __owner: Optional[Type[Any]] = None) -> Any:
        """Base descriptor access."""
        if __instance is not None:
            raise AttributeError
        return self

class PrefectBaseModel(BaseModel):
    """A base pydantic.BaseModel for all Prefect schemas and pydantic models."""
    _reset_fields: ClassVar[Set[str]] = set()
    model_config: ClassVar[ConfigDict] = ConfigDict(
        ser_json_timedelta='float',
        extra='ignore' if os.getenv('PREFECT_TEST_MODE', '0').lower() not in ['true', '1'] and os.getenv('PREFECT_TESTING_TEST_MODE', '0').lower() not in ['true', '1'] else 'forbid',
        ignored_types=(PrefectDescriptorBase,)
    )

    def __eq__(self, other: Any) -> bool:
        """Equality operator that ignores the resettable fields of the PrefectBaseModel."""
        copy_dict = self.model_dump(exclude=self._reset_fields)
        if isinstance(other, PrefectBaseModel):
            return copy_dict == other.model_dump(exclude=other._reset_fields)
        if isinstance(other, BaseModel):
            return copy_dict == other.model_dump()
        else:
            return copy_dict == other

    def __rich_repr__(self) -> 'RichReprResult':
        for name, field in self.model_fields.items():
            value = getattr(self, name)
            if isinstance(value, UUID):
                value = str(value)
            elif isinstance(value, datetime.datetime):
                value = value.isoformat() if name == 'timestamp' else human_friendly_diff(value)
            yield (name, value, field.get_default())

    def reset_fields(self) -> Self:
        """
        Reset the fields of the model that are in the `_reset_fields` set.
        """
        return self.model_copy(
            update={field: self.model_fields[field].get_default(call_default_factory=True) 
                   for field in self._reset_fields}
        )

    def model_dump_for_orm(
        self,
        *,
        include: Optional[Union[Set[str], Dict[str, Any]]] = None,
        exclude: Optional[Union[Set[str], Dict[str, Any]]] = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False
    ) -> Dict[str, Any]:
        """
        Prefect extension to `BaseModel.model_dump`.
        """
        deep = self.model_dump(
            mode='python',
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            context={'for_orm': True}
        )
        for k, v in self:
            if k in deep and isinstance(v, BaseModel):
                deep[k] = v
        return deep

def _ensure_fields_required(field_names: List[str], schema: Dict[str, Any]) -> None:
    for field_name in field_names:
        if 'required' not in schema:
            schema['required'] = []
        if (required := schema.get('required')) and isinstance(required, list) and (field_name not in required):
            required.append(field_name)

class IDBaseModel(PrefectBaseModel):
    """
    A PrefectBaseModel with an auto-generated UUID ID value.
    """
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra=partial(_ensure_fields_required, ['id'])
    )
    _reset_fields: ClassVar[Set[str]] = {'id'}
    id: UUID = Field(default_factory=uuid4)

class ORMBaseModel(IDBaseModel):
    """
    A PrefectBaseModel with an auto-generated UUID ID value and created /
    updated timestamps.
    """
    _reset_fields: ClassVar[Set[str]] = {'id', 'created', 'updated'}
    model_config: ClassVar[ConfigDict] = ConfigDict(
        from_attributes=True,
        json_schema_extra=partial(_ensure_fields_required, ['id', 'created', 'updated'])
    )
    created: Optional[DateTime] = Field(default=None, repr=False)
    updated: Optional[DateTime] = Field(default=None, repr=False)

class ActionBaseModel(PrefectBaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid')
