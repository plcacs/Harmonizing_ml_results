import inspect
import sys
from datetime import datetime
from decimal import Decimal, DecimalTuple
from functools import lru_cache
from operator import attrgetter
from typing import Any, Callable, Iterable, Mapping, Optional, Set, Tuple, Type, TypeVar, cast
from mode.utils.objects import cached_property
from mode.utils.text import pluralize
from faust.exceptions import ValidationError
from faust.types.models import FieldDescriptorT, ModelT, T
from faust.utils import iso8601
from .tags import Tag
from .typing import NodeType, TypeExpression
__all__ = ['TYPE_TO_FIELD', 'FieldDescriptor', 'BooleanField',
    'NumberField', 'FloatField', 'IntegerField', 'DatetimeField',
    'DecimalField', 'BytesField', 'StringField', 'field_for_type']
CharacterType = TypeVar('CharacterType', str, bytes)


def _is_concrete_model(typ=None):
    return typ is not None and inspect.isclass(typ) and issubclass(typ, ModelT
        ) and typ is not ModelT and not getattr(typ, '__is_abstract__', False)


class FieldDescriptor(FieldDescriptorT[T]):
    """Describes a field.

    Used for every field in Record so that they can be used in join's
    /group_by etc.

    Examples:
        >>> class Withdrawal(Record):
        ...    account_id: str
        ...    amount: float = 0.0

        >>> Withdrawal.account_id
        <FieldDescriptor: Withdrawal.account_id: str>
        >>> Withdrawal.amount
        <FieldDescriptor: Withdrawal.amount: float = 0.0>

    Arguments:
        field (str): Name of field.
        type (Type): Field value type.
        required (bool): Set to false if field is optional.
        default (Any): Default value when `required=False`.

    Keyword Arguments:
        model (Type): Model class the field belongs to.
        parent (FieldDescriptorT): parent field if any.
    """
    field: str
    input_name: str
    output_name: str
    type: Type[T]
    model: Type[ModelT]
    required: bool = True
    default: Optional[T] = None
    coerce: bool = False
    exclude: bool = False
    tag: Optional[Type[Tag]]
    _to_python: Optional[Callable[[T], T]]
    _expr: Optional[TypeExpression]

    def __init__(self, *, field: str=None, input_name: str=None,
        output_name: str=None, type: Type[T]=None, model: Type[ModelT]=None,
        required: bool=True, default: Optional[T]=None, parent: Optional[
        FieldDescriptorT]=None, coerce: Optional[bool]=None, exclude:
        Optional[bool]=None, date_parser: Optional[Callable[[Any], datetime
        ]]=None, tag: Optional[Type[Tag]]=None, **options: Any):
        self.field = cast(str, field)
        self.input_name = cast(str, input_name or field)
        self.output_name = output_name or self.input_name
        self.type = cast(Type[T], type)
        self.model = cast(Type[ModelT], model)
        self.required = required
        self.default = default
        self.parent = parent
        self._copy_descriptors(self.type)
        if coerce is not None:
            self.coerce = coerce
        if exclude is not None:
            self.exclude = exclude
        self.options = options
        if date_parser is None:
            date_parser = iso8601.parse
        self.date_parser: Callable[[Any], datetime] = date_parser
        self.tag = tag
        self._to_python = None
        self._expr = None

    def on_model_attached(self):
        self._expr = self._prepare_type_expression()
        self._to_python = self._compile_type_expression()

    def _prepare_type_expression(self):
        expr = TypeExpression(self.type, user_types=self.options[
            'model_coercions'], date_parser=self.date_parser)
        return expr

    def _compile_type_expression(self):
        assert self._expr is not None
        expr = self._expr
        comprehension = expr.as_function(stacklevel=2)
        if (expr.has_generic_types and expr.has_custom_types or expr.
            has_nonfield_types):
            return cast(Callable, comprehension)
        return None

    def __set_name__(self, owner, name):
        self.model = owner
        self.field = name

    def clone(self, **kwargs: Any):
        return type(self)(**{**self.as_dict(), **kwargs})

    def as_dict(self):
        return {'field': self.field, 'input_name': self.input_name,
            'output_name': self.output_name, 'type': self.type, 'model':
            self.model, 'required': self.required, 'default': self.default,
            'parent': self.parent, 'coerce': self.coerce, 'exclude': self.
            exclude, 'date_parser': self.date_parser, 'tag': self.tag, **
            self.options}

    def validate_all(self, value):
        need_coercion = not self.coerce
        try:
            v = self.prepare_value(value, coerce=need_coercion)
        except (TypeError, ValueError) as exc:
            vt = type(value)
            yield self.validation_error(
                f'{self.field} is not correct type for {self}, got {vt!r}: {exc!r}'
                )
        except Exception as exc:
            yield self.validation_error(
                f'{self.field} got internal error for value {value!r} {exc!r}')
        else:
            if v is not None or self.required:
                yield from self.validate(cast(T, v))

    def validate(self, value):
        return iter([])

    def to_python(self, value):
        to_python = self._to_python
        if to_python is not None:
            value = to_python(value)
        return self.prepare_value(value)

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        return cast(T, value)

    def _copy_descriptors(self, typ=None):
        if typ is not None and _is_concrete_model(typ):
            typ._contribute_field_descriptors(self, typ._options, parent=self)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        field = self.field
        instance_dict = instance.__dict__
        to_python = self._to_python
        value = instance_dict[field]
        if self.lazy_coercion and to_python is not None:
            evaluated_fields: Set[str]
            evaluated_fields = instance.__evaluated_fields__
            if field not in evaluated_fields:
                if value is not None or self.required:
                    value = instance_dict[field] = to_python(value)
                evaluated_fields.add(field)
        return value

    def should_coerce(self, value, coerce=None):
        c = coerce if coerce is not None else self.coerce
        return c and (self.required or value is not None)

    def getattr(self, obj):
        """Get attribute from model recursively.

        Supports recursive lookups e.g. ``model.getattr('x.y.z')``.
        """
        return attrgetter('.'.join(reversed(list(self._parents_path()))))(obj)

    def _parents_path(self):
        node: Optional[FieldDescriptorT] = self
        while node:
            yield node.field
            node = node.parent

    def validation_error(self, reason):
        return ValidationError(reason, field=self)

    def __set__(self, instance, value):
        value = cast(T, self.prepare_value(value))
        if self.tag:
            value = cast(T, self.tag(value, field=self.field))
        else:
            value = value
        instance.__dict__[self.field] = value

    def __repr__(self):
        default = '' if self.required else f' = {self.default!r}'
        typ = getattr(self.type, '__name__', self.type)
        return f'<{type(self).__name__}: {self.ident}: {typ}{default}>'

    @property
    def ident(self):
        """Return the fields identifier."""
        return f'{self.model.__name__}.{self.field}'

    @cached_property
    def related_models(self):
        assert self._expr is not None
        return self._expr.found_types[NodeType.MODEL]

    @cached_property
    def lazy_coercion(self):
        assert self._expr is not None
        return self._expr.has_generic_types or self._expr.has_models


class BooleanField(FieldDescriptor[bool]):

    def validate(self, value):
        if not isinstance(value, bool):
            yield self.validation_error(
                f'{self.field} must be True or False, of type bool')

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        if self.should_coerce(value, coerce):
            return True if value else False
        return value


class NumberField(FieldDescriptor[T]):
    max_value: Optional[int]
    min_value: Optional[int]

    def __init__(self, *, max_value: Optional[int]=None, min_value:
        Optional[int]=None, **kwargs: Any):
        self.max_value = max_value
        self.min_value = min_value
        super().__init__(**kwargs, **{'max_value': max_value, 'min_value':
            min_value})

    def validate(self, value):
        val = cast(int, value)
        max_ = self.max_value
        if max_ is not None:
            if val > max_:
                yield self.validation_error(
                    f'{self.field} cannot be more than {max_}')
        min_ = self.min_value
        if min_ is not None:
            if val < min_:
                yield self.validation_error(
                    f'{self.field} must be at least {min_}')


class IntegerField(NumberField[int]):

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        return int(value) if self.should_coerce(value, coerce) else value


class FloatField(NumberField[float]):

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        return float(value) if self.should_coerce(value, coerce) else value


class DecimalField(NumberField[Decimal]):
    max_digits: Optional[int] = None
    max_decimal_places: Optional[int] = None

    def __init__(self, *, max_digits: Optional[int]=None,
        max_decimal_places: Optional[int]=None, **kwargs: Any):
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places
        super().__init__(**kwargs, **{'max_digits': max_digits,
            'max_decimal_places': max_decimal_places})

    def to_python(self, value):
        if self._to_python is None:
            if self.model._options.decimals:
                return self.prepare_value(value, coerce=True)
            return self.prepare_value(value)
        else:
            return self._to_python(value)

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        return Decimal(value) if self.should_coerce(value, coerce) else value

    def validate(self, value):
        if not value.is_finite():
            yield self.validation_error(f'Illegal value in decimal: {value!r}')
        decimal_tuple: Optional[DecimalTuple] = None
        mdp = self.max_decimal_places
        if mdp is not None:
            decimal_tuple = value.as_tuple()
            if abs(decimal_tuple.exponent) > mdp:
                yield self.validation_error(
                    f'{self.field} must have less than {mdp} decimal places.')
        max_digits = self.max_digits
        if max_digits is not None:
            if decimal_tuple is None:
                decimal_tuple = value.as_tuple()
            digits = len(decimal_tuple.digits[:decimal_tuple.exponent])
            if digits > max_digits:
                yield self.validation_error(
                    f'{self.field} must have less than {max_digits} digits.')


class CharField(FieldDescriptor[CharacterType]):
    max_length: Optional[int]
    min_length: Optional[int]
    trim_whitespace: bool
    allow_blank: bool

    def __init__(self, *, max_length: Optional[int]=None, min_length:
        Optional[int]=None, trim_whitespace: bool=False, allow_blank: bool=
        False, **kwargs: Any):
        self.max_length = max_length
        self.min_length = min_length
        self.trim_whitespace = trim_whitespace
        self.allow_blank = allow_blank
        super().__init__(**kwargs, **{'max_length': max_length,
            'min_length': min_length, 'trim_whitespace': trim_whitespace,
            'allow_blank': allow_blank})

    def validate(self, value):
        allow_blank = self.allow_blank
        if not allow_blank and not len(value):
            yield self.validation_error(f'{self.field} cannot be left blank')
        max_ = self.max_length
        length = len(value)
        min_ = self.min_length
        if min_ is not None:
            if length < min_:
                chars = pluralize(min_, 'character')
                yield self.validation_error(
                    f'{self.field} must have at least {min_} {chars}')
        if max_ is not None:
            if length > max_:
                chars = pluralize(max_, 'character')
                yield self.validation_error(
                    f'{self.field} must be at most {max_} {chars}')


class StringField(CharField[str]):

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        if self.should_coerce(value, coerce):
            val = str(value) if not isinstance(value, str) else value
            if self.trim_whitespace:
                return val.strip()
            return val
        else:
            return value


class DatetimeField(FieldDescriptor[datetime]):

    def to_python(self, value):
        if self._to_python is None:
            if self.model._options.isodates:
                return self.prepare_value(value, coerce=True)
            return self.prepare_value(value)
        else:
            return self._to_python(value)

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        if self.should_coerce(value, coerce):
            if value is not None and not isinstance(value, datetime):
                return self.date_parser(value)
            else:
                return value
        else:
            return value


class BytesField(CharField[bytes]):
    encoding: str = sys.getdefaultencoding()
    errors: str = 'strict'

    def __init__(self, *, encoding: Optional[str]=None, errors: Optional[
        str]=None, **kwargs: Any):
        if encoding is not None:
            self.encoding = encoding
        if errors is not None:
            self.errors = errors
        super().__init__(encoding=self.encoding, errors=self.errors, **kwargs)

    def prepare_value(self, value, *, coerce: Optional[bool]=None):
        if self.should_coerce(value, coerce):
            if isinstance(value, bytes):
                val = value
            else:
                val = cast(str, value).encode(encoding=self.encoding,
                    errors=self.errors)
            if self.trim_whitespace:
                return val.strip()
            return val
        else:
            return value


TYPE_TO_FIELD: Mapping[Type[Any], Type[FieldDescriptorT]] = {bool:
    BooleanField, int: IntegerField, float: FloatField, Decimal:
    DecimalField, str: StringField, bytes: BytesField, datetime: DatetimeField}


@lru_cache(maxsize=2048)
def field_for_type(typ):
    try:
        return TYPE_TO_FIELD[typ], None
    except KeyError:
        try:
            origin = typ.__origin__
        except AttributeError:
            origin = None
        if origin is not None and issubclass(origin, Tag):
            args = typ.__args__
            if args:
                return field_for_type(args[0])[0], origin
        for basecls, DescriptorType in TYPE_TO_FIELD.items():
            try:
                if issubclass(typ, basecls):
                    return DescriptorType, None
            except TypeError:
                continue
        return FieldDescriptor, None
