import functools
import importlib.util
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Annotated,
    Any,
    Callable,
    ForwardRef,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)
import pytest
from dirty_equals import HasRepr, IsStr
from pydantic_core import (
    ErrorDetails,
    InitErrorDetails,
    PydanticSerializationError,
    PydanticUndefined,
    core_schema,
)
from typing_extensions import TypeAliasType, TypedDict, get_args
from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    PrivateAttr,
    PydanticDeprecatedSince20,
    PydanticSchemaGenerationError,
    PydanticUserError,
    RootModel,
    TypeAdapter,
    ValidationError,
    constr,
    errors,
    field_validator,
    model_validator,
    root_validator,
    validator,
)
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import Field, computed_field
from pydantic.functional_serializers import field_serializer, model_serializer


def test_str_bytes() -> None:
    class Model(BaseModel):
        v: Union[str, bytes]

    m = Model(v='s')
    assert m.v == 's'
    assert repr(Model.model_fields['v']) == 'FieldInfo(annotation=Union[str, bytes], required=True)'
    m = Model(v=b'b')
    assert m.v == b'b'
    with pytest.raises(ValidationError) as exc_info:
        Model(v=None)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'string_type', 'loc': ('v', 'str'), 'msg': 'Input should be a valid string', 'input': None},
        {'type': 'bytes_type', 'loc': ('v', 'bytes'), 'msg': 'Input should be a valid bytes', 'input': None}
    ]


def test_str_bytes_none() -> None:
    class Model(BaseModel):
        v: Optional[Union[str, bytes]] = None

    m = Model(v='s')
    assert m.v == 's'
    m = Model(v=b'b')
    assert m.v == b'b'
    m = Model(v=None)
    assert m.v is None


def test_union_int_str() -> None:
    class Model(BaseModel):
        v: Union[int, str, bytes]

    m = Model(v=123)
    assert m.v == 123
    m = Model(v='123')
    assert m.v == '123'
    m = Model(v=b'foobar')
    assert m.v == b'foobar'
    m = Model(v=12.0)
    assert m.v == 12
    with pytest.raises(ValidationError) as exc_info:
        Model(v=None)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_type', 'loc': ('v', 'int'), 'msg': 'Input should be a valid integer', 'input': None},
        {'type': 'string_type', 'loc': ('v', 'str'), 'msg': 'Input should be a valid string', 'input': None}
    ]


def test_union_int_any() -> None:
    class Model(BaseModel):
        v: Any

    m = Model(v=123)
    assert m.v == 123
    m = Model(v='123')
    assert m.v == '123'
    m = Model(v='foobar')
    assert m.v == 'foobar'
    m = Model(v=None)
    assert m.v is None


def test_typed_list() -> None:
    class Model(BaseModel):
        v: list[int]

    m = Model(v=[1, 2, '3'])
    assert m.v == [1, 2, 3]
    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, 'x', 'y'])
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_parsing', 'loc': ('v', 1), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'},
        {'type': 'int_parsing', 'loc': ('v', 2), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'y'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model(v=1)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'list_type', 'loc': ('v',), 'msg': 'Input should be a valid list', 'input': 1}
    ]


def test_typed_set() -> None:
    class Model(BaseModel):
        v: set[int]

    assert Model(v={1, 2, '3'}).v == {1, 2, 3}
    assert Model(v=[1, 2, '3']).v == {1, 2, 3}
    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, 'x'])
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_parsing', 'loc': ('v', 1), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}
    ]


def test_dict_dict() -> None:
    class Model(BaseModel):
        v: dict[Any, Any]

    assert Model(v={'foo': 1}).model_dump() == {'v': {'foo': 1}}


def test_none_list() -> None:
    class Model(BaseModel):
        v: list[None] = [None]

    assert Model.model_json_schema() == {
        'title': 'Model',
        'type': 'object',
        'properties': {
            'v': {
                'title': 'V',
                'default': [None],
                'type': 'array',
                'items': {'type': 'null'}
            }
        }
    }


T = TypeVar('T')


@pytest.mark.parametrize(
    'value,result',
    [
        ({'a': 2, 'b': 4}, {'a': 2, 'b': 4}),
        ({b'a': '2', 'b': 4}, {'a': 2, 'b': 4}),
    ]
)
def test_typed_dict(value: dict[Any, Any], result: dict[Any, Any]) -> None:
    class Model(BaseModel):
        v: dict[int, int]

    assert Model(v=value).v == result


@pytest.mark.parametrize(
    'value,errors',
    [
        (
            1,
            [{'type': 'dict_type', 'loc': ('v',), 'msg': 'Input should be a valid dictionary', 'input': 1}]
        ),
        (
            {'a': 'b'},
            [{'type': 'int_parsing', 'loc': ('v', 'a'), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'b'}]
        ),
        (
            [1, 2, 3],
            [{'type': 'dict_type', 'loc': ('v',), 'msg': 'Input should be a valid dictionary', 'input': [1, 2, 3]}]
        ),
    ]
)
def test_typed_dict_error(value: Any, errors: list[dict[str, Any]]) -> None:
    class Model(BaseModel):
        v: dict[int, int]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=value)
    assert exc_info.value.errors(include_url=False) == errors


def test_dict_key_error() -> None:
    class Model(BaseModel):
        v: dict[int, int]

    assert Model(v={1: 2, '3': '4'}).v == {1: 2, 3: 4}
    with pytest.raises(ValidationError) as exc_info:
        Model(v={'foo': 2, '3': '4'})
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_parsing', 'loc': ('v', 'foo', '[key]'), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'foo'}
    ]


def test_tuple() -> None:
    class Model(BaseModel):
        v: tuple[int, float, bool]

    m = Model(v=['1.0', '2.2', 'true'])
    assert m.v == (1, 2.2, True)


def test_tuple_more() -> None:
    class Model(BaseModel):
        empty_tuple: tuple = ()
        simple_tuple: tuple[int, ...]
        tuple_of_different_types: tuple[int, float, str, bool]
        tuple_of_single_tuples: tuple[tuple[int, ...], ...]

    m = Model(
        empty_tuple=[],
        simple_tuple=[1, 2, 3, 4],
        tuple_of_different_types=[4, 3.1, 'str', 1],
        tuple_of_single_tuples=(('1',), (2,))
    )
    assert m.model_dump() == {
        'empty_tuple': (),
        'simple_tuple': (1, 2, 3, 4),
        'tuple_of_different_types': (4, 3.1, 'str', True),
        'tuple_of_single_tuples': ((1,), (2,))
    }


@pytest.mark.parametrize(
    'dict_cls,frozenset_cls,list_cls,set_cls,tuple_cls,type_cls',
    [
        (
            typing.Dict,
            typing.FrozenSet,
            typing.List,
            typing.Set,
            typing.Tuple,
            typing.Type
        ),
        (
            dict,
            frozenset,
            list,
            set,
            tuple,
            type
        )
    ]
)
def test_pep585_generic_types(
    dict_cls: Type[Any],
    frozenset_cls: Type[Any],
    list_cls: Type[Any],
    set_cls: Type[Any],
    tuple_cls: Type[Any],
    type_cls: Type[Any]
) -> None:
    class Type1:
        pass

    class Type2:
        pass

    class Model(BaseModel, arbitrary_types_allowed=True):
        a: dict_cls[Any, Any]
        a1: dict_cls[str, int]
        b: frozenset_cls[Any]
        b1: frozenset_cls[int]
        c: list_cls[Any]
        c1: list_cls[int]
        d: set_cls[Any]
        d1: set_cls[int]
        e: tuple_cls[Any, ...]
        e1: tuple_cls[int, ...]
        e2: tuple_cls[int, ...]
        e3: tuple_cls[()]
        f: type_cls[Type1]
        f1: type_cls[Type1]

    default_model_kwargs: dict[str, Any] = {
        'a': {},
        'a1': {'a': '1'},
        'b': [],
        'b1': ('1',),
        'c': [],
        'c1': ('1',),
        'd': [],
        'd1': ['1'],
        'e': [],
        'e1': ['1'],
        'e2': ['1', '2'],
        'e3': [],
        'f': Type1,
        'f1': Type1
    }
    m = Model(**default_model_kwargs)
    assert m.a == {}
    assert m.a1 == {'a': 1}
    assert m.b == frozenset()
    assert m.b1 == frozenset({1})
    assert m.c == []
    assert m.c1 == [1]
    assert m.d == set()
    assert m.d1 == {1}
    assert m.e == ()
    assert m.e1 == (1,)
    assert m.e2 == (1, 2)
    assert m.e3 == ()
    assert m.f == Type1
    assert m.f1 == Type1
    with pytest.raises(ValidationError) as exc_info:
        Model(**{**default_model_kwargs, 'e3': (1,)})
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'too_long',
            'loc': ('e3',),
            'msg': 'Tuple should have at most 0 items after validation, not 1',
            'input': (1,),
            'ctx': {'field_type': 'Tuple', 'max_length': 0, 'actual_length': 1}
        }
    ]
    Model(**{**default_model_kwargs, 'f': Type2})
    with pytest.raises(ValidationError) as exc_info:
        Model(**{**default_model_kwargs, 'f1': Type2})
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_subclass_of',
            'loc': ('f1',),
            'msg': 'Input should be a subclass of test_pep585_generic_types.<locals>.Type1',
            'input': HasRepr(IsStr(regex=".+\\.Type2'>")),
            'ctx': {'class': 'test_pep585_generic_types.<locals>.Type1'}
        }
    ]


def test_tuple_length_error() -> None:
    class Model(BaseModel):
        v: tuple[int, int, int]
        w: tuple[()]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, 2], w=[1])
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'missing', 'loc': ('v', 2), 'msg': 'Field required', 'input': [1, 2]},
        {
            'type': 'too_long',
            'loc': ('w',),
            'msg': 'Tuple should have at most 0 items after validation, not 1',
            'input': [1],
            'ctx': {'field_type': 'Tuple', 'max_length': 0, 'actual_length': 1}
        }
    ]


def test_tuple_invalid() -> None:
    class Model(BaseModel):
        v: tuple[int, float, bool]

    with pytest.raises(ValidationError) as exc_info:
        Model(v='xxx')
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'tuple_type', 'loc': ('v',), 'msg': 'Input should be a valid tuple', 'input': 'xxx'}
    ]


def test_tuple_value_error() -> None:
    class Model(BaseModel):
        v: tuple[int, float, Decimal]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=['x', 'y', 'x'])
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_parsing', 'loc': ('v', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'},
        {'type': 'float_parsing', 'loc': ('v', 1), 'msg': 'Input should be a valid number, unable to parse string as a number', 'input': 'y'},
        {'type': 'decimal_parsing', 'loc': ('v', 2), 'msg': 'Input should be a valid decimal', 'input': 'x'}
    ]


def test_recursive_list() -> None:
    class SubModel(BaseModel):
        name: str
        count: Optional[int]

    class Model(BaseModel):
        v: list[SubModel]

    m = Model(v=[])
    assert m.v == []
    m = Model(v=[{'name': 'testing', 'count': 4}])
    assert repr(m) == "Model(v=[SubModel(name='testing', count=4)])"
    assert m.v[0].name == 'testing'
    assert m.v[0].count == 4
    assert m.model_dump() == {'v': [{'count': 4, 'name': 'testing'}]}
    with pytest.raises(ValidationError) as exc_info:
        Model(v=['x'])
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'model_type', 'loc': ('v', 0), 'msg': 'Input should be a valid dictionary or instance of SubModel', 'input': 'x', 'ctx': {'class_name': 'SubModel'}}
    ]


def test_recursive_list_error() -> None:
    class SubModel(BaseModel):
        name: str
        count: Optional[int]

    class Model(BaseModel):
        v: list[SubModel]

    with pytest.raises(ValidationError) as exc_info:
        Model(v=[{}])
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('v', 0, 'name'), 'msg': 'Field required', 'type': 'missing'}
    ]


def test_list_unions() -> None:
    class Model(BaseModel):
        v: list[Union[int, str]]

    assert Model(v=[123, '456', 'foobar']).v == [123, '456', 'foobar']
    with pytest.raises(ValidationError) as exc_info:
        Model(v=[1, 2, None])
    errors = exc_info.value.errors(include_url=False)
    expected_errors = [
        {'input': None, 'loc': ('v', 2, 'int'), 'msg': 'Input should be a valid integer', 'type': 'int_type'},
        {'input': None, 'loc': ('v', 2, 'str'), 'msg': 'Input should be a valid string', 'type': 'string_type'}
    ]
    assert sorted(errors, key=str) == sorted(expected_errors, key=str)


def test_recursive_lists() -> None:
    class Model(BaseModel):
        v: list[list[Union[int, float]]]

    assert Model(v=[[1, 2], [3, '4', '4.1']]).v == [[1, 2], [3, 4, 4.1]]
    assert Model.model_fields['v'].annotation == list[list[Union[int, float]]]
    assert Model.model_fields['v'].is_required()


class StrEnum(str, Enum):
    a = 'a10'
    b = 'b10'


def test_str_enum() -> None:
    class Model(BaseModel):
        v: StrEnum

    assert Model(v='a10').v is StrEnum.a
    with pytest.raises(ValidationError):
        Model(v='different')


def test_any_dict() -> None:
    class Model(BaseModel):
        v: dict[Any, Any]

    assert Model(v={1: 'foobar'}).model_dump() == {'v': {1: 'foobar'}}
    assert Model(v={123: 456}).model_dump() == {'v': {123: 456}}
    assert Model(v={2: [1, 2, 3]}).model_dump() == {'v': {2: [1, 2, 3]}}


def test_success_values_include() -> None:
    class Model(BaseModel):
        a: int = 1
        b: int = 2
        c: int = 3

    m = Model()
    assert m.model_dump() == {'a': 1, 'b': 2, 'c': 3}
    assert m.model_dump(include={'a'}) == {'a': 1}
    assert m.model_dump(exclude={'a'}) == {'b': 2, 'c': 3}
    assert m.model_dump(include={'a', 'b'}, exclude={'a'}) == {'b': 2}


def test_include_exclude_unset() -> None:
    class Model(BaseModel):
        a: Optional[int] = None
        b: Optional[int] = None
        c: int = 3
        d: int = 4
        e: int = 5
        f: int = 6

    m = Model(a=1, b=2, e=5, f=7)
    assert m.model_dump() == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 7}
    assert m.model_fields_set == {'a', 'b', 'e', 'f'}
    assert m.model_dump(exclude_unset=True) == {'a': 1, 'b': 2, 'e': 5, 'f': 7}
    assert m.model_dump(include={'a'}, exclude_unset=True) == {'a': 1}
    assert m.model_dump(include={'c'}, exclude_unset=True) == {}
    assert m.model_dump(exclude={'a'}, exclude_unset=True) == {'b': 2, 'e': 5, 'f': 7}
    assert m.model_dump(exclude={'c'}, exclude_unset=True) == {'a': 1, 'b': 2, 'e': 5, 'f': 7}
    assert m.model_dump(include={'a', 'b', 'c'}, exclude={'b'}, exclude_unset=True) == {'a': 1}
    assert m.model_dump(include={'a', 'b', 'c'}, exclude={'a', 'c'}, exclude_unset=True) == {'b': 2}


def test_include_exclude_defaults() -> None:
    class Model(BaseModel):
        a: Optional[int] = None
        b: Optional[int] = None
        c: int = 3
        d: int = 4
        e: int = 5
        f: int = 6

    m = Model(a=1, b=2, e=5, f=7)
    assert m.model_dump() == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 7}
    assert m.model_fields_set == {'a', 'b', 'e', 'f'}
    assert m.model_dump(exclude_defaults=True) == {'a': 1, 'b': 2, 'f': 7}
    assert m.model_dump(include={'a'}, exclude_defaults=True) == {'a': 1}
    assert m.model_dump(include={'c'}, exclude_defaults=True) == {}
    assert m.model_dump(exclude={'a'}, exclude_defaults=True) == {'b': 2, 'f': 7}
    assert m.model_dump(exclude={'c'}, exclude_defaults=True) == {'a': 1, 'b': 2, 'f': 7}
    assert m.model_dump(include={'a', 'b', 'c'}, exclude={'b'}, exclude_defaults=True) == {'a': 1}
    assert m.model_dump(include={'a', 'b', 'c'}, exclude={'a', 'c'}, exclude_defaults=True) == {'b': 2}
    assert m.model_dump(include={'a': 1}.keys()) == {'a': 1}
    assert m.model_dump(exclude={'a': 1}.keys()) == {'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 7}
    assert m.model_dump(include={'a': 1}.keys(), exclude_unset=True) == {'a': 1}
    assert m.model_dump(exclude={'a': 1}.keys(), exclude_unset=True) == {'b': 2, 'e': 5, 'f': 7}
    assert m.model_dump(include=['a']) == {'a': 1}
    assert m.model_dump(exclude=['a']) == {'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 7}
    assert m.model_dump(include=['a'], exclude_unset=True) == {'a': 1}
    assert m.model_dump(exclude=['a'], exclude_unset=True) == {'b': 2, 'e': 5, 'f': 7}


def test_advanced_exclude() -> None:
    class SubSubModel(BaseModel):
        a: Optional[str] = None
        b: Optional[str] = None

    class SubModel(BaseModel):
        c: str
        d: list[SubSubModel]

    class Model(BaseModel):
        e: str
        f: SubModel

    m = Model(
        e='e',
        f=SubModel(
            c='foo',
            d=[
                SubSubModel(a='a', b='b'),
                SubSubModel(a='c', b='e')
            ]
        )
    )
    assert m.model_dump(exclude={'f': {'c': ..., 'd': {-1: {'a'}}}}) == {
        'e': 'e',
        'f': {
            'd': [
                {'a': 'a', 'b': 'b'},
                {'b': 'e'}
            ]
        }
    }
    assert m.model_dump(exclude={'e': ..., 'f': {'d'}}) == {'f': {'c': 'foo'}}


def test_advanced_exclude_by_alias() -> None:
    class SubSubModel(BaseModel):
        aliased_b: Optional[str] = Field(alias='b_alias')

    class SubModel(BaseModel):
        aliased_c: str = Field(alias='c_alias')
        aliased_d: list[SubSubModel] = Field(alias='d_alias')

    class Model(BaseModel):
        aliased_e: str = Field(alias='e_alias')
        aliased_f: SubModel = Field(alias='f_alias')

    m = Model(
        e_alias='e',
        f_alias=SubModel(
            c_alias='foo',
            d_alias=[
                SubSubModel(a='a', b_alias='b'),
                SubSubModel(a='c', b_alias='e')
            ]
        )
    )
    excludes = {'aliased_f': {'aliased_c': ..., 'aliased_d': {-1: {'a'}}}}
    assert m.model_dump(exclude=excludes, by_alias=True) == {
        'e_alias': 'e',
        'f_alias': {
            'd_alias': [
                {'a': 'a', 'b_alias': 'b'},
                {'b_alias': 'e'}
            ]
        }
    }
    excludes = {'aliased_e': ..., 'aliased_f': {'aliased_d'}}
    assert m.model_dump(exclude=excludes, by_alias=True) == {'f_alias': {'c_alias': 'foo'}}


def test_advanced_value_include() -> None:
    class SubSubModel(BaseModel):
        a: str = None
        b: str = None

    class SubModel(BaseModel):
        c: str = None
        d: list[SubSubModel] = None

    class Model(BaseModel):
        e: str = None
        f: SubModel = None

    m = Model(
        e='e',
        f=SubModel(
            c='foo',
            d=[
                SubSubModel(a='a', b='b'),
                SubSubModel(a='c', b='e')
            ]
        )
    )
    assert m.model_dump(include={'f'}) == {
        'f': {
            'c': 'foo',
            'd': [
                {'a': 'a', 'b': 'b'},
                {'a': 'c', 'b': 'e'}
            ]
        }
    }
    assert m.model_dump(include={'e'}) == {'e': 'e'}
    assert m.model_dump(include={'f': {'d': {0: ..., -1: {'b'}}}}) == {
        'f': {
            'd': [
                {'a': 'a', 'b': 'b'},
                {'b': 'e'}
            ]
        }
    }


def test_advanced_value_exclude_include() -> None:
    class SubSubModel(BaseModel):
        a: str = None
        b: str = None

    class SubModel(BaseModel):
        c: str = None
        d: list[SubSubModel] = None

    class Model(BaseModel):
        e: str = None
        f: SubModel = None

    m = Model(
        e='e',
        f=SubModel(
            c='foo',
            d=[
                SubSubModel(a='a', b='b'),
                SubSubModel(a='c', b='e')
            ]
        )
    )
    assert m.model_dump(
        exclude={'f': {'c': ..., 'd': {-1: {'a'}}}},
        include={'f'}
    ) == {
        'f': {
            'd': [
                {'a': 'a', 'b': 'b'},
                {'b': 'e'}
            ]
        }
    }
    assert m.model_dump(
        exclude={'e': ..., 'f': {'d'}},
        include={'e', 'f'}
    ) == {'f': {'c': 'foo'}}
    assert m.model_dump(
        exclude={'f': {'d': {-1: {'a'}}}},
        include={'f': {'d'}}
    ) == {
        'f': {
            'd': [
                {'a': 'a', 'b': 'b'},
                {'b': 'e'}
            ]
        }
    }


@pytest.mark.parametrize(
    'exclude,expected',
    [
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}}}}},
            {'subs': [{'k': 1, 'subsubs': [{'j': 1}, {'j': 2}]}, {'k': 2, 'subsubs': [{'j': 3}]}]},
            id='Normal nested __all__'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}}}, 0: {'subsubs': {'__all__': {'j'}}}}},
            {'subs': [{'k': 1, 'subsubs': [{}, {}]}, {'k': 2, 'subsubs': [{'j': 3}]}]},
            id='Merge sub dicts 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': ...}, 0: {'subsubs': {'__all__': {'j'}}}}},
            {'subs': [{'k': 1, 'subsubs': [{'j': 1}, {'j': 2}]}, {'subsubs': [{'j': 3}]}]},
            id='Merge sub sets 2'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}, 0: {'j'}}}}},
            {'subs': [{'k': 1, 'subsubs': [{'i': 1, 'j': 1}, {'i': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Merge sub sets 3'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {0}, 0: {'subsubs': {1}}}}},
            {'subs': [{'k': 1, 'subsubs': []}, {'k': 2, 'subsubs': []}]},
            id='Merge sub sets 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {0: {'i'}}}, 0: {'subsubs': {1}}}},
            {'subs': [{'k': 1, 'subsubs': [{'j': 1}]}, {'k': 2, 'subsubs': [{'j': 3}]}]},
            id='Merge sub dict-set'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs'}, 0: {'k'}}},
            {'subs': [{}, {'k': 2}]},
            id='Different keys 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': ...}, 0: {'k'}}},
            {'subs': [{}, {'k': 2}]},
            id='Different keys 2'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs'}, 0: {'k': ...}}},
            {'subs': [{}, {'k': 2}]},
            id='Different keys 3'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}}}}},  # Nested different keys
            {'subs': [{'k': 1, 'subsubs': [{'i': 1}, {'i': 2}]}, {'k': 2, 'subsubs': [{'i': 3}]}]},
            id='Nested different keys 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i': ...}, 0: {'j'}}}}},
            {'subs': [{'k': 1, 'subsubs': [{'i': 1, 'j': 1}, {'i': 2}]}, {'k': 2, 'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Nested different keys 2'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}, 0: {'j': ...}}}}},
            {'subs': [{'k': 1, 'subsubs': [{'i': 1, 'j': 1}, {'i': 2}]}, {'k': 2, 'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Nested different keys 3'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'j'}}}, 0: ...}},
            {'subs': [{'k': 1, 'subsubs': [{'j': 1}, {'j': 2}]}, {'subsubs': [{'j': 3}]}]},
            id='Ignore __all__ for index with defined exclude 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'j'}}}, 0: ...}},
            {'subs': [{'k': 2, 'subsubs': [{'j': 3}]}]},
            id='Ignore __all__ for index with defined exclude 2'
        ),
        pytest.param(
            {'subs': {'__all__': ..., 0: {'subsubs'}}},
            {'subs': [{'k': 1}]},
            id='Ignore __all__ for index with defined exclude 3'
        ),
    ]
)
def test_advanced_exclude_nested_lists(exclude: dict[str, Any], expected: dict[str, Any]) -> None:
    class SubSubModel(BaseModel):
        i: Optional[int] = None
        j: Optional[int] = None
        k: Optional[int] = None

    class SubModel(BaseModel):
        k: Optional[int] = None
        subsubs: list[SubSubModel] = []

    class Model(BaseModel):
        subs: list[SubModel] = []

    m = Model(subs=[
        SubModel(k=1, subsubs=[SubSubModel(i=1, j=1), SubSubModel(i=2, j=2)]),
        SubModel(k=2, subsubs=[SubSubModel(i=3, j=3)])
    ])
    assert m.model_dump(exclude=exclude) == expected


@pytest.mark.parametrize(
    'include,expected',
    [
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}}}}},
            {'subs': [{'subsubs': [{'i': 1}, {'i': 2}]}, {'subsubs': [{'i': 3}]}]},
            id='Normal nested __all__'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}}}, 0: {'subsubs': {'__all__': {'j'}}}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2, 'j': 2}]}, {'subsubs': [{'i': 3}]}]},
            id='Merge sub dicts 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': ...}, 0: {'subsubs': {'__all__': {'j'}}}}},
            {'subs': [{'subsubs': [{'j': 1}, {'j': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Merge sub dicts 2'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'j'}}}, 0: {'subsubs': ...}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2, 'j': 2}]}, {'subsubs': [{'j': 3}]}]},
            id='Merge sub dicts 3'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {0}}, 0: {'subsubs': {1}}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2, 'j': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Merge sub sets'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {0: {'i'}}}, 0: {'subsubs': {1}}}},
            {'subs': [{'subsubs': [{'i': 1}, {'i': 2, 'j': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Merge sub dict-set'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}}}}},
            {'subs': [{'subsubs': [{'i': 1}, {'i': 2}]}, {'subsubs': [{'i': 3}]}]},
            id='Nested different keys 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}}}, 0: {'k'}}},
            {'subs': [{'k': 1, 'subsubs': [{'i': 1}, {'i': 2}]}, {'subsubs': [{'i': 3}]}]},
            id='Nested different keys 2'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}, 0: {'j'}}}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Nested different keys 3'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}, 0: {'j'}}}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Nested different keys 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i': ...}, 0: {'j'}}}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Nested different keys 2'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'i'}, 0: {'j': ...}}}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2}]}, {'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Nested different keys 3'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'j'}}}, 0: {'subsubs': {'__all__': {'j'}}}}},
            {'subs': [{'subsubs': [{'j': 1}, {'j': 2}]}, {'subsubs': [{'j': 3}]}]},
            id='Ignore __all__ for index with defined include 1'
        ),
        pytest.param(
            {'subs': {'__all__': {'subsubs': {'__all__': {'j'}}}, 0: ...}},
            {'subs': [{'k': 1, 'subsubs': [{'i': 1, 'j': 1}, {'i': 2, 'j': 2}]}, {'subsubs': [{'j': 3}]}]},
            id='Ignore __all__ for index with defined include 2'
        ),
        pytest.param(
            {'subs': {'__all__': ..., 0: {'subsubs'}}},
            {'subs': [{'subsubs': [{'i': 1, 'j': 1}, {'i': 2, 'j': 2}]}, {'k': 2, 'subsubs': [{'i': 3, 'j': 3}]}]},
            id='Ignore __all__ for index with defined include 3'
        ),
    ]
)
def test_advanced_include_nested_lists(include: dict[str, Any], expected: dict[str, Any]) -> None:
    class SubSubModel(BaseModel):
        i: Optional[int] = None
        j: Optional[int] = None

    class SubModel(BaseModel):
        k: Optional[int] = None
        subsubs: list[SubSubModel] = []

    class Model(BaseModel):
        subs: list[SubModel] = []

    m = Model(subs=[
        SubModel(k=1, subsubs=[SubSubModel(i=1, j=1), SubSubModel(i=2, j=2)]),
        SubModel(k=2, subsubs=[SubSubModel(i=3, j=3)])
    ])
    assert m.model_dump(include=include) == expected


def test_field_set_ignore_extra() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='ignore')
        c: int = 3

    m = Model(a=1, b=2)
    assert m.model_dump() == {'a': 1, 'b': 2, 'c': 3}
    assert m.model_fields_set == {'a', 'b'}
    assert m.model_dump(exclude_unset=True) == {'a': 1, 'b': 2}
    m2 = Model(a=1, b=2, d=4)
    assert m2.model_dump() == {'a': 1, 'b': 2, 'c': 3}
    assert m2.model_fields_set == {'a', 'b'}
    assert m2.model_dump(exclude_unset=True) == {'a': 1, 'b': 2}


def test_field_set_allow_extra() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        c: int = 3

    m = Model(a=1, b=2)
    assert m.model_dump() == {'a': 1, 'b': 2, 'c': 3}
    assert m.model_fields_set == {'a', 'b'}
    assert m.model_dump(exclude_unset=True) == {'a': 1, 'b': 2}
    m2 = Model(a=1, b=2, d=4)
    assert m2.model_dump() == {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert m2.model_fields_set == {'a', 'b', 'd'}
    assert m2.model_dump(exclude_unset=True) == {'a': 1, 'b': 2, 'd': 4}


def test_field_set_field_name() -> None:
    class Model(BaseModel):
        b: int = 3

    assert Model(a=1, field_set=2).model_dump() == {'a': 1, 'field_set': 2, 'b': 3}
    assert Model(a=1, field_set=2).model_dump(exclude_unset=True) == {'a': 1, 'field_set': 2}
    assert Model.model_construct(a=1, field_set=3).model_dump() == {'a': 1, 'field_set': 3, 'b': 3}


def test_values_order() -> None:
    class Model(BaseModel):
        a: int = 1
        b: int = 2
        c: int = 3

    m = Model(c=30, b=20, a=10)
    assert list(m) == [('a', 10), ('b', 20), ('c', 30)]


def test_inheritance() -> None:
    class Foo(BaseModel):
        a: int

    with pytest.raises(TypeError, match="Field 'a' defined on a base class was overridden by a non-annotated attribute. All field definitions, including overrides, require a type annotation."):
        class Bar(Foo):
            x = 12.3
            a = 123.0

    class Bar2(Foo):
        x: float = 12.3
        a: float = 123.0

    assert Bar2().model_dump() == {'x': 12.3, 'a': 123.0}

    class Bar3(Foo):
        x: float = 12.3
        a: float = Field(default=123.0)

    assert Bar3().model_dump() == {'x': 12.3, 'a': 123.0}


def test_inheritance_subclass_default() -> None:
    class MyStr(str):
        pass

    class Simple(BaseModel):
        x: MyStr = MyStr('test')

        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    class Base(BaseModel):
        pass

    class Sub(Base):
        x: MyStr = MyStr('test')
        y: MyStr = MyStr('test')

        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    assert Sub.model_fields['x'].annotation == str
    assert Sub.model_fields['y'].annotation == MyStr


def test_invalid_type() -> None:
    with pytest.raises(PydanticSchemaGenerationError) as exc_info:
        class Model(BaseModel):
            x = 123
    assert 'Unable to generate pydantic-core schema for 123' in exc_info.value.args[0]


class CustomStr(str):
    def foobar(self) -> int:
        return 7


@pytest.mark.parametrize(
    'value,expected',
    [
        ('a string', 'a string'),
        (b'some bytes', 'some bytes'),
        (bytearray('foobar', encoding='utf8'), 'foobar'),
        (StrEnum.a, 'a10'),
        (CustomStr('whatever'), 'whatever')
    ]
)
def test_valid_string_types(value: Union[str, bytes, bytearray, StrEnum, CustomStr], expected: str) -> None:
    class Model(BaseModel):
        v: str

    assert Model(v=value).v == expected


@pytest.mark.parametrize(
    'value,errors',
    [
        (
            {'foo': 'bar'},
            [{'input': {'foo': 'bar'}, 'loc': ('v',), 'msg': 'Input should be a valid string', 'type': 'string_type'}]
        ),
        (
            [1, 2, 3],
            [{'input': [1, 2, 3], 'loc': ('v',), 'msg': 'Input should be a valid string', 'type': 'string_type'}]
        )
    ]
)
def test_invalid_string_types(value: Any, errors: list[dict[str, Any]]) -> None:
    class Model(BaseModel):
        v: str

    with pytest.raises(ValidationError) as exc_info:
        Model(v=value)
    assert exc_info.value.errors(include_url=False) == errors


def test_inheritance_config() -> None:
    class Parent(BaseModel):
        a: str

    class Child(Parent):
        model_config: ConfigDict = ConfigDict(str_to_lower=True)

    m1 = Parent(a='A')
    m2 = Child(a='A', b='B')
    assert repr(m1) == "Parent(a='A')"
    assert repr(m2) == "Child(a='a', b='b')"


def test_partial_inheritance_config() -> None:
    class Parent(BaseModel):
        a: int = Field(ge=0)

    class Child(Parent):
        b: int = Field(ge=0)

    Child(a=0, b=0)
    with pytest.raises(ValidationError) as exc_info:
        Child(a=-1, b=0)
    assert exc_info.value.errors(include_url=False) == [
        {'ctx': {'ge': 0}, 'input': -1, 'loc': ('a',), 'msg': 'Input should be greater than or equal to 0', 'type': 'greater_than_equal'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        Child(a=0, b=-1)
    assert exc_info.value.errors(include_url=False) == [
        {'ctx': {'ge': 0}, 'input': -1, 'loc': ('b',), 'msg': 'Input should be greater than or equal to 0', 'type': 'greater_than_equal'}
    ]


def test_annotation_inheritance() -> None:
    class A(BaseModel):
        integer: int = 1

    class B(A):
        integer: int = 2
    assert B.model_fields['integer'].annotation == int

    class C(A):
        integer: str = 'G'
    assert C.__annotations__['integer'] == str
    assert C.model_fields['integer'].annotation == str
    with pytest.raises(TypeError, match="Field 'integer' defined on a base class was overridden by a non-annotated attribute. All field definitions, including overrides, require a type annotation."):
        class D(A):
            integer = 'G'


def test_string_none() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='ignore')
        a: Optional[str] = None

    with pytest.raises(ValidationError) as exc_info:
        Model(a=None)
    assert exc_info.value.errors(include_url=False) == [
        {'input': None, 'loc': ('a',), 'msg': 'Input should be a valid string', 'type': 'string_type'}
    ]


def test_optional_required() -> None:
    class Model(BaseModel):
        bar: Optional[int]

    assert Model(bar=123).model_dump() == {'bar': 123}
    assert Model(bar=None).model_dump() == {'bar': None}
    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('bar',), 'msg': 'Field required', 'type': 'missing'}
    ]


def test_unable_to_infer() -> None:
    with pytest.raises(errors.PydanticUserError, match=re.escape(
        "A non-annotated attribute was detected: `x = None`. All model fields require a type annotation; if `x` is not meant to be a field, you may be able to resolve this error by annotating it as a `ClassVar` or updating `model_config['ignored_types']`"
    )):
        class InvalidDefinitionModel(BaseModel):
            x = None


def test_multiple_errors() -> None:
    class Model(BaseModel):
        a: Optional[int]

    with pytest.raises(ValidationError) as exc_info:
        Model(a='foobar')
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_parsing', 'loc': ('a', 'int'), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'foobar'},
        {'type': 'float_parsing', 'loc': ('a', 'float'), 'msg': 'Input should be a valid number, unable to parse string as a number', 'input': 'foobar'},
        {'type': 'decimal_parsing', 'loc': ('a', 'decimal'), 'msg': 'Input should be a valid decimal', 'input': 'foobar'}
    ]
    assert Model(a=1.5).a == 1.5
    assert Model(a=None).a is None


def test_validate_default() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_default=True)
        a: int
        b: int

    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('a',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {}, 'loc': ('b',), 'msg': 'Field required', 'type': 'missing'}
    ]


def test_force_extra() -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='ignore')

    assert Model.model_config['extra'] == 'ignore'


def test_submodel_different_type() -> None:
    class Foo(BaseModel):
        a: int

    class Bar(BaseModel):
        b: int

    class Spam(BaseModel):
        c: Union[Foo, Bar]

    assert Spam(c={'a': '123'}).model_dump() == {'c': {'a': 123}}
    with pytest.raises(ValidationError):
        Spam(c={'b': '123'})
    assert Spam(c=Foo(a='123')).model_dump() == {'c': {'a': 123}}
    with pytest.raises(ValidationError):
        Spam(c=Bar(b='123'})


def test_self() -> None:
    class Model(BaseModel):
        self: str

    m = Model.model_validate({'self': 'some value'})
    assert m.model_dump() == {'self': 'some value'}
    assert m.self == 'some value'
    assert m.model_json_schema() == {
        'title': 'Model',
        'type': 'object',
        'properties': {'self': {'title': 'Self', 'type': 'string'}},
        'required': ['self']
    }


def test_no_name_conflict_in_constructor() -> None:
    class Model(BaseModel):
        self: int

    m = Model(**{'__pydantic_self__': 4, 'self': 2})
    assert m.self == 2


def test_self_recursive() -> None:
    class SubModel(BaseModel):
        self: str

    class Model(BaseModel):
        sm: SubModel

    m = Model.model_validate({'sm': {'self': '123'}})
    assert m.model_dump() == {'sm': {'self': 123}}


def test_custom_init() -> None:
    class Model(BaseModel):
        x: int

        def __init__(self, x: int, y: Union[int, str]) -> None:
            if isinstance(y, str):
                y = len(y)
            super().__init__(x=x + int(y))

    assert Model(x=1, y=1).x == 2
    assert Model.model_validate({'x': 1, 'y': 1}).x == 2
    assert Model.model_validate_json('{"x": 1, "y": 2}').x == 3
    assert Model.model_validate({'x': 1, 'y': 'abc'}).x == 4


def test_nested_custom_init() -> None:
    class NestedModel(BaseModel):
        modified_number: int = 1

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.modified_number += 1

    class TopModel(BaseModel):
        nest: NestedModel
        self: str

    m = TopModel.model_validate({'self': 'Top Model', 'nest': {'self': 'Nested Model', 'modified_number': 0}})
    assert m.model_dump() == {'self': 'Top Model', 'nest': {'self': 'Nested Model', 'modified_number': 1}}


def test_init_inspection() -> None:
    from typing import Any

    calls: list[dict[str, Any]] = []

    class Foobar(BaseModel):
        def __init__(self, **data: Any) -> None:
            with pytest.raises(AttributeError):
                calls.append(data)
                assert self.x  # type: ignore
            super().__init__(**data)

    Foobar(x=1)
    Foobar.model_validate({'x': 2})
    Foobar.model_validate_json('{"x": 3}')
    assert calls == [{'x': 1}, {'x': 2}, {'x': 3}]


def test_type_on_annotation() -> None:
    class FooBar:
        pass

    class Model(BaseModel):
        b: type
        d: type[FooBar]
        e: list[type[FooBar]]
        f: FooBar
        g: list[FooBar]

        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    assert Model.model_fields.keys() == set(['a', 'b', 'c', 'd', 'e', 'f', 'g'])


def test_type_union() -> None:
    class Model(BaseModel):
        a: type[int]
        b: type[str]

    m = Model(a=bytes, b=int)
    assert m.model_dump() == {'a': bytes, 'b': int}
    assert m.a == bytes


def test_type_on_none() -> None:
    class Model(BaseModel):
        a: type[None]

    Model(a=type(None))
    with pytest.raises(ValidationError) as exc_info:
        Model(a=None)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_subclass_of',
            'loc': ('a',),
            'msg': 'Input should be a subclass of NoneType',
            'input': None,
            'ctx': {'class': 'NoneType'}
        }
    ]


def test_type_on_typealias():
    Float = TypeAliasType('Float', float)

    class MyFloat(float):
        pass

    adapter = TypeAdapter(type[Float])
    adapter.validate_python(float)
    adapter.validate_python(MyFloat)
    with pytest.raises(ValidationError) as exc_info:
        adapter.validate_python(str)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_subclass_of',
            'loc': (),
            'msg': 'Input should be a subclass of float',
            'input': str,
            'ctx': {'class': 'float'}
        }
    ]


def test_type_on_annotated() -> None:
    class Model(BaseModel):
        a: type[int]

    Model(a=int)
    with pytest.raises(ValidationError) as exc_info:
        Model(a=str)
    assert exc_info.value.errors(include_url=False) == [
        {
            'type': 'is_subclass_of',
            'loc': ('a',),
            'msg': 'Input should be a subclass of int',
            'input': str,
            'ctx': {'class': 'int'}
        }
    ]


def test_type_on_generic_alias() -> None:
    class Model(BaseModel):
        pass

    with pytest.raises(PydanticUserError, match='Instead of using type[list[int]], use type[list].'):
        class ModelInvalid(BaseModel):
            v: type[list[int]]


def test_typing_type_on_generic_alias() -> None:
    class Model(BaseModel):
        pass

    with pytest.raises(PydanticUserError, match='Instead of using type[typing.List[int]], use type[list].'):
        class ModelInvalid(BaseModel):
            v: type[typing.List[int]]


def test_type_assign() -> None:
    class Parent:
        def echo(self) -> str:
            return 'parent'

    class Child(Parent):
        def echo(self) -> str:
            return 'child'

    class Different:
        def echo(self) -> str:
            return 'different'

    class Model(BaseModel):
        v: type[Parent]

    assert Model(v=Parent).v().echo() == 'parent'
    assert Model().v().echo() == 'parent'
    assert Model(v=Child).v().echo() == 'child'
    with pytest.raises(ValidationError) as exc_info:
        Model(v=Different)
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'class': 'Parent'},
            'input': HasRepr(repr(Different)),
            'loc': ('v',),
            'msg': 'Input should be a subclass of Parent',
            'type': 'is_subclass_of'
        }
    ]


def test_optional_subfields() -> None:
    class Model(BaseModel):
        a: Optional[int]

    assert Model.model_fields['a'].annotation == Optional[int]
    with pytest.raises(ValidationError) as exc_info:
        Model(a='foobar')
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'foobar', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('a',), 'msg': 'Field required', 'type': 'missing'}
    ]
    assert Model(a=None).a is None
    assert Model(a=12).a == 12


def test_validated_optional_subfields() -> None:
    class Model(BaseModel):
        a: Optional[int]

        @field_validator('a')
        @classmethod
        def check_a(cls, v: Optional[int]) -> Optional[int]:
            return v

    assert Model.model_fields['a'].annotation == Optional[int]
    with pytest.raises(ValidationError) as exc_info:
        Model(a='foobar')
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'foobar', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('a',), 'msg': 'Field required', 'type': 'missing'}
    ]
    assert Model(a=None).a is None
    assert Model(a=12).a == 12


def test_optional_field_constraints() -> None:
    class MyModel(BaseModel):
        my_int: int = Field(ge=3)

    with pytest.raises(ValidationError) as exc_info:
        MyModel(my_int=2)
    assert exc_info.value.errors(include_url=False) == [
        {
            'ctx': {'ge': 3},
            'input': 2,
            'loc': ('my_int',),
            'msg': 'Input should be greater than or equal to 3',
            'type': 'greater_than_equal'
        }
    ]


def test_field_str_shape() -> None:
    class Model(BaseModel):
        a: list[int]

    assert repr(Model.model_fields['a']) == 'FieldInfo(annotation=list[int], required=True)'
    assert str(Model.model_fields['a']) == 'annotation=list[int] required=True'


T1 = TypeVar('T1')
T2 = TypeVar('T2')


class DisplayGen(Generic[T1, T2]):
    def __init__(self, t1: T1, t2: T2) -> None:
        self.t1 = t1
        self.t2 = t2


@pytest.mark.parametrize(
    'type_,expected',
    [
        (int, 'int'),
        (Optional[int], 'Union[int, NoneType]'),
        (Union[None, int, str], 'Union[NoneType, int, str]'),
        (Union[int, str, bytes], 'Union[int, str, bytes]'),
        (list[int], 'list[int]'),
        (tuple[int, str, bytes], 'tuple[int, str, bytes]'),
        (Union[list[int], set[bytes]], 'Union[list[int], set[bytes]]'),
        (list[tuple[int, int]], 'list[tuple[int, int]]'),
        (dict[int, str], 'dict[int, str]'),
        (frozenset[int], 'frozenset[int]'),
        (tuple[int, ...], 'tuple[int, ...]'),
        (Optional[list[int]], 'Union[list[int], NoneType]'),
        (dict, 'dict'),
        pytest.param(
            DisplayGen[bool, str],
            'tests.test_edge_cases.DisplayGen[bool, str]',
            marks=pytest.mark.skipif(sys.version_info[:2] > (3, 9), reason='difference in __name__ between versions')
        )
    ]
)
def test_field_type_display(type_: Any, expected: str) -> None:
    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    assert re.search(f'\\(annotation={re.escape(expected)},', str(Model.model_fields)) is not None


def test_any_none() -> None:
    class MyModel(BaseModel):
        foo: Any

    m = MyModel(foo=None)
    assert dict(m) == {'foo': None}


def test_type_var_any() -> None:
    Foobar = TypeVar('Foobar')

    class MyModel(BaseModel):
        foo: Foobar

    assert MyModel.model_json_schema() == {
        'properties': {'foo': {'title': 'Foo'}},
        'required': ['foo'],
        'title': 'MyModel',
        'type': 'object'
    }
    assert MyModel(foo=None).foo is None
    assert MyModel(foo='x').foo == 'x'
    assert MyModel(foo=123).foo == 123


def test_type_var_constraint() -> None:
    Foobar = TypeVar('Foobar', int, str)

    class MyModel(BaseModel):
        foo: Foobar

    assert MyModel.model_json_schema() == {
        'title': 'MyModel',
        'type': 'object',
        'properties': {
            'foo': {
                'title': 'Foo',
                'anyOf': [
                    {'type': 'integer'},
                    {'type': 'string'}
                ]
            }
        },
        'required': ['foo']
    }
    with pytest.raises(ValidationError) as exc_info:
        MyModel(foo=None)
    assert exc_info.value.errors(include_url=False) == [
        {'input': None, 'loc': ('foo', 'int'), 'msg': 'Input should be a valid integer', 'type': 'int_type'},
        {'input': None, 'loc': ('foo', 'str'), 'msg': 'Input should be a valid string', 'type': 'string_type'}
    ]
    with pytest.raises(ValidationError):
        MyModel(foo=[1, 2, 3])
    assert exc_info.value.errors(include_url=False) == [
        {'input': None, 'loc': ('foo', 'int'), 'msg': 'Input should be a valid integer', 'type': 'int_type'},
        {'input': None, 'loc': ('foo', 'str'), 'msg': 'Input should be a valid string', 'type': 'string_type'}
    ]
    assert MyModel(foo='x').foo == 'x'
    assert MyModel(foo=123).foo == 123


def test_type_var_bound() -> None:
    Foobar = TypeVar('Foobar', bound=int)

    class MyModel(BaseModel):
        foo: Foobar

    assert MyModel.model_json_schema() == {
        'title': 'MyModel',
        'type': 'object',
        'properties': {
            'foo': {
                'title': 'Foo',
                'type': 'integer'
            }
        },
        'required': ['foo']
    }
    with pytest.raises(ValidationError) as exc_info:
        MyModel(foo=None)
    assert exc_info.value.errors(include_url=False) == [
        {'input': None, 'loc': ('foo',), 'msg': 'Input should be a valid integer', 'type': 'int_type'}
    ]
    with pytest.raises(ValidationError):
        MyModel(foo='x')
    assert exc_info.value.errors(include_url=False) == [
        {'input': None, 'loc': ('foo',), 'msg': 'Input should be a valid integer', 'type': 'int_type'}
    ]
    assert MyModel(foo=123).foo == 123


def test_dict_bare() -> None:
    class MyModel(BaseModel):
        foo: dict

    m = MyModel(foo={'x': 'a', 'y': None})
    assert m.foo == {'x': 'a', 'y': None}


def test_list_bare() -> None:
    class MyModel(BaseModel):
        foo: list

    m = MyModel(foo=[1, 2, None])
    assert m.foo == [1, 2, None]


def test_dict_any() -> None:
    class MyModel(BaseModel):
        v: dict[Any, Any]

    m = MyModel(v={'x': 'a', 'y': None})
    assert m.v == {'x': 'a', 'y': None}


def test_modify_fields() -> None:
    class Foo(BaseModel):
        foo: list[list[int]]

        @field_validator('foo')
        @classmethod
        def check_something(cls, value: list[list[int]]) -> list[list[int]]:
            return value

    class Bar(Foo):
        pass

    assert repr(Foo.model_fields['foo']) == 'FieldInfo(annotation=list[list[int]], required=True)'
    assert repr(Bar.model_fields['foo']) == 'FieldInfo(annotation=list[list[int]], required=True)'
    assert Foo(foo=[[0, 1]]).foo == [[0, 1]]
    assert Bar(foo=[[0, 1]]).foo == [[0, 1]]


def test_exclude_none() -> None:
    class MyModel(BaseModel):
        a: Optional[int] = None
        b: int = 2

    m = MyModel(a=5)
    assert m.model_dump(exclude_none=True) == {'a': 5, 'b': 2}
    m = MyModel(b=3)
    assert m.model_dump(exclude_none=True) == {'b': 3}
    assert m.model_dump_json(exclude_none=True) == '{"b":3}'


def test_exclude_none_recursive() -> None:
    class ModelA(BaseModel):
        a: Optional[int] = None
        b: int = 1

    class ModelB(BaseModel):
        c: int
        d: int = 2
        e: ModelA
        f: Optional[str] = None

    m = ModelB(c=5, e={'a': 0})
    assert m.model_dump() == {'c': 5, 'd': 2, 'e': {'a': 0, 'b': 1}, 'f': None}
    assert m.model_dump(exclude_none=True) == {'c': 5, 'd': 2, 'e': {'a': 0, 'b': 1}}
    assert dict(m) == {'c': 5, 'd': 2, 'e': ModelA(a=0), 'f': None}
    m = ModelB(c=5, e={'b': 20}, f='test')
    assert m.model_dump() == {'c': 5, 'd': 2, 'e': {'a': None, 'b': 20}, 'f': 'test'}
    assert m.model_dump(exclude_none=True) == {'c': 5, 'd': 2, 'e': {'b': 20}, 'f': 'test'}
    assert dict(m) == {'c': 5, 'd': 2, 'e': ModelA(b=20), 'f': 'test'}


def test_exclude_none_with_extra() -> None:
    class MyModel(BaseModel):
        model_config: ConfigDict = ConfigDict(extra='allow')
        a: Optional[str] = 'default'
        b: Optional[str] = None

    m = MyModel(a='a', c='c')
    assert m.model_dump(exclude_none=True) == {'a': 'a', 'c': 'c'}
    assert m.model_dump() == {'a': 'a', 'b': None, 'c': 'c'}
    m = MyModel(a='a', b='b', c=None)
    assert m.model_dump(exclude_none=True) == {'a': 'a', 'b': 'b'}
    assert m.model_dump() == {'a': 'a', 'b': 'b', 'c': None}


def test_str_method_inheritance() -> None:
    import pydantic

    class Foo(pydantic.BaseModel):
        x: int = 3
        y: int = 4

        def __str__(self) -> str:
            return str(self.y + self.x)

    class Bar(Foo):
        z: bool = False

    assert str(Foo()) == '7'
    assert str(Bar()) == '7'


def test_repr_method_inheritance() -> None:
    import pydantic

    class Foo(pydantic.BaseModel):
        x: int = 3
        y: int = 4

        def __repr__(self) -> str:
            return repr(self.y + self.x)

    class Bar(Foo):
        z: bool = False

    assert repr(Foo()) == '7'
    assert repr(Bar()) == '7'


def test_optional_validator() -> None:
    val_calls: list[Any] = []

    class Model(BaseModel):
        something: Optional[Any]

        @field_validator('something')
        @classmethod
        def check_something(cls, v: Optional[Any]) -> Optional[Any]:
            val_calls.append(v)
            return v

    with pytest.raises(ValidationError):
        assert Model().model_dump() == {'something': None}
    assert Model(something=None).model_dump() == {'something': None}
    assert Model(something='hello').model_dump() == {'something': 'hello'}
    assert val_calls == [None, 'hello']


def test_required_optional() -> None:
    class Model(BaseModel):
        nullable1: Optional[int] = None
        nullable2: Optional[int] = Field(None)
        nullable3: Optional[int] = None
        nullable4: Optional[int] = Field(None)

    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('nullable1',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {}, 'loc': ('nullable2',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {}, 'loc': ('nullable3',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {}, 'loc': ('nullable4',), 'msg': 'Field required', 'type': 'missing'},
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model(nullable1=1)
    assert exc_info.value.errors(include_url=False) == [
        {'input': {'nullable1': 1}, 'loc': ('nullable2',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'nullable1': 1}, 'loc': ('nullable3',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'nullable1': 1}, 'loc': ('nullable4',), 'msg': 'Field required', 'type': 'missing'},
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model(nullable2=2)
    assert exc_info.value.errors(include_url=False) == [
        {'input': {'nullable2': 2}, 'loc': ('nullable1',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'nullable2': 2}, 'loc': ('nullable3',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'nullable2': 2}, 'loc': ('nullable4',), 'msg': 'Field required', 'type': 'missing'},
    ]
    assert Model(nullable1=None, nullable2=None).model_dump() == {
        'nullable1': None,
        'nullable2': None,
        'nullable3': None,
        'nullable4': None
    }
    assert Model(nullable1=1, nullable2=2, nullable3=3, nullable4=4).model_dump() == {
        'nullable1': 1,
        'nullable2': 2,
        'nullable3': 3,
        'nullable4': 4
    }
    with pytest.raises(ValidationError) as exc_info:
        Model(nullable1='some text')
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'some text', 'loc': ('nullable1',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'},
        {'input': {'nullable1': 'some text'}, 'loc': ('nullable2',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'nullable1': 'some text'}, 'loc': ('nullable3',), 'msg': 'Field required', 'type': 'missing'},
        {'input': {'nullable1': 'some text'}, 'loc': ('nullable4',), 'msg': 'Field required', 'type': 'missing'},
    ]


def test_custom_generic_validators() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    class MyGen(Generic[T1, T2]):
        t1: T1
        t2: T2

        def __init__(self, t1: T1, t2: T2) -> None:
            self.t1 = t1
            self.t2 = t2

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> Any:
            schema = core_schema.is_instance_schema(cls)
            args = get_args(source)
            if not args:
                return schema

            t1_f = TypeAdapter(get_args(source)[0]).validate_python
            t2_f = TypeAdapter(get_args(source)[1]).validate_python

            def convert_to_init_error(e: dict[str, Any], loc: str) -> dict[str, Any]:
                init_e = {'type': e['type'], 'loc': e['loc'] + (loc,), 'input': e['input']}
                if 'ctx' in e:
                    init_e['ctx'] = e['ctx']
                return init_e

            def validate(v: MyGen[T1, T2], _info: Any) -> MyGen[T1, T2]:
                if not args:
                    return v
                try:
                    v.t1 = t1_f(v.t1)
                except ValidationError as exc:
                    raise ValidationError.from_exception_data(
                        exc.title,
                        [convert_to_init_error(e, 't1') for e in exc.errors()]
                    ) from exc
                try:
                    v.t2 = t2_f(v.t2)
                except ValidationError as exc:
                    raise ValidationError.from_exception_data(
                        exc.title,
                        [convert_to_init_error(e, 't2') for e in exc.errors()]
                    ) from exc
                return v

            return core_schema.with_info_after_validator_function(validate, schema)

    class Model(BaseModel):
        a: str
        gen: MyGen[str, bool]
        gen2: MyGen[int, int]

        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    with pytest.raises(ValidationError) as exc_info:
        Model(a='foo', gen='invalid', gen2='invalid')
    assert exc_info.value.errors(include_url=False) == [
        {'ctx': {'class': 'test_custom_generic_validators.<locals>.MyGen'}, 'input': 'invalid', 'loc': ('gen',), 'msg': 'Input should be an instance of test_custom_generic_validators.<locals>.MyGen', 'type': 'is_instance_of'},
        {'ctx': {'class': 'test_custom_generic_validators.<locals>.MyGen'}, 'input': 'invalid', 'loc': ('gen2',), 'msg': 'Input should be an instance of test_custom_generic_validators.<locals>.MyGen', 'type': 'is_instance_of'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model(a='foo', gen=MyGen(t1='bar', t2='baz'), gen2=MyGen(t1='bar', t2='baz'))
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'baz', 'loc': ('gen', 't2'), 'msg': 'Input should be a valid bool, unable to interpret input', 'type': 'bool_parsing'}
    ]
    m = Model(a='foo', gen=MyGen(t1='bar', t2=True), gen2=MyGen(t1=1, t2=2))
    assert m.a == 'foo'
    assert m.gen.t1 == 'bar'
    assert m.gen.t2 is True
    assert m.gen2.t1 == 1
    assert m.gen2.t2 == 2


def test_custom_generic_arbitrary_allowed() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    class MyGen(Generic[T1, T2]):
        t1: T1
        t2: T2

        def __init__(self, t1: T1, t2: T2) -> None:
            self.t1 = t1
            self.t2 = t2

    class Model(BaseModel):
        a: str
        gen: MyGen[str, Any]

        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    with pytest.raises(ValidationError) as exc_info:
        Model(a='foo', gen='invalid')
    assert exc_info.value.errors(include_url=False) == [
        {'ctx': {'class': 'test_custom_generic_arbitrary_allowed.<locals>.MyGen'}, 'input': 'invalid', 'loc': ('gen',), 'msg': 'Input should be an instance of test_custom_generic_arbitrary_allowed.<locals>.MyGen', 'type': 'is_instance_of'}
    ]
    m = Model(a='foo', gen=MyGen(t1='bar', t2='baz'))
    assert m.a == 'foo'
    assert m.gen.t1 == 'bar'
    assert m.gen.t2 == 'baz'
    m = Model(a='foo', gen=MyGen(t1='bar', t2=True))
    assert m.a == 'foo'
    assert m.gen.t1 == 'bar'
    assert m.gen.t2 is True


def test_custom_generic_disallowed() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    class MyGen(Generic[T1, T2]):
        t1: T1
        t2: T2

        def __init__(self, t1: T1, t2: T2) -> None:
            self.t1 = t1
            self.t2 = t2

    match = 'Unable to generate pydantic-core schema for .*MyGen\[str, bool\]. Set `arbitrary_types_allowed=True` in the model_config to ignore this error'
    with pytest.raises(TypeError, match=match):
        class Model(BaseModel):
            v: MyGen[str, bool]


def test_hashable_required() -> None:
    class Model(BaseModel):
        v: Hashable

    Model(v=None)
    with pytest.raises(ValidationError) as exc_info:
        Model(v=[])
    assert exc_info.value.errors(include_url=False) == [
        {'input': [], 'loc': ('v',), 'msg': 'Input should be hashable', 'type': 'is_hashable'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        Model()
    assert exc_info.value.errors(include_url=False) == [
        {'input': {}, 'loc': ('v',), 'msg': 'Field required', 'type': 'missing'}
    ]


@pytest.mark.parametrize(
    'default',
    [1, None]
)
def test_hashable_optional(default: Optional[int]) -> None:
    class Model(BaseModel):
        v: Optional[Hashable] = default

    Model(v=None)
    Model()


def test_hashable_serialization() -> None:
    class Model(BaseModel):
        v: Hashable

    class HashableButNotSerializable:
        def __hash__(self) -> int:
            return 0

    assert Model(v=(1,)).model_dump_json() == '{"v":[1]}'
    m = Model(v=HashableButNotSerializable())
    with pytest.raises(PydanticSerializationError, match='Unable to serialize unknown type:.*HashableButNotSerializable'):
        m.model_dump_json()


def test_hashable_validate_json() -> None:
    class Model(BaseModel):
        v: Hashable

    ta = TypeAdapter(Model)
    for validate in (Model.model_validate_json, ta.validate_json):
        for testcase in ('{"v": "a"}', '{"v": 1}', '{"v": 1.0}', '{"v": true}', '{"v": null}'):
            assert hash(validate(testcase).v) == hash(validate(testcase).v)


@pytest.mark.parametrize(
    'non_hashable',
    ['{"v": []}', '{"v": {"a": 0}}']
)
def test_hashable_invalid_json(non_hashable: str) -> None:
    class Model(BaseModel):
        v: Hashable

    with pytest.raises(ValidationError):
        Model.model_validate_json(non_hashable)


def test_hashable_json_schema() -> None:
    class Model(BaseModel):
        v: Hashable

    assert Model.model_json_schema() == {
        'properties': {'v': {'title': 'V'}},
        'required': ['v'],
        'title': 'Model',
        'type': 'object'
    }


def test_default_factory_called_once() -> None:
    v = 0

    def factory() -> int:
        nonlocal v
        v += 1
        return v

    class MyModel(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_default=True)
        id: int = Field(default_factory=factory)

    m1 = MyModel()
    assert m1.id == 1

    class MyBadModel(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_default=True)
        id: int = Field(default_factory=factory)

    with pytest.raises(ValidationError) as exc_info:
        MyBadModel()
    assert v == 2
    assert exc_info.value.errors(include_url=False) == [
        {'input': 2, 'loc': ('id',), 'msg': 'Input should be a valid list', 'type': 'list_type'}
    ]


def test_default_factory_validator_child() -> None:
    class Parent(BaseModel):
        foo: list[str] = Field(default_factory=list)

        @field_validator('foo', mode='before')
        @classmethod
        def mutate_foo(cls, v: list[str]) -> list[str]:
            return [f'{x}-1' for x in v]

    assert Parent(foo=['a', 'b']).foo == ['a-1', 'b-1']

    class Child(Parent):
        pass

    assert Child(foo=['a', 'b']).foo == ['a-1', 'b-1']


def test_resolve_annotations_module_missing(tmp_path: Any) -> None:
    file_path = tmp_path / 'module_to_load.py'
    file_path.write_text(
        "\nfrom pydantic import BaseModel\nclass User(BaseModel):\n    id: int\n    name: str = 'Jane Doe'\n"
    )
    spec = importlib.util.spec_from_file_location('my_test_module', file_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    assert module.User(id=12).model_dump() == {'id': 12, 'name': 'Jane Doe'}


def test_iter_coverage() -> None:
    class MyModel(BaseModel):
        x: int = 1
        y: str = 'a'

    with pytest.warns(PydanticDeprecatedSince20, match='The private method `_iter` will be removed and should no longer be used.'):
        assert list(MyModel()._iter(by_alias=True)) == [('x', 1), ('y', 'a')]


def test_frozen_config_and_field() -> None:
    class Foo(BaseModel):
        model_config: ConfigDict = ConfigDict(frozen=False, validate_assignment=True)
        a: Any = None

    assert Foo.model_fields['a'].metadata == []
    f = Foo(a='x')
    f.a = 'y'
    assert f.model_dump() == {'a': 'y'}

    class Bar(BaseModel):
        model_config: ConfigDict = ConfigDict(validate_assignment=True)
        a: Any = Field(default='x', frozen=True)

    assert Bar.model_fields['a'].frozen
    b = Bar(a='x', c='z')
    with pytest.raises(ValidationError) as exc_info:
        b.a = 'y'
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'y', 'loc': ('a',), 'msg': 'Field is frozen', 'type': 'frozen_field'}
    ]
    with pytest.raises(ValidationError) as exc_info:
        b.c = 'y'
    assert exc_info.value.errors(include_url=False) == [
        {'input': 'y', 'loc': ('c',), 'msg': 'Field is frozen', 'type': 'frozen_field'}
    ]
    assert b.model_dump() == {'a': 'x', 'c': 'z'}


def test_arbitrary_types_allowed_custom_eq() -> None:
    class Foo:
        def __eq__(self, other: Any) -> bool:
            if other.__class__ is not Foo:
                raise TypeError(f'Cannot interpret {other.__class__.__name__!r} as a valid type')
            return True

    class Model(BaseModel):
        model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
        x: Foo

    assert Model().x == Foo()


def test_bytes_subclass() -> None:
    class MyModel(BaseModel):
        my_bytes: bytes

    class BytesSubclass(bytes):
        def __new__(cls, data: bytes) -> 'BytesSubclass':
            self = bytes.__new__(cls, data)
            return self

    m = MyModel(my_bytes=BytesSubclass(b'foobar'))
    assert m.my_bytes.__class__ == BytesSubclass


def test_int_subclass() -> None:
    class MyModel(BaseModel):
        my_int: int

    class IntSubclass(int):
        def __new__(cls, data: int) -> 'IntSubclass':
            self = int.__new__(cls, data)
            return self

    m = MyModel(my_int=IntSubclass(123))
    assert m.my_int.__class__ != IntSubclass
    assert isinstance(m.my_int, int)


def test_model_issubclass() -> None:
    assert not issubclass(int, BaseModel)

    class MyModel(BaseModel):
        pass

    assert issubclass(MyModel, BaseModel)

    class Custom:
        __fields__ = True

    assert not issubclass(Custom, BaseModel)


def test_long_int() -> None:
    class Model(BaseModel):
        x: int

    assert Model(x='1' * 4300).x == int('1' * 4300)
    too_long = '1' * 4301
    with pytest.raises(ValidationError) as exc_info:
        Model(x=too_long)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'int_parsing_size', 'loc': ('x',), 'msg': 'Unable to parse input string as an integer, exceeded maximum size', 'input': too_long}
    ]
    with pytest.raises(ValidationError):
        Model(x='1' * 10**7)


def test_parent_field_with_default() -> None:
    class Parent(BaseModel):
        a: int = 1
        b: int = Field(2)

    class Child(Parent):
        c: int = 3

    c = Child()
    assert c.a == 1
    assert c.b == 2
    assert c.c == 3


@pytest.mark.skipif(sys.version_info < (3, 12), reason='requires Python 3.12+')
@pytest.mark.parametrize(
    'bases',
    [
        (BaseModel, ABC),
        (ABC, BaseModel),
        (BaseModel,),
    ]
)
def test_abstractmethod_missing_for_all_decorators(bases: tuple[Type[Any], ...]) -> None:
    class AbstractSquare(*bases):
        @field_validator('side')
        @classmethod
        @abstractmethod
        def my_field_validator(cls, v: Any) -> Any:
            raise NotImplementedError

        @model_validator(mode='wrap')
        @classmethod
        @abstractmethod
        def my_model_validator(cls, values: Any, handler: Any, info: Any) -> Any:
            raise NotImplementedError

        with pytest.warns(PydanticDeprecatedSince20):
            @root_validator(skip_on_failure=True)
            @classmethod
            @abstractmethod
            def my_root_validator(cls, values: Any) -> Any:
                raise NotImplementedError

        with pytest.warns(PydanticDeprecatedSince20):
            @validator('side')
            @classmethod
            @abstractmethod
            def my_validator(cls, value: Any, **kwargs: Any) -> Any:
                raise NotImplementedError

        @model_serializer(mode='wrap')
        @abstractmethod
        def my_model_serializer(self, handler: Any, info: Any) -> Any:
            raise NotImplementedError

        @field_serializer('side')
        @abstractmethod
        def my_serializer(self, v: Any, _info: Any) -> Any:
            raise NotImplementedError

        @computed_field
        @property
        @abstractmethod
        def my_computed_field(self) -> Any:
            raise NotImplementedError

    class Square(AbstractSquare):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class Square without an implementation for abstract methods 'my_computed_field', 'my_field_validator', 'my_model_serializer', 'my_model_validator', 'my_root_validator', 'my_serializer', 'my_validator'"):
        Square(side=1.0)


def test_generic_wrapped_forwardref() -> None:
    class Operation(BaseModel):
        callbacks: list['PathItem']

    class PathItem(BaseModel):
        pass

    Operation.model_rebuild()
    Operation.model_validate({'callbacks': [PathItem()]})
    with pytest.raises(ValidationError) as exc_info:
        Operation.model_validate({'callbacks': [1]})
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'model_type', 'loc': ('callbacks', 0), 'msg': 'Input should be a valid dictionary or instance of PathItem', 'input': 1, 'ctx': {'class_name': 'PathItem'}}
    ]


def test_plain_basemodel_field() -> None:
    class Model(BaseModel):
        x: BaseModel

    class Model2(BaseModel):
        pass

    assert Model(x=Model2()).x == Model2()
    with pytest.raises(ValidationError) as exc_info:
        Model(x=1)
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'model_type', 'loc': ('x',), 'msg': 'Input should be a valid dictionary or instance of BaseModel', 'input': 1, 'ctx': {'class_name': 'BaseModel'}}
    ]


def test_invalid_forward_ref_model() -> None:
    if sys.version_info >= (3, 11):
        error = errors.PydanticUserError
    else:
        error = TypeError
    with pytest.raises(error):
        class M(BaseModel):
            B: Any = Field(default=None)

    class A(BaseModel):
        B: ForwardRef('B')

    assert A.model_fields['B'].annotation == ForwardRef('__types["B"]')
    A.model_rebuild(raise_errors=False)
    assert A.model_fields['B'].annotation == ForwardRef('__types["B"]')

    class B(BaseModel):
        pass

    class C(BaseModel):
        pass

    assert not A.__pydantic_complete__
    types = {'B': B}
    A.model_rebuild(_types_namespace={'__types': types})
    assert A.__pydantic_complete__
    assert A(B=B()).B == B()
    with pytest.raises(ValidationError) as exc_info:
        A(B=C())
    assert exc_info.value.errors(include_url=False) == [
        {'type': 'model_type', 'loc': ('B',), 'msg': 'Input should be a valid dictionary or instance of B', 'input': C(), 'ctx': {'class_name': 'B'}}
    ]


@pytest.mark.parametrize(
    'sequence_type,input_data,expected_error_type,expected_error_msg,expected_error_ctx',
    [
        pytest.param(
            list[str],
            '1bc',
            'list_type',
            'Input should be a valid list',
            None,
            id='list[str]'
        ),
        pytest.param(
            Sequence[str],
            '1bc',
            'sequence_str',
            "'str' instances are not allowed as a Sequence value",
            {'type_name': 'str'},
            id='Sequence[str]'
        ),
        pytest.param(
            Sequence[bytes],
            b'1bc',
            'sequence_str',
            "'bytes' instances are not allowed as a Sequence value",
            {'type_name': 'bytes'},
            id='Sequence[bytes]'
        ),
    ]
)
def test_sequences_str(
    sequence_type: Any,
    input_data: Any,
    expected_error_type: str,
    expected_error_msg: str,
    expected_error_ctx: Optional[dict[str, Any]]
) -> None:
    class Model(BaseModel):
        str_sequence: sequence_type

    input_sequence = [input_data[:1], input_data[1:]]
    expected_error = {
        'type': expected_error_type,
        'input': input_data,
        'loc': ('str_sequence',),
        'msg': expected_error_msg
    }
    if expected_error_ctx is not None:
        expected_error['ctx'] = expected_error_ctx

    assert Model(str_sequence=input_sequence).str_sequence == input_sequence
    with pytest.raises(ValidationError) as e:
        Model(str_sequence=input_data)
    assert e.value.errors(include_url=False) == [expected_error]


def test_multiple_enums() -> None:
    class MyEnum(Enum):
        a = auto()

    class MyModel(TypedDict):
        a: MyEnum
        b: MyEnum

    ta = TypeAdapter(MyModel)
    assert ta.validate_json('{"a": 1, "b": 1}') == {'a': MyEnum.a, 'b': MyEnum.a}


@pytest.mark.parametrize(
    'literal_type,other_type,data,json_value,data_reversed,json_value_reversed',
    [
        (Literal[False], str, False, 'false', False, 'false'),
        (Literal[True], str, True, 'true', True, 'true'),
        (Literal[False], str, 'abc', '"abc"', 'abc', '"abc"'),
        (Literal[False], int, False, 'false', False, 'false'),
        (Literal[True], int, True, 'true', True, 'true'),
        (Literal[False], int, 42, '42', 42, '42'),
    ]
)
def test_union_literal_with_other_type(
    literal_type: Any,
    other_type: Any,
    data: Any,
    json_value: str,
    data_reversed: Any,
    json_value_reversed: str
) -> None:
    class Model(BaseModel):
        value: Union[literal_type, other_type]
        value_types_reversed: Union[other_type, literal_type]

    m = Model(value=data, value_types_reversed=data_reversed)
    assert m.model_dump() == {'value': data, 'value_types_reversed': data_reversed}
    assert m.model_dump_json() == f'{{"value":{json_value},"value_types_reversed":{json_value_reversed}}}'


def test_model_repr_before_validation() -> None:
    class MyModel(BaseModel):
        x: int = 0

        def __init__(self, **kwargs: Any) -> None:
            log.append(f'before={self!r}')
            super().__init__(**kwargs)
            log.append(f'after={self!r}')

    log: list[str] = []
    m = MyModel(x=10)
    assert m.x == 10
    assert log == ['before=MyModel()', 'after=MyModel(x=10)']


def test_custom_exception_handler() -> None:
    from traceback import TracebackException
    from pydantic import BaseModel
    from typing import Any

    traceback_exceptions: list[TracebackException] = []

    class MyModel(BaseModel):
        pass

    class CustomErrorCatcher:
        def __enter__(self) -> None:
            return None

        def __exit__(self, _exception_type: Any, exception: Any, exception_traceback: Any) -> bool:
            if exception is not None:
                te = TracebackException(
                    exc_type=type(exception),
                    exc_value=exception,
                    exc_traceback=exception_traceback,
                    capture_locals=True
                )
                traceback_exceptions.append(te)
                return True
            return False

    with CustomErrorCatcher():
        data = {'age': 'John Doe'}
        MyModel(**data)
    assert len(traceback_exceptions) == 1


def test_recursive_walk_fails_on_double_diamond_composition() -> None:
    class A(BaseModel):
        pass

    class B(BaseModel):
        pass

    class C(BaseModel):
        pass

    class D(BaseModel):
        pass

    class E(BaseModel):
        c: C
        d: D

    assert E(c=C(b=B(a_1=A(), a_2=A())), d=D()).model_dump() == {'c': {'b': {'a_1': {}, 'a_2': {}}}, 'd': {}}


def test_recursive_root_models_in_discriminated_union() -> None:
    class Model1(BaseModel):
        kind: Literal['1']

    class Model2(BaseModel):
        kind: Literal['2']

    class Root1(RootModel[Model1]):
        @property
        def kind(self) -> str:
            return self.root.kind

    class Root2(RootModel[Model2]):
        @property
        def kind(self) -> str:
            return self.root.kind

    class Outer(BaseModel):
        a: Union[Root1, Root2]
        b: Union[Root1, Root2]

    validated = Outer.model_validate({'a': {'kind': '1', 'two': None}, 'b': {'kind': '2', 'one': None}})
    assert validated == Outer(a=Root1(root=Model1(kind='1')), b=Root2(root=Model2(kind='2')))
    assert Outer.model_json_schema() == {
        '$defs': {
            'Model1': {
                'properties': {
                    'kind': {'const': '1', 'default': '1', 'title': 'Kind', 'type': 'string'},
                    'two': {'anyOf': [{'$ref': '#/$defs/Model2'}, {'type': 'null'}]}
                },
                'required': ['two'],
                'title': 'Model1',
                'type': 'object'
            },
            'Model2': {
                'properties': {
                    'kind': {'const': '2', 'default': '2', 'title': 'Kind', 'type': 'string'},
                    'one': {'anyOf': [{'$ref': '#/$defs/Model1'}, {'type': 'null'}]}
                },
                'required': ['one'],
                'title': 'Model2',
                'type': 'object'
            },
            'Root1': {'$ref': '#/$defs/Model1', 'title': 'Root1'},
            'Root2': {'$ref': '#/$defs/Model2', 'title': 'Root2'}
        },
        'properties': {
            'a': {
                'discriminator': {'mapping': {'1': '#/$defs/Root1', '2': '#/$defs/Root2'}, 'propertyName': 'kind'},
                'oneOf': [{'$ref': '#/$defs/Root1'}, {'$ref': '#/$defs/Root2'}],
                'title': 'A'
            },
            'b': {
                'discriminator': {'mapping': {'1': '#/$defs/Root1', '2': '#/$defs/Root2'}, 'propertyName': 'kind'},
                'oneOf': [{'$ref': '#/$defs/Root1'}, {'$ref': '#/$defs/Root2'}],
                'title': 'B'
            }
        },
        'required': ['a', 'b'],
        'title': 'Outer',
        'type': 'object'
    }


def test_eq_with_cached_property() -> None:
    from functools import cached_property

    class Model(BaseModel):
        @cached_property
        def cached(self) -> int:
            return 0

    obj1 = Model(attr=1)
    obj2 = Model(attr=1)
    assert obj1 == obj2
    obj1.cached
    assert obj1 == obj2


def test_model_metaclass_on_other_class() -> None:
    class OtherClass(metaclass=ModelMetaclass):
        pass


@pytest.mark.skipif(sys.version_info < (3, 12), reason='error message different on older versions')
def test_nested_type_statement() -> None:
    globs: dict[str, Any] = {}
    exec('\nfrom pydantic import BaseModel\nclass A(BaseModel):\n    type Int = int\n    a: Int\n', globs)
    A = globs['A']
    assert A(a=1).a == 1


def test_method_descriptors_default() -> None:
    class SomeModel(BaseModel):
        @staticmethod
        def default_int_factory() -> Callable[[], int]:
            return lambda: 0

        int_factory: Callable[[], int] = Field(default=default_int_factory)

    assert SomeModel.model_fields['int_factory'].default is SomeModel.default_int_factory


def test_setattr_handler_memo_does_not_inherit() -> None:
    class Model1(BaseModel):
        a: int

    class Model2(Model1):
        a: int

    m1 = Model1(a=1)
    m2 = Model2(a=10)
    assert not Model1.__pydantic_setattr_handlers__
    assert not Model2.__pydantic_setattr_handlers__
    m2.a = 11
    assert not Model1.__pydantic_setattr_handlers__
    assert 'a' in Model2.__pydantic_setattr_handlers__
    handler2 = Model2.__pydantic_setattr_handlers__['a']
    m1.a = 2
    assert 'a' in Model1.__pydantic_setattr_handlers__
    assert Model1.__pydantic_setattr_handlers__['a'] is handler2
    assert Model2.__pydantic_setattr_handlers__['a'] is handler2
    assert m1.a == 2 and m2.a == 11


def test_setattr_handler_does_not_memoize_unknown_field() -> None:
    class Model(BaseModel):
        a: int

    m = Model(a=1)
    with pytest.raises(ValueError, match='object has no field "unknown"'):
        m.unknown = 'x'
    assert not Model.__pydantic_setattr_handlers__
    m.a = 2
    assert 'a' in Model.__pydantic_setattr_handlers__


def test_setattr_handler_does_not_memoize_unknown_private_field() -> None:
    class Model(BaseModel):
        a: int

    m = Model(a=1)
    assert not Model.__pydantic_setattr_handlers__
    m.a = 2
    assert len(Model.__pydantic_setattr_handlers__) == 1
    m._unknown = 'x'
    assert len(Model.__pydantic_setattr_handlers__) == 1
    m._p = 'y'
    assert len(Model.__pydantic_setattr_handlers__) == 2


def test_setattr_handler_does_not_memoize_on_validate_assignment_field_failure() -> None:
    class Model(BaseModel):
        a: int

        model_config: ConfigDict = ConfigDict(validate_assignment=True)

    m = Model(a=1)
    with pytest.raises(ValidationError):
        m.unknown = 'x'
    with pytest.raises(ValidationError):
        m.a = 'y'
    assert not Model.__pydantic_setattr_handlers__
    m.a = 2
    assert 'a' in Model.__pydantic_setattr_handlers__


def test_get_pydantic_core_schema_on_referenceable_type() -> None:
    counter = 0

    class Model(BaseModel):
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> Any:
            nonlocal counter
            rv = handler(source)
            counter += 1
            return rv

    counter = 0

    class Test(Model):
        pass

    assert counter == 1


def test_validator_and_serializer_not_reused_during_rebuild() -> None:
    class Model(BaseModel):
        a: int

    Model.model_fields['a'].exclude = True
    Model.model_rebuild(force=True)
    m = Model(a=1)
    assert m.model_dump() == {}


def test_any_none() -> None:
    class MyModel(BaseModel):
        foo: Any

    m = MyModel(foo=None)
    assert dict(m) == {'foo': None}


def test_now_field_alias_conflict():
    class MyModel(BaseModel):
        model_config = ConfigDict(extra='ignore')
        a: Optional[int] = None

    with pytest.raises(ValidationError) as exc_info:
        MyModel(field_set=2)
    assert exc_info.value.errors(include_url=False) == [
        {'input': 2, 'loc': ('field_set',), 'msg': 'Input should be a valid int', 'type': 'int_type'}
    ]


def test_default_factory_factory() -> None:
    from typing import Callable

    def factory() -> Callable[[], int]:
        return lambda: 0

    class MyModel(BaseModel):
        a: Callable[[], int] = Field(default_factory=factory)

    m = MyModel()
    assert m.a() == 0


def test_model_rebuild_with_force() -> None:
    class Model(BaseModel):
        a: int
        b: int

    Model.model_rebuild(force=True)
    m = Model(a=1, b=2)
    assert m.model_dump() == {'a': 1, 'b': 2}

    Model.model_fields['a'].exclude = True
    Model.model_rebuild(force=True)
    m = Model(a=1, b=2)
    assert m.model_dump() == {'b': 2}


def test_computed_field() -> None:
    class Model(BaseModel):
        a: int

        @computed_field
        @property
        def b(self) -> int:
            return self.a * 2

    m = Model(a=3)
    assert m.b == 6
    assert m.model_dump() == {'a': 3, 'b': 6}


def test_private_attr_ellipsis() -> None:
    class Model(BaseModel):
        _a: Any = PrivateAttr(...)

    assert not hasattr(Model(), '_a')


def test_optional_typevar_any() -> None:
    Foobar = TypeVar('Foobar')

    class MyModel(BaseModel):
        foo: Optional[Foobar]

    assert MyModel.model_json_schema() == {
        'properties': {'foo': {'title': 'Foo', 'anyOf': [{'type': 'null'}, {'title': 'Foo'}]}},
        'required': ['foo'],
        'title': 'MyModel',
        'type': 'object'
    }
    assert MyModel(foo=None).foo is None
    assert MyModel(foo='x').foo == 'x'
    assert MyModel(foo=123).foo == 123
