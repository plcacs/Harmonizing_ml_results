import json
import re
import sys
from collections.abc import Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from decimal import Decimal
from inspect import signature, Parameter
from typing import Any, Type, Union, List, Optional, Dict, Callable, get_type_hints

import pytest
from dirty_equals import HasRepr, IsPartialDict
from pydantic_core import SchemaError, SchemaSerializer, SchemaValidator
from pydantic import (
    BaseConfig,
    BaseModel,
    Field,
    PrivateAttr,
    PydanticDeprecatedSince20,
    PydanticSchemaGenerationError,
    ValidationError,
    create_model,
    field_validator,
    validate_call,
    with_config,
)
from pydantic._internal._config import ConfigWrapper, config_defaults
from pydantic._internal._generate_schema import GenerateSchema
from pydantic._internal._mock_val_ser import MockValSer
from pydantic._internal._typing_extra import get_type_hints
from pydantic.config import ConfigDict, JsonValue
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.dataclasses import rebuild_dataclass
from pydantic.errors import PydanticUserError
from pydantic.fields import ComputedFieldInfo, FieldInfo
from pydantic.type_adapter import TypeAdapter
from pydantic.warnings import PydanticDeprecatedSince210, PydanticDeprecationWarning

from .conftest import CallCounter


@pytest.fixture(scope='session', name='BaseConfigModelWithStrictConfig')
def model_with_strict_config() -> Type[BaseModel]:
    class ModelWithStrictConfig(BaseModel):
        model_config = ConfigDict(strict=True)
    return ModelWithStrictConfig


def _equals(a: Any, b: Any) -> bool:
    """
    Compare strings with spaces removed
    """
    if isinstance(a, str) and isinstance(b, str):
        return a.replace(' ', '') == b.replace(' ', '')
    elif isinstance(a, Iterable) and isinstance(b, Iterable):
        return all((_equals(a_, b_) for a_, b_ in zip(a, b)))
    else:
        raise TypeError(f'arguments must be both strings or both lists, not {type(a)}, {type(b)}')


def test_config_dict_missing_keys() -> None:
    assert ConfigDict().get('missing_property') is None
    with pytest.raises(KeyError, match="'missing_property'"):
        _ = ConfigDict()['missing_property']


class TestsBaseConfig:
    @pytest.mark.filterwarnings('ignore:.* is deprecated.*:DeprecationWarning')
    def test_base_config_equality_defaults_of_config_dict_class(self) -> None:
        for key, value in config_defaults.items():
            assert getattr(BaseConfig, key) == value

    def test_config_and_module_config_cannot_be_used_together(self) -> None:
        with pytest.raises(PydanticUserError):
            class MyModel(BaseModel):
                model_config = ConfigDict(title='MyTitle')

                class Config:
                    title = 'MyTitleConfig'

    @pytest.mark.filterwarnings('ignore:.* is deprecated.*:DeprecationWarning')
    def test_base_config_properly_converted_to_dict(self) -> None:

        class MyConfig(BaseConfig):
            title: str = 'MyTitle'
            frozen: bool = True

        class MyBaseModel(BaseModel):

            class Config(MyConfig):
                ...

        class MyModel(MyBaseModel):
            ...
        MyModel.model_config['title'] = 'MyTitle'
        MyModel.model_config['frozen'] = True
        assert 'str_to_lower' not in MyModel.model_config

    def test_base_config_custom_init_signature(self) -> None:
        class MyModel(BaseModel):
            name: str = 'John Doe'
            f__ = Field(alias='foo')
            model_config = ConfigDict(extra='allow')

            def __init__(self, id: int = 1, bar: int = 2, *, baz: Any, **data: Any) -> None:
                super().__init__(id=id, **data)
                self.bar = bar
                self.baz = baz

        sig = signature(MyModel)
        parameters: List[Parameter] = list(sig.parameters.values())
        expected_params: List[str] = [
            'id: int = 1',
            'bar=2',
            'baz: Any',
            "name: str = 'John Doe'",
            'foo: str',
            '**data',
        ]
        assert _equals(list(map(str, parameters)), expected_params)
        assert _equals(str(sig), "(id: int = 1, bar=2, *, baz: Any, name: str = 'John Doe', foo: str, **data) -> None")

    def test_base_config_custom_init_signature_with_no_var_kw(self) -> None:
        class Model(BaseModel):
            b: int = 2

            def __init__(self, a: Any, b: Any) -> None:
                super().__init__(a=a, b=b, c=1)

            model_config = ConfigDict(extra='allow')
        assert _equals(str(signature(Model)), '(a: float, b: int) -> None')

    def test_base_config_use_field_name(self) -> None:
        class Foo(BaseModel):
            foo: str
            model_config = ConfigDict(populate_by_name=True)
            foo.__field_info__ = Field(alias='this is invalid')  # type: ignore

        assert _equals(str(signature(Foo)), '(*, foo: str) -> None')

    def test_base_config_does_not_use_reserved_word(self) -> None:
        class Foo(BaseModel):
            from_: str
            model_config = ConfigDict(populate_by_name=True)
            Foo.__annotations__['from_'] = str  # type: ignore
            Foo.__fields__ = {}  # Ensure field setup via Field(alias=...) is not confused
            Foo.from_ = Field(alias='from')

        assert _equals(str(signature(Foo)), '(*, from_: str) -> None')

    def test_base_config_extra_allow_no_conflict(self) -> None:
        class Model(BaseModel):
            model_config = ConfigDict(extra='allow')
        assert _equals(str(signature(Model)), '(*, spam: str, **extra_data: Any) -> None')

    def test_base_config_extra_allow_conflict_twice(self) -> None:
        class Model(BaseModel):
            model_config = ConfigDict(extra='allow')
        assert _equals(str(signature(Model)), '(*, extra_data: str, extra_data_: str, **extra_data__: Any) -> None')

    def test_base_config_extra_allow_conflict_custom_signature(self) -> None:
        class Model(BaseModel):

            def __init__(self, extra_data: int = 1, **foobar: Any) -> None:
                super().__init__(extra_data=extra_data, **foobar)
            model_config = ConfigDict(extra='allow')
        assert _equals(str(signature(Model)), '(extra_data: int = 1, **foobar: Any) -> None')

    def test_base_config_private_attribute_intersection_with_extra_field(self) -> None:
        class Model(BaseModel):
            _foo: str = PrivateAttr('private_attribute')
            model_config = ConfigDict(extra='allow')
        assert set(Model.__private_attributes__) == {'_foo'}
        m = Model(_foo='field')
        assert m._foo == 'private_attribute'
        assert m.__dict__ == {}
        assert m.__pydantic_extra__ == {'_foo': 'field'}
        assert m.model_dump() == {'_foo': 'field'}
        m._foo = 'still_private'
        assert m._foo == 'still_private'
        assert m.__dict__ == {}
        assert m.__pydantic_extra__ == {'_foo': 'field'}
        assert m.model_dump() == {'_foo': 'field'}

    def test_base_config_parse_model_with_strict_config_disabled(self, BaseConfigModelWithStrictConfig: Type[BaseModel]) -> None:
        class Model(BaseConfigModelWithStrictConfig):
            model_config = ConfigDict(strict=False)
        values: List[BaseModel] = [
            Model(a='1', b=2, c=3, d=4),
            Model(a=1, b=2, c='3', d=4),
            Model(a=1, b=2, c=3, d='4'),
            Model(a=1, b='2', c=3, d=4),
            Model(a=1, b=2, c=3, d=4),
        ]
        assert all((v.model_dump() == {'a': 1, 'b': 2, 'c': 3, 'd': 4} for v in values))

    def test_finite_float_config(self) -> None:
        class Model(BaseModel):
            a: Any
            model_config = ConfigDict(allow_inf_nan=False)
        assert Model(a=42).a == 42
        with pytest.raises(ValidationError) as exc_info:
            Model(a=float('nan'))
        assert exc_info.value.errors(include_url=False) == [{
            'type': 'finite_number',
            'loc': ('a',),
            'msg': 'Input should be a finite number',
            'input': HasRepr('nan')
        }]

    @pytest.mark.parametrize('enabled,str_check,result_str_check', [
        (True, '  123  ', '123'),
        (True, '  123\t\n', '123'),
        (False, '  123  ', '  123  ')
    ])
    def test_str_strip_whitespace(self, enabled: bool, str_check: str, result_str_check: str) -> None:
        class Model(BaseModel):
            str_check: str
            model_config = ConfigDict(str_strip_whitespace=enabled)
        m = Model(str_check=str_check)
        assert m.str_check == result_str_check

    @pytest.mark.parametrize('enabled,str_check,result_str_check', [
        (True, 'ABCDefG', 'ABCDEFG'),
        (False, 'ABCDefG', 'ABCDefG')
    ])
    def test_str_to_upper(self, enabled: bool, str_check: str, result_str_check: str) -> None:
        class Model(BaseModel):
            str_check: str
            model_config = ConfigDict(str_to_upper=enabled)
        m = Model(str_check=str_check)
        assert m.str_check == result_str_check

    @pytest.mark.parametrize('enabled,str_check,result_str_check', [
        (True, 'ABCDefG', 'abcdefg'),
        (False, 'ABCDefG', 'ABCDefG')
    ])
    def test_str_to_lower(self, enabled: bool, str_check: str, result_str_check: str) -> None:
        class Model(BaseModel):
            str_check: str
            model_config = ConfigDict(str_to_lower=enabled)
        m = Model(str_check=str_check)
        assert m.str_check == result_str_check

    def test_namedtuple_arbitrary_type(self) -> None:
        class CustomClass:
            pass

        from typing import NamedTuple

        class Tup(NamedTuple):
            pass

        class Model(BaseModel):
            x: Any
            model_config = ConfigDict(arbitrary_types_allowed=True)
        data: Dict[str, Any] = {'x': Tup(c=CustomClass())}
        model = Model.model_validate(data)
        assert isinstance(model.x.c, CustomClass)
        with pytest.raises(PydanticSchemaGenerationError):
            class ModelNoArbitraryTypes(BaseModel):
                pass

    @pytest.mark.parametrize(
        'use_construct, populate_by_name_config, arg_name, expectation',
        [
            (False, True, 'bar', does_not_raise()),
            (False, True, 'bar_', does_not_raise()),
            (False, False, 'bar', does_not_raise()),
            (False, False, 'bar_', pytest.raises(ValueError)),
            (True, True, 'bar', does_not_raise()),
            (True, True, 'bar_', does_not_raise()),
            (True, False, 'bar', does_not_raise()),
            (True, False, 'bar_', does_not_raise()),
        ]
    )
    def test_populate_by_name_config(
        self,
        use_construct: bool,
        populate_by_name_config: bool,
        arg_name: str,
        expectation: AbstractContextManager,
    ) -> None:
        expected_value: int = 7

        class Foo(BaseModel):
            bar_: int
            model_config = ConfigDict(populate_by_name=populate_by_name_config)
            bar_.__field_info__ = Field(alias='bar')  # type: ignore

        with expectation:
            if use_construct:
                f = Foo.model_construct(**{arg_name: expected_value})
            else:
                f = Foo(**{arg_name: expected_value})
            assert f.bar_ == expected_value

    def test_immutable_copy_with_frozen(self) -> None:
        class Model(BaseModel):
            a: Any
            b: Any
            model_config = ConfigDict(frozen=True)
        m = Model(a=40, b=10)
        assert m == m.model_copy()

    def test_config_class_is_deprecated(self) -> None:
        with pytest.warns(PydanticDeprecatedSince20) as all_warnings:

            class Config(BaseConfig):
                pass
        assert len(all_warnings) in [1, 2]
        expected_warnings: List[str] = ['Support for class-based `config` is deprecated, use ConfigDict instead']
        if len(all_warnings) == 2:
            expected_warnings.insert(0, 'BaseConfig is deprecated. Use the `pydantic.ConfigDict` instead')
        assert [w.message.message for w in all_warnings] == expected_warnings

    def test_config_class_attributes_are_deprecated(self) -> None:
        with pytest.warns(PydanticDeprecatedSince20) as all_warnings:
            assert BaseConfig.validate_assignment is False
            assert BaseConfig().validate_assignment is False

            class Config(BaseConfig):
                pass
            assert Config.validate_assignment is False
            assert Config().validate_assignment is False
        assert len(all_warnings) == 7
        expected_warnings: set = {
            'Support for class-based `config` is deprecated, use ConfigDict instead',
            'BaseConfig is deprecated. Use the `pydantic.ConfigDict` instead'
        }
        assert set((w.message.message for w in all_warnings)) <= expected_warnings

    @pytest.mark.filterwarnings('ignore:.* is deprecated.*:DeprecationWarning')
    def test_config_class_missing_attributes(self) -> None:
        with pytest.raises(AttributeError, match="type object 'BaseConfig' has no attribute 'missing_attribute'"):
            _ = BaseConfig.missing_attribute
        with pytest.raises(AttributeError, match="'BaseConfig' object has no attribute 'missing_attribute'"):
            _ = BaseConfig().missing_attribute

        class Config(BaseConfig):
            pass
        with pytest.raises(AttributeError, match="type object 'Config' has no attribute 'missing_attribute'"):
            _ = Config.missing_attribute
        with pytest.raises(AttributeError, match="'Config' object has no attribute 'missing_attribute'"):
            _ = Config().missing_attribute


def test_config_key_deprecation() -> None:
    config_dict: Dict[str, Any] = {
        'allow_mutation': None,
        'error_msg_templates': None,
        'fields': None,
        'getter_dict': None,
        'schema_extra': None,
        'smart_union': None,
        'underscore_attrs_are_private': None,
        'allow_population_by_field_name': None,
        'anystr_lower': None,
        'anystr_strip_whitespace': None,
        'anystr_upper': None,
        'keep_untouched': None,
        'max_anystr_length': None,
        'min_anystr_length': None,
        'orm_mode': None,
        'validate_all': None,
    }
    warning_message: str = (
        "\nValid config keys have changed in V2:\n* 'allow_population_by_field_name' has been renamed to 'populate_by_name'\n* 'anystr_lower' has been renamed to 'str_to_lower'\n* 'anystr_strip_whitespace' has been renamed to 'str_strip_whitespace'\n* 'anystr_upper' has been renamed to 'str_to_upper'\n* 'keep_untouched' has been renamed to 'ignored_types'\n* 'max_anystr_length' has been renamed to 'str_max_length'\n* 'min_anystr_length' has been renamed to 'str_min_length'\n* 'orm_mode' has been renamed to 'from_attributes'\n* 'schema_extra' has been renamed to 'json_schema_extra'\n* 'validate_all' has been renamed to 'validate_default'\n* 'allow_mutation' has been removed\n* 'error_msg_templates' has been removed\n* 'fields' has been removed\n* 'getter_dict' has been removed\n* 'smart_union' has been removed\n* 'underscore_attrs_are_private' has been removed\n    "
    ).strip()
    with pytest.warns(UserWarning, match=re.escape(warning_message)):

        class MyModel(BaseModel):
            model_config = config_dict
    with pytest.warns(UserWarning, match=re.escape(warning_message)):
        create_model('MyCreatedModel', __config__=config_dict)
    with pytest.warns(UserWarning, match=re.escape(warning_message)):

        @pydantic_dataclass(config=config_dict)
        class MyDataclass:
            pass
    with pytest.warns(UserWarning, match=re.escape(warning_message)):

        @validate_call(config=config_dict)
        def my_function() -> None:
            pass


def test_invalid_extra() -> None:
    ConfigDict(extra='invalid-value')
    extra_error: str = re.escape('Invalid extra_behavior: `invalid-value`')
    config_dict: Dict[str, Any] = {'extra': 'invalid-value'}
    with pytest.raises(SchemaError, match=extra_error):

        class MyModel(BaseModel):
            model_config = config_dict
    with pytest.raises(SchemaError, match=extra_error):
        create_model('MyCreatedModel', __config__=config_dict)
    with pytest.raises(SchemaError, match=extra_error):
        @pydantic_dataclass(config=config_dict)
        class MyDataclass:
            pass


def test_invalid_config_keys() -> None:

    @validate_call(config={'alias_generator': lambda x: x})
    def my_function() -> None:
        pass


def test_multiple_inheritance_config() -> None:
    class Parent(BaseModel):
        model_config = ConfigDict(frozen=True, extra='forbid')

    class Mixin(BaseModel):
        model_config = ConfigDict(use_enum_values=True)

    class Child(Mixin, Parent):
        model_config = ConfigDict(populate_by_name=True)
    assert BaseModel.model_config.get('frozen') is None
    assert BaseModel.model_config.get('populate_by_name') is None
    assert BaseModel.model_config.get('extra') is None
    assert BaseModel.model_config.get('use_enum_values') is None
    assert Parent.model_config.get('frozen') is True
    assert Parent.model_config.get('populate_by_name') is None
    assert Parent.model_config.get('extra') == 'forbid'
    assert Parent.model_config.get('use_enum_values') is None
    assert Mixin.model_config.get('frozen') is None
    assert Mixin.model_config.get('populate_by_name') is None
    assert Mixin.model_config.get('extra') is None
    assert Mixin.model_config.get('use_enum_values') is True
    assert Child.model_config.get('frozen') is True
    assert Child.model_config.get('populate_by_name') is True
    assert Child.model_config.get('extra') == 'forbid'
    assert Child.model_config.get('use_enum_values') is True


def test_config_wrapper_match() -> None:
    localns: Dict[str, Any] = {
        '_GenerateSchema': GenerateSchema,
        'GenerateSchema': GenerateSchema,
        'JsonValue': JsonValue,
        'FieldInfo': FieldInfo,
        'ComputedFieldInfo': ComputedFieldInfo,
    }
    config_dict_annotations: List[tuple[str, str]] = [(k, str(v)) for k, v in get_type_hints(ConfigDict, localns=localns).items()]
    config_dict_annotations.sort()
    config_wrapper_annotations: List[tuple[str, str]] = [(k, str(v)) for k, v in get_type_hints(ConfigWrapper, localns=localns).items() if k != 'config_dict']
    config_wrapper_annotations.sort()
    assert config_dict_annotations == config_wrapper_annotations, 'ConfigDict and ConfigWrapper must have the same annotations (except ConfigWrapper.config_dict)'


@pytest.mark.skipif(sys.version_info < (3, 11), reason='requires backport pre 3.11, fully tested in pydantic core')
def test_config_validation_error_cause() -> None:
    class Foo(BaseModel):
        foo: Any

        @field_validator('foo')
        def check_foo(cls, v: Any) -> Any:
            assert v > 5, 'Must be greater than 5'
            return v

    with pytest.raises(ValidationError) as exc_info:
        Foo(foo=4)
    assert exc_info.value.__cause__ is None
    Foo.model_config = ConfigDict(validation_error_cause=True)
    Foo.model_rebuild(force=True)
    with pytest.raises(ValidationError) as exc_info:
        Foo(foo=4)
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ExceptionGroup)
    assert len(exc_info.value.__cause__.exceptions) == 1
    src_exc: Exception = exc_info.value.__cause__.exceptions[0]
    assert repr(src_exc) == "AssertionError('Must be greater than 5\\nassert 4 > 5')"
    assert len(src_exc.__notes__) == 1
    assert src_exc.__notes__[0] == '\nPydantic: cause of loc: foo'


def test_config_defaults_match() -> None:
    localns: Dict[str, Any] = {'_GenerateSchema': GenerateSchema, 'GenerateSchema': GenerateSchema, 'FieldInfo': FieldInfo, 'ComputedFieldInfo': ComputedFieldInfo}
    config_dict_keys: List[str] = sorted(list(get_type_hints(ConfigDict, localns=localns).keys()))
    config_defaults_keys: List[str] = sorted(list(config_defaults.keys()))
    assert config_dict_keys == config_defaults_keys, 'ConfigDict and config_defaults must have the same keys'


def test_config_is_not_inherited_in_model_fields() -> None:
    from pydantic import BaseModel, ConfigDict

    class Inner(BaseModel):
        pass

    class Outer(BaseModel):
        model_config = ConfigDict(str_to_lower=True)
        x: Any
        inner: Inner
    m = Outer.model_validate(dict(x=['Abc'], inner=dict(a='Def')))
    assert m.model_dump() == {'x': ['abc'], 'inner': {'a': 'Def'}}


@pytest.mark.parametrize('config,input_str', (
    ({}, 'type=string_type, input_value=123, input_type=int'),
    ({'hide_input_in_errors': False}, 'type=string_type, input_value=123, input_type=int'),
    ({'hide_input_in_errors': True}, 'type=string_type'),
))
def test_hide_input_in_errors(config: Dict[str, Any], input_str: str) -> None:
    class Model(BaseModel):
        x: Any
        model_config = ConfigDict(**config)
    with pytest.raises(ValidationError, match=re.escape(f'Input should be a valid string [{input_str}]')):
        Model(x=123)


@pytest.mark.parametrize('inf_nan_capable_type', [float, Decimal])
@pytest.mark.parametrize('inf_nan_value', ['Inf', 'NaN'])
def test_config_inf_nan_enabled(
    inf_nan_capable_type: Union[type[float], type[Decimal]],
    inf_nan_value: str
) -> None:
    class Model(BaseModel):
        value: Any
        model_config = ConfigDict(allow_inf_nan=True)
    assert Model(value=inf_nan_capable_type(inf_nan_value))


@pytest.mark.parametrize('inf_nan_capable_type', [float, Decimal])
@pytest.mark.parametrize('inf_nan_value', ['Inf', 'NaN'])
def test_config_inf_nan_disabled(
    inf_nan_capable_type: Union[type[float], type[Decimal]],
    inf_nan_value: str
) -> None:
    class Model(BaseModel):
        value: Any
        model_config = ConfigDict(allow_inf_nan=False)
    with pytest.raises(ValidationError) as e:
        Model(value=inf_nan_capable_type(inf_nan_value))
    assert e.value.errors(include_url=False)[0] == IsPartialDict({
        'loc': ('value',),
        'msg': 'Input should be a finite number',
        'type': 'finite_number'
    })


@pytest.mark.parametrize('config,expected', (
    (ConfigDict(), 'ConfigWrapper()'),
    (ConfigDict(title='test'), "ConfigWrapper(title='test')")
))
def test_config_wrapper_repr(config: ConfigDict, expected: str) -> None:
    assert repr(ConfigWrapper(config=config)) == expected


def test_config_wrapper_get_item() -> None:
    config_wrapper: ConfigWrapper = ConfigWrapper(config=ConfigDict(title='test'))
    assert config_wrapper.title == 'test'
    with pytest.raises(AttributeError, match="Config has no attribute 'test'"):
        _ = config_wrapper.test


def test_config_inheritance_with_annotations() -> None:
    class Parent(BaseModel):
        model_config = {'extra': 'allow'}

    class Child(Parent):
        model_config = {'str_to_lower': True}
    assert Child.model_config == {'extra': 'allow', 'str_to_lower': True}


def test_json_encoders_model() -> None:
    with pytest.warns(PydanticDeprecationWarning):
        class Model(BaseModel):
            value: Decimal
            x: int
            model_config = ConfigDict(json_encoders={
                Decimal: lambda x: str(x * 2),
                int: lambda x: str(x * 3)
            })
    m = Model(value=Decimal('1.1'), x=1)
    assert json.loads(Model(value=Decimal('1.1'), x=1).model_dump_json()) == {'value': '2.2', 'x': '3'}


@pytest.mark.filterwarnings('ignore::pydantic.warnings.PydanticDeprecationWarning')
def test_json_encoders_type_adapter() -> None:
    config: ConfigDict = ConfigDict(json_encoders={
        Decimal: lambda x: str(x * 2),
        int: lambda x: str(x * 3)
    })
    ta: TypeAdapter = TypeAdapter(int, config=config)
    assert json.loads(ta.dump_json(1)) == '3'
    ta = TypeAdapter(Decimal, config=config)
    assert json.loads(ta.dump_json(Decimal('1.1'))) == '2.2'
    ta = TypeAdapter(Union[Decimal, int], config=config)
    assert json.loads(ta.dump_json(Decimal('1.1'))) == '2.2'
    assert json.loads(ta.dump_json(1)) == '2'


@pytest.mark.parametrize('defer_build', [True, False])
def test_config_model_defer_build(defer_build: bool, generate_schema_calls: CallCounter) -> None:
    config: ConfigDict = ConfigDict(defer_build=defer_build)

    class MyModel(BaseModel):
        x: int
        model_config = config
    if defer_build:
        assert isinstance(MyModel.__pydantic_validator__, MockValSer)
        assert isinstance(MyModel.__pydantic_serializer__, MockValSer)
        assert generate_schema_calls.count == 0, 'Should respect defer_build'
    else:
        assert isinstance(MyModel.__pydantic_validator__, SchemaValidator)
        assert isinstance(MyModel.__pydantic_serializer__, SchemaSerializer)
        assert generate_schema_calls.count == 1, 'Should respect defer_build'
    m = MyModel(x=1)
    assert m.x == 1
    assert m.model_dump()['x'] == 1
    assert m.model_validate({'x': 2}).x == 2
    assert m.model_json_schema()['type'] == 'object'
    assert isinstance(MyModel.__pydantic_validator__, SchemaValidator)
    assert isinstance(MyModel.__pydantic_serializer__, SchemaSerializer)
    assert generate_schema_calls.count == 1, 'Should not build duplicated core schemas'


@pytest.mark.parametrize('defer_build', [True, False])
def test_config_dataclass_defer_build(defer_build: bool, generate_schema_calls: CallCounter) -> None:
    config: ConfigDict = ConfigDict(defer_build=defer_build)

    @pydantic_dataclass(config=config)
    class MyDataclass:
        x: int
    if defer_build:
        assert isinstance(MyDataclass.__pydantic_validator__, MockValSer)
        assert isinstance(MyDataclass.__pydantic_serializer__, MockValSer)
        assert generate_schema_calls.count == 0, 'Should respect defer_build'
    else:
        assert isinstance(MyDataclass.__pydantic_validator__, SchemaValidator)
        assert isinstance(MyDataclass.__pydantic_serializer__, SchemaSerializer)
        assert generate_schema_calls.count == 1, 'Should respect defer_build'
    m = MyDataclass(x=1)
    assert m.x == 1
    assert isinstance(MyDataclass.__pydantic_validator__, SchemaValidator)
    assert isinstance(MyDataclass.__pydantic_serializer__, SchemaSerializer)
    assert generate_schema_calls.count == 1, 'Should not build duplicated core schemas'


def test_dataclass_defer_build_override_on_rebuild_dataclass(generate_schema_calls: CallCounter) -> None:
    config: ConfigDict = ConfigDict(defer_build=True)

    @pydantic_dataclass(config=config)
    class MyDataclass:
        x: int
    assert isinstance(MyDataclass.__pydantic_validator__, MockValSer)
    assert isinstance(MyDataclass.__pydantic_serializer__, MockValSer)
    assert generate_schema_calls.count == 0, 'Should respect defer_build'
    rebuild_dataclass(MyDataclass, force=True)
    assert isinstance(MyDataclass.__pydantic_validator__, SchemaValidator)
    assert isinstance(MyDataclass.__pydantic_serializer__, SchemaSerializer)
    assert generate_schema_calls.count == 1, 'Should have called generate_schema once'


@pytest.mark.parametrize('defer_build', [True, False])
def test_config_model_type_adapter_defer_build(defer_build: bool, generate_schema_calls: CallCounter) -> None:
    config: ConfigDict = ConfigDict(defer_build=defer_build)

    class MyModel(BaseModel):
        x: int
        model_config = config
    assert generate_schema_calls.count == (0 if defer_build is True else 1)
    generate_schema_calls.reset()
    ta: TypeAdapter = TypeAdapter(MyModel)
    assert generate_schema_calls.count == 0, 'Should use model generated schema'
    assert ta.validate_python({'x': 1}).x == 1
    assert ta.validate_python({'x': 2}).x == 2
    assert ta.dump_python(MyModel.model_construct(x=1))['x'] == 1
    assert ta.json_schema()['type'] == 'object'
    expected_count: int = 1 if defer_build is True else 0
    assert generate_schema_calls.count == expected_count, 'Should not build duplicate core schemas'


@pytest.mark.parametrize('defer_build', [True, False])
def test_config_plain_type_adapter_defer_build(defer_build: bool, generate_schema_calls: CallCounter) -> None:
    config: ConfigDict = ConfigDict(defer_build=defer_build)
    ta: TypeAdapter = TypeAdapter(dict[str, int], config=config)
    assert generate_schema_calls.count == (0 if defer_build else 1)
    generate_schema_calls.reset()
    assert ta.validate_python({}) == {}
    assert ta.validate_python({'x': 1}) == {'x': 1}
    assert ta.dump_python({'x': 2}) == {'x': 2}
    assert ta.json_schema()['type'] == 'object'
    expected_count: int = 1 if defer_build else 0
    assert generate_schema_calls.count == expected_count, 'Should not build duplicate core schemas'


@pytest.mark.parametrize('defer_build', [True, False])
def test_config_model_defer_build_nested(defer_build: bool, generate_schema_calls: CallCounter) -> None:
    config: ConfigDict = ConfigDict(defer_build=defer_build)
    assert generate_schema_calls.count == 0

    class MyNestedModel(BaseModel):
        x: int
        model_config = config

    class MyModel(BaseModel):
        y: MyNestedModel
    assert isinstance(MyModel.__pydantic_validator__, SchemaValidator)
    assert isinstance(MyModel.__pydantic_serializer__, SchemaSerializer)
    expected_schema_count: int = 1 if defer_build is True else 2
    assert generate_schema_calls.count == expected_schema_count, 'Should respect defer_build'
    if defer_build:
        assert isinstance(MyNestedModel.__pydantic_validator__, MockValSer)
        assert isinstance(MyNestedModel.__pydantic_serializer__, MockValSer)
    else:
        assert isinstance(MyNestedModel.__pydantic_validator__, SchemaValidator)
        assert isinstance(MyNestedModel.__pydantic_serializer__, SchemaSerializer)
    m = MyModel(y={'x': 1})
    assert m.y.x == 1
    assert m.model_dump() == {'y': {'x': 1}}
    assert m.model_validate({'y': {'x': 1}}).y.x == 1
    assert m.model_json_schema()['type'] == 'object'
    if defer_build:
        assert isinstance(MyNestedModel.__pydantic_validator__, MockValSer)
        assert isinstance(MyNestedModel.__pydantic_serializer__, MockValSer)
    else:
        assert isinstance(MyNestedModel.__pydantic_validator__, SchemaValidator)
        assert isinstance(MyNestedModel.__pydantic_serializer__, SchemaSerializer)
    assert generate_schema_calls.count == expected_schema_count, 'Should not build duplicated core schemas'


def test_config_model_defer_build_ser_first() -> None:
    class M1(BaseModel, defer_build=True):
        pass

    class M2(BaseModel, defer_build=True):
        b: Any
    m = M2.model_validate({'b': {'a': 'foo'}})
    assert m.b.model_dump() == {'a': 'foo'}


def test_defer_build_json_schema() -> None:
    class M(BaseModel, defer_build=True):
        a: int
    # Assuming default field 'a' is required
    schema = M.model_json_schema()
    assert isinstance(schema, dict)
    assert schema.get('type') == 'object'


def test_partial_creation_with_defer_build() -> None:
    class M(BaseModel):
        a: Any
        b: Any

    def create_partial(model: Type[BaseModel], optionals: set[str]) -> Type[BaseModel]:
        override_fields: Dict[str, Any] = {}
        model.model_rebuild()
        for name, field in model.model_fields.items():
            if field.is_required() and name in optionals:
                override_fields[name] = (Optional[field.annotation], FieldInfo.merge_field_infos(field, default=None))
        return create_model(f'Partial{model.__name__}', __base__=model, **override_fields)
    partial: Type[BaseModel] = create_partial(M, {'a'})
    assert M.model_json_schema()['required'] == ['a', 'b']
    assert partial.model_json_schema()['required'] == ['b']


def test_model_config_as_model_field_raises() -> None:
    with pytest.raises(PydanticUserError) as exc_info:
        class MyModel(BaseModel):
            # Using "model_config" as a model field name is not allowed
            model_config: Any
    assert exc_info.value.code == 'model-config-invalid-field-name'


def test_dataclass_allows_model_config_as_model_field() -> None:
    config_title: str = 'from_config'
    field_title: str = 'from_field'

    @pydantic_dataclass(config={'title': config_title})
    class MyDataclass:
        model_config: Dict[str, Any]

    m = MyDataclass(model_config={'title': field_title})
    assert m.model_config['title'] == field_title
    assert m.__pydantic_config__['title'] == config_title


def test_with_config_disallowed_with_model() -> None:
    msg: str = 'Cannot use `with_config` on Model as it is a Pydantic model'
    with pytest.raises(PydanticUserError, match=msg):

        @with_config({'coerce_numbers_to_str': True})
        class Model(BaseModel):
            pass


def test_empty_config_with_annotations() -> None:
    class Model(BaseModel):
        model_config = {}
    assert Model.model_config == {}


def test_generate_schema_deprecation_warning() -> None:
    with pytest.warns(PydanticDeprecatedSince210, match='The `schema_generator` setting has been deprecated since v2.10.'):
        class Model(BaseModel):
            model_config = ConfigDict(schema_generator=GenerateSchema)
