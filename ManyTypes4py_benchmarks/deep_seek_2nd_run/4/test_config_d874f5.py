import json
import re
import sys
from collections.abc import Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from decimal import Decimal
from inspect import signature
from typing import Any, NamedTuple, Optional, Union, TypeVar, Type, Dict, List, Tuple, Set, FrozenSet, Callable, cast
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
    with_config
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

T = TypeVar('T')

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
        ConfigDict()['missing_property']

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
            title = 'MyTitle'
            frozen = True

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
            name = 'John Doe'
            f__ = Field(alias='foo')
            model_config = ConfigDict(extra='allow')

            def __init__(self, id: int = 1, bar: int = 2, *, baz: Any, **data: Any) -> None:
                super().__init__(id=id, **data)
                self.bar = bar
                self.baz = baz
        sig = signature(MyModel)
        assert _equals(map(str, sig.parameters.values()), ('id: int = 1', 'bar=2', 'baz: Any', "name: str = 'John Doe'", 'foo: str', '**data'))
        assert _equals(str(sig), "(id: int = 1, bar=2, *, baz: Any, name: str = 'John Doe', foo: str, **data) -> None")

    def test_base_config_custom_init_signature_with_no_var_kw(self) -> None:

        class Model(BaseModel):
            b = 2

            def __init__(self, a: float, b: int) -> None:
                super().__init__(a=a, b=b, c=1)
            model_config = ConfigDict(extra='allow')
        assert _equals(str(signature(Model)), '(a: float, b: int) -> None')

    def test_base_config_use_field_name(self) -> None:

        class Foo(BaseModel):
            foo = Field(alias='this is invalid')
            model_config = ConfigDict(populate_by_name=True)
        assert _equals(str(signature(Foo)), '(*, foo: str) -> None')

    def test_base_config_does_not_use_reserved_word(self) -> None:

        class Foo(BaseModel):
            from_ = Field(alias='from')
            model_config = ConfigDict(populate_by_name=True)
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
            _foo = PrivateAttr('private_attribute')
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
        values = [Model(a='1', b=2, c=3, d=4), Model(a=1, b=2, c='3', d=4), Model(a=1, b=2, c=3, d='4'), Model(a=1, b='2', c=3, d=4), Model(a=1, b=2, c=3, d=4)]
        assert all((v.model_dump() == {'a': 1, 'b': 2, 'c': 3, 'd': 4} for v in values))

    def test_finite_float_config(self) -> None:

        class Model(BaseModel):
            model_config = ConfigDict(allow_inf_nan=False)
        assert Model(a=42).a == 42
        with pytest.raises(ValidationError) as exc_info:
            Model(a=float('nan'))
        assert exc_info.value.errors(include_url=False) == [{'type': 'finite_number', 'loc': ('a',), 'msg': 'Input should be a finite number', 'input': HasRepr('nan')}]

    @pytest.mark.parametrize('enabled,str_check,result_str_check', [(True, '  123  ', '123'), (True, '  123\t\n', '123'), (False, '  123  ', '  123  ')])
    def test_str_strip_whitespace(self, enabled: bool, str_check: str, result_str_check: str) -> None:

        class Model(BaseModel):
            model_config = ConfigDict(str_strip_whitespace=enabled)
        m = Model(str_check=str_check)
        assert m.str_check == result_str_check

    @pytest.mark.parametrize('enabled,str_check,result_str_check', [(True, 'ABCDefG', 'ABCDEFG'), (False, 'ABCDefG', 'ABCDefG')])
    def test_str_to_upper(self, enabled: bool, str_check: str, result_str_check: str) -> None:

        class Model(BaseModel):
            model_config = ConfigDict(str_to_upper=enabled)
        m = Model(str_check=str_check)
        assert m.str_check == result_str_check

    @pytest.mark.parametrize('enabled,str_check,result_str_check', [(True, 'ABCDefG', 'abcdefg'), (False, 'ABCDefG', 'ABCDefG')])
    def test_str_to_lower(self, enabled: bool, str_check: str, result_str_check: str) -> None:

        class Model(BaseModel):
            model_config = ConfigDict(str_to_lower=enabled)
        m = Model(str_check=str_check)
        assert m.str_check == result_str_check

    def test_namedtuple_arbitrary_type(self) -> None:

        class CustomClass:
            pass

        class Tup(NamedTuple):
            pass

        class Model(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
        data = {'x': Tup(c=CustomClass())}
        model = Model.model_validate(data)
        assert isinstance(model.x.c, CustomClass)
        with pytest.raises(PydanticSchemaGenerationError):

            class ModelNoArbitraryTypes(BaseModel):
                pass

    @pytest.mark.parametrize('use_construct, populate_by_name_config, arg_name, expectation', [[False, True, 'bar', does_not_raise()], [False, True, 'bar_', does_not_raise()], [False, False, 'bar', does_not_raise()], [False, False, 'bar_', pytest.raises(ValueError)], [True, True, 'bar', does_not_raise()], [True, True, 'bar_', does_not_raise()], [True, False, 'bar', does_not_raise()], [True, False, 'bar_', does_not_raise()]])
    def test_populate_by_name_config(self, use_construct: bool, populate_by_name_config: bool, arg_name: str, expectation: AbstractContextManager) -> None:
        expected_value = 7

        class Foo(BaseModel):
            bar_ = Field(alias='bar')
            model_config = dict(populate_by_name=populate_by_name_config)
        with expectation:
            if use_construct:
                f = Foo.model_construct(**{arg_name: expected_value})
            else:
                f = Foo(**{arg_name: expected_value})
            assert f.bar_ == expected_value

    def test_immutable_copy_with_frozen(self) -> None:

        class Model(BaseModel):
            model_config = ConfigDict(frozen=True)
        m = Model(a=40, b=10)
        assert m == m.model_copy()

    def test_config_class_is_deprecated(self) -> None:
        with pytest.warns(PydanticDeprecatedSince20) as all_warnings:

            class Config(BaseConfig):
                pass
        assert len(all_warnings) in [1, 2]
        expected_warnings = ['Support for class-based `config` is deprecated, use ConfigDict instead']
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
        expected_warnings = {'Support for class-based `config` is deprecated, use ConfigDict instead', 'BaseConfig is deprecated. Use the `pydantic.ConfigDict` instead'}
        assert set((w.message.message for w in all_warnings)) <= expected_warnings

    @pytest.mark.filterwarnings('ignore:.* is deprecated.*:DeprecationWarning')
    def test_config_class_missing_attributes(self) -> None:
        with pytest.raises(AttributeError, match="type object 'BaseConfig' has no attribute 'missing_attribute'"):
            BaseConfig.missing_attribute
        with pytest.raises(AttributeError, match="'BaseConfig' object has no attribute 'missing_attribute'"):
            BaseConfig().missing_attribute

        class Config(BaseConfig):
            pass
        with pytest.raises(AttributeError, match="type object 'Config' has no attribute 'missing_attribute'"):
            Config.missing_attribute
        with pytest.raises(AttributeError, match="'Config' object has no attribute 'missing_attribute'"):
            Config().missing_attribute

def test_config_key_deprecation() -> None:
    config_dict = {'allow_mutation': None, 'error_msg_templates': None, 'fields': None, 'getter_dict': None, 'schema_extra': None, 'smart_union': None, 'underscore_attrs_are_private': None, 'allow_population_by_field_name': None, 'anystr_lower': None, 'anystr_strip_whitespace': None, 'anystr_upper': None, 'keep_untouched': None, 'max_anystr_length': None, 'min_anystr_length': None, 'orm_mode': None, 'validate_all': None}
    warning_message = "\nValid config keys have changed in V2:\n* 'allow_population_by_field_name' has been renamed to 'populate_by_name'\n* 'anystr_lower' has been renamed to 'str_to_lower'\n* 'anystr_strip_whitespace' has been renamed to 'str_strip_whitespace'\n* 'anystr_upper' has been renamed to 'str_to_upper'\n* 'keep_untouched' has been renamed to 'ignored_types'\n* 'max_anystr_length' has been renamed to 'str_max_length'\n* 'min_anystr_length' has been renamed to 'str_min_length'\n* 'orm_mode' has been renamed to 'from_attributes'\n* 'schema_extra' has been renamed to 'json_schema_extra'\n* 'validate_all' has been renamed to 'validate_default'\n* 'allow_mutation' has been removed\n* 'error_msg_templates' has been removed\n* 'fields' has been removed\n* 'getter_dict' has been removed\n* 'smart_union' has been removed\n* 'underscore_attrs_are_private' has been removed\n    ".strip()
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
    extra_error = re.escape('Invalid extra_behavior: `invalid-value`')
    config_dict = {'extra': 'invalid-value'}
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
    assert BaseModel.model