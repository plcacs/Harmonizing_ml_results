import sys
from collections.abc import Iterable
from inspect import Parameter, Signature, signature
from typing import Annotated, Any, Generic, Optional, TypeVar, Union
import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic._internal._typing_extra import is_annotated

def _equals(a: Union[str, Iterable[Any]], b: Union[str, Iterable[Any]]) -> bool:
    """
    compare strings with spaces removed
    """
    if isinstance(a, str) and isinstance(b, str):
        return a.replace(' ', '') == b.replace(' ', '')
    elif isinstance(a, Iterable) and isinstance(b, Iterable):
        return all((_equals(a_, b_) for a_, b_ in zip(a, b)))
    else:
        raise TypeError(f'arguments must be both strings or both lists, not {type(a)}, {type(b)}')

def test_model_signature() -> None:
    class Model(BaseModel):
        a: float = Field(title='A')
        b: int = Field(10)
        c: int = Field(default_factory=lambda: 1)
    sig: Signature = signature(Model)
    assert sig != signature(BaseModel)
    assert _equals(map(str, sig.parameters.values()), ('a: float', 'b: int = 10', 'c: int = <factory>'))
    assert _equals(str(sig), '(*, a: float, b: int = 10, c: int = <factory>) -> None')

def test_generic_model_signature() -> None:
    T = TypeVar('T')
    class Model(BaseModel, Generic[T]):
        a: T
    sig: Signature = signature(Model[int])
    assert sig != signature(BaseModel)
    assert _equals(map(str, sig.parameters.values()), ('a: int',))
    assert _equals(str(sig), '(*, a: int) -> None')

def test_custom_init_signature() -> None:
    class MyModel(BaseModel):
        name: str = 'John Doe'
        f__: Any = Field(alias='foo')
        model_config = ConfigDict(extra='allow')

        def __init__(self, id: int = 1, bar: int = 2, *, baz: Any, **data: Any) -> None:
            super().__init__(id=id, **data)
            self.bar = bar
            self.baz = baz
    sig: Signature = signature(MyModel)
    assert _equals(
        map(str, sig.parameters.values()),
        ("id: int = 1", "bar=2", "baz: Any", "name: str = 'John Doe'", "foo: str", '**data')
    )
    assert _equals(
        str(sig),
        "(id: int = 1, bar=2, *, baz: Any, name: str = 'John Doe', foo: str, **data) -> None"
    )

def test_custom_init_signature_with_no_var_kw() -> None:
    class Model(BaseModel):
        b: int = 2

        def __init__(self, a: float, b: int) -> None:
            super().__init__(a=a, b=b, c=1)
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(a: float, b: int) -> None')

def test_invalid_identifiers_signature() -> None:
    model = create_model(
        'Model',
        **{
            '123 invalid identifier!': (int, Field(123, alias='valid_identifier')),
            '!': (int, Field(0, alias='yeah')),
        }
    )
    assert _equals(str(signature(model)), '(*, valid_identifier: int = 123, yeah: int = 0) -> None')
    model = create_model(
        'Model',
        **{
            '123 invalid identifier!': (int, 123),
            '!': (int, Field(0, alias='yeah')),
        }
    )
    assert _equals(str(signature(model)), '(*, yeah: int = 0, **extra_data: Any) -> None')

def test_use_field_name() -> None:
    class Foo(BaseModel):
        foo: str = Field(alias='this is invalid')
        model_config = ConfigDict(populate_by_name=True)
    assert _equals(str(signature(Foo)), '(*, foo: str) -> None')

def test_does_not_use_reserved_word() -> None:
    class Foo(BaseModel):
        from_: Any = Field(alias='from')
        model_config = ConfigDict(populate_by_name=True)
    assert _equals(str(signature(Foo)), '(*, from_: str) -> None')

def test_extra_allow_no_conflict() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(*, spam: str, **extra_data: Any) -> None')

def test_extra_allow_conflict() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(*, extra_data: str, **extra_data_: Any) -> None')

def test_extra_allow_conflict_twice() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(*, extra_data: str, extra_data_: str, **extra_data__: Any) -> None')

def test_extra_allow_conflict_custom_signature() -> None:
    class Model(BaseModel):
        def __init__(self, extra_data: int = 1, **foobar: Any) -> None:
            super().__init__(extra_data=extra_data, **foobar)
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(extra_data: int = 1, **foobar: Any) -> None')

def test_signature_is_class_only() -> None:
    class Model(BaseModel):
        foo: int = 123

        def __call__(self, a: int) -> bool:
            pass
    assert _equals(str(signature(Model)), '(*, foo: int = 123) -> None')
    assert _equals(str(signature(Model())), '(a: int) -> bool')
    assert not hasattr(Model(), '__signature__')

def test_optional_field() -> None:
    class Model(BaseModel):
        foo: Optional[int] = None
    sig: Signature = signature(Model)
    expected_param = Parameter('foo', Parameter.KEYWORD_ONLY, default=None, annotation=Optional[int])
    assert sig == Signature([expected_param], return_annotation=None)

@pytest.mark.skipif(sys.version_info < (3, 12), reason='repr different on older versions')
def test_annotated_field() -> None:
    from annotated_types import Gt
    class Model(BaseModel):
        foo: int = 1
    sig: Signature = signature(Model)
    assert str(sig) == '(*, foo: Annotated[int, Gt(gt=1)] = 1) -> None'
    assert is_annotated(sig.parameters['foo'].annotation)

@pytest.mark.skipif(sys.version_info < (3, 10), reason='repr different on older versions')
def test_annotated_optional_field() -> None:
    from annotated_types import Gt
    class Model(BaseModel):
        foo: Optional[int] = None
    assert str(signature(Model)) == '(*, foo: Annotated[Optional[int], Gt(gt=1)] = None) -> None'