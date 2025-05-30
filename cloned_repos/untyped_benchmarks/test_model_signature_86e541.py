import sys
from collections.abc import Iterable
from inspect import Parameter, Signature, signature
from typing import Annotated, Any, Generic, Optional, TypeVar, Union
import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic._internal._typing_extra import is_annotated

def _equals(a, b):
    """
    compare strings with spaces removed
    """
    if isinstance(a, str) and isinstance(b, str):
        return a.replace(' ', '') == b.replace(' ', '')
    elif isinstance(a, Iterable) and isinstance(b, Iterable):
        return all((_equals(a_, b_) for a_, b_ in zip(a, b)))
    else:
        raise TypeError(f'arguments must be both strings or both lists, not {type(a)}, {type(b)}')

def test_model_signature():

    class Model(BaseModel):
        a = Field(title='A')
        b = Field(10)
        c = Field(default_factory=lambda: 1)
    sig = signature(Model)
    assert sig != signature(BaseModel)
    assert _equals(map(str, sig.parameters.values()), ('a: float', 'b: int = 10', 'c: int = <factory>'))
    assert _equals(str(sig), '(*, a: float, b: int = 10, c: int = <factory>) -> None')

def test_generic_model_signature():
    T = TypeVar('T')

    class Model(BaseModel, Generic[T]):
        pass
    sig = signature(Model[int])
    assert sig != signature(BaseModel)
    assert _equals(map(str, sig.parameters.values()), ('a: int',))
    assert _equals(str(sig), '(*, a: int) -> None')

def test_custom_init_signature():

    class MyModel(BaseModel):
        name = 'John Doe'
        f__ = Field(alias='foo')
        model_config = ConfigDict(extra='allow')

        def __init__(self, id=1, bar=2, *, baz, **data):
            super().__init__(id=id, **data)
            self.bar = bar
            self.baz = baz
    sig = signature(MyModel)
    assert _equals(map(str, sig.parameters.values()), ('id: int = 1', 'bar=2', 'baz: Any', "name: str = 'John Doe'", 'foo: str', '**data'))
    assert _equals(str(sig), "(id: int = 1, bar=2, *, baz: Any, name: str = 'John Doe', foo: str, **data) -> None")

def test_custom_init_signature_with_no_var_kw():

    class Model(BaseModel):
        b = 2

        def __init__(self, a, b):
            super().__init__(a=a, b=b, c=1)
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(a: float, b: int) -> None')

def test_invalid_identifiers_signature():
    model = create_model('Model', **{'123 invalid identifier!': (int, Field(123, alias='valid_identifier')), '!': (int, Field(0, alias='yeah'))})
    assert _equals(str(signature(model)), '(*, valid_identifier: int = 123, yeah: int = 0) -> None')
    model = create_model('Model', **{'123 invalid identifier!': (int, 123), '!': (int, Field(0, alias='yeah'))})
    assert _equals(str(signature(model)), '(*, yeah: int = 0, **extra_data: Any) -> None')

def test_use_field_name():

    class Foo(BaseModel):
        foo = Field(alias='this is invalid')
        model_config = ConfigDict(populate_by_name=True)
    assert _equals(str(signature(Foo)), '(*, foo: str) -> None')

def test_does_not_use_reserved_word():

    class Foo(BaseModel):
        from_ = Field(alias='from')
        model_config = ConfigDict(populate_by_name=True)
    assert _equals(str(signature(Foo)), '(*, from_: str) -> None')

def test_extra_allow_no_conflict():

    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(*, spam: str, **extra_data: Any) -> None')

def test_extra_allow_conflict():

    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(*, extra_data: str, **extra_data_: Any) -> None')

def test_extra_allow_conflict_twice():

    class Model(BaseModel):
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(*, extra_data: str, extra_data_: str, **extra_data__: Any) -> None')

def test_extra_allow_conflict_custom_signature():

    class Model(BaseModel):

        def __init__(self, extra_data=1, **foobar):
            super().__init__(extra_data=extra_data, **foobar)
        model_config = ConfigDict(extra='allow')
    assert _equals(str(signature(Model)), '(extra_data: int = 1, **foobar: Any) -> None')

def test_signature_is_class_only():

    class Model(BaseModel):
        foo = 123

        def __call__(self, a):
            pass
    assert _equals(str(signature(Model)), '(*, foo: int = 123) -> None')
    assert _equals(str(signature(Model())), '(a: int) -> bool')
    assert not hasattr(Model(), '__signature__')

def test_optional_field():

    class Model(BaseModel):
        foo = None
    assert signature(Model) == Signature([Parameter('foo', Parameter.KEYWORD_ONLY, default=None, annotation=Optional[int])], return_annotation=None)

@pytest.mark.skipif(sys.version_info < (3, 12), reason='repr different on older versions')
def test_annotated_field():
    from annotated_types import Gt

    class Model(BaseModel):
        foo = 1
    sig = signature(Model)
    assert str(sig) == '(*, foo: Annotated[int, Gt(gt=1)] = 1) -> None'
    assert is_annotated(sig.parameters['foo'].annotation)

@pytest.mark.skipif(sys.version_info < (3, 10), reason='repr different on older versions')
def test_annotated_optional_field():
    from annotated_types import Gt

    class Model(BaseModel):
        foo = None
    assert str(signature(Model)) == '(*, foo: Annotated[Optional[int], Gt(gt=1)] = None) -> None'