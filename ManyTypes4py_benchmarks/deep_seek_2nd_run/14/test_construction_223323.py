import pickle
from typing import Any, Optional, Dict, List, Set, Tuple, Union
import pytest
from pydantic_core import PydanticUndefined, ValidationError
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, PrivateAttr, PydanticDeprecatedSince20

class Model(BaseModel):
    b: int = 10

def test_simple_construct() -> None:
    m: Model = Model.model_construct(a=3.14)
    assert m.a == 3.14
    assert m.b == 10
    assert m.model_fields_set == {'a'}
    assert m.model_dump() == {'a': 3.14, 'b': 10}

def test_construct_misuse() -> None:
    m: Model = Model.model_construct(b='foobar')
    assert m.b == 'foobar'
    with pytest.warns(UserWarning, match='Expected `int` but got `str`'):
        assert m.model_dump() == {'b': 'foobar'}
    with pytest.raises(AttributeError, match="'Model' object has no attribute 'a'"):
        print(m.a)

def test_construct_fields_set() -> None:
    m: Model = Model.model_construct(a=3.0, b=-1, _fields_set={'a'})
    assert m.a == 3
    assert m.b == -1
    assert m.model_fields_set == {'a'}
    assert m.model_dump() == {'a': 3, 'b': -1}

def test_construct_allow_extra() -> None:
    """model_construct() should allow extra fields only in the case of extra='allow'"""

    class Foo(BaseModel, extra='allow'):
        pass
    model: Foo = Foo.model_construct(x=1, y=2)
    assert model.x == 1
    assert model.y == 2

@pytest.mark.parametrize('extra', ['ignore', 'forbid'])
def test_construct_ignore_extra(extra: str) -> None:
    """model_construct() should ignore extra fields only in the case of extra='ignore' or extra='forbid'"""

    class Foo(BaseModel, extra=extra):
        pass
    model: Foo = Foo.model_construct(x=1, y=2)
    assert model.x == 1
    assert model.__pydantic_extra__ is None
    assert 'y' not in model.__dict__

def test_construct_keep_order() -> None:

    class Foo(BaseModel):
        b: int = 42
    instance: Foo = Foo(a=1, b=321, c=3.14)
    instance_construct: Foo = Foo.model_construct(**instance.model_dump())
    assert instance == instance_construct
    assert instance.model_dump() == instance_construct.model_dump()
    assert instance.model_dump_json() == instance_construct.model_dump_json()

def test_construct_with_aliases() -> None:

    class MyModel(BaseModel):
        x: int = Field(alias='x_alias')
    my_model: MyModel = MyModel.model_construct(x_alias=1)
    assert my_model.x == 1
    assert my_model.model_fields_set == {'x'}
    assert my_model.model_dump() == {'x': 1}

def test_construct_with_validation_aliases() -> None:

    class MyModel(BaseModel):
        x: int = Field(validation_alias='x_alias')
    my_model: MyModel = MyModel.model_construct(x_alias=1)
    assert my_model.x == 1
    assert my_model.model_fields_set == {'x'}
    assert my_model.model_dump() == {'x': 1}

def test_large_any_str() -> None:

    class Model(BaseModel):
        pass
    content_bytes: bytes = b'x' * (2 ** 16 + 1)
    content_str: str = 'x' * (2 ** 16 + 1)
    m: Model = Model(a=content_bytes, b=content_str)
    assert m.a == content_bytes
    assert m.b == content_str

def deprecated_copy(m: BaseModel, *, include: Optional[Any] = None, exclude: Optional[Any] = None, update: Optional[Dict[str, Any]] = None, deep: bool = False) -> BaseModel:
    """
    This should only be used to make calls to the deprecated `copy` method with arguments
    that have been removed from `model_copy`. Otherwise, use the `copy_method` fixture below
    """
    with pytest.warns(PydanticDeprecatedSince20, match='The `copy` method is deprecated; use `model_copy` instead. See the docstring of `BaseModel.copy` for details about how to handle `include` and `exclude`.'):
        return m.copy(include=include, exclude=exclude, update=update, deep=deep)

@pytest.fixture(params=['copy', 'model_copy'])
def copy_method(request: pytest.FixtureRequest) -> Any:
    """
    Fixture to test both the old/deprecated `copy` and new `model_copy` methods.
    """
    if request.param == 'copy':
        return deprecated_copy
    else:

        def new_copy_method(m: BaseModel, *, update: Optional[Dict[str, Any]] = None, deep: bool = False) -> BaseModel:
            return m.model_copy(update=update, deep=deep)
        return new_copy_method

def test_simple_copy(copy_method: Any) -> None:
    m: Model = Model(a=24)
    m2: Model = copy_method(m)
    assert m.a == m2.a == 24
    assert m.b == m2.b == 10
    assert m == m2

@pytest.fixture(scope='session', name='ModelTwo')
def model_two_fixture() -> Any:

    class ModelTwo(BaseModel):
        _foo_: Dict[str, str] = PrivateAttr({'private'})
        b: int = 10
        c: str = 'foobar'
    return ModelTwo

def test_deep_copy(ModelTwo: Any, copy_method: Any) -> None:
    m: Any = ModelTwo(a=24, d=Model(a='12'))
    m._foo_ = {'new value'}
    m2: Any = copy_method(m, deep=True)
    assert m.a == m2.a == 24
    assert m.b == m2.b == 10
    assert m.c == m2.c == 'foobar'
    assert m.d is not m2.d
    assert m == m2
    assert m._foo_ == m2._foo_
    assert m._foo_ is not m2._foo_

def test_copy_exclude(ModelTwo: Any) -> None:
    m: Any = ModelTwo(a=24, d=Model(a='12'))
    m2: Any = deprecated_copy(m, exclude={'b'})
    assert m.a == m2.a == 24
    assert isinstance(m2.d, Model)
    assert m2.d.a == 12
    assert hasattr(m2, 'c')
    assert not hasattr(m2, 'b')
    assert set(m.model_dump().keys()) == {'a', 'b', 'c', 'd'}
    assert set(m2.model_dump().keys()) == {'a', 'c', 'd'}
    assert m != m2

def test_copy_include(ModelTwo: Any) -> None:
    m: Any = ModelTwo(a=24, d=Model(a='12'))
    m2: Any = deprecated_copy(m, include={'a'})
    assert m.a == m2.a == 24
    assert set(m.model_dump().keys()) == {'a', 'b', 'c', 'd'}
    assert set(m2.model_dump().keys()) == {'a'}
    assert m != m2

def test_copy_include_exclude(ModelTwo: Any) -> None:
    m: Any = ModelTwo(a=24, d=Model(a='12'))
    m2: Any = deprecated_copy(m, include={'a', 'b', 'c'}, exclude={'c'})
    assert set(m.model_dump().keys()) == {'a', 'b', 'c', 'd'}
    assert set(m2.model_dump().keys()) == {'a', 'b'}

def test_copy_advanced_exclude() -> None:

    class SubSubModel(BaseModel):
        a: str
        b: str

    class SubModel(BaseModel):
        c: str
        d: List[SubSubModel]

    class Model(BaseModel):
        e: str
        f: SubModel
    m: Model = Model(e='e', f=SubModel(c='foo', d=[SubSubModel(a='a', b='b'), SubSubModel(a='c', b='e')]))
    m2: Model = deprecated_copy(m, exclude={'f': {'c': ..., 'd': {-1: {'a'}}}})
    assert hasattr(m.f, 'c')
    assert not hasattr(m2.f, 'c')
    assert m2.model_dump() == {'e': 'e', 'f': {'d': [{'a': 'a', 'b': 'b'}, {'b': 'e'}]}}
    m2 = deprecated_copy(m, exclude={'e': ..., 'f': {'d'}})
    assert m2.model_dump() == {'f': {'c': 'foo'}}

def test_copy_advanced_include() -> None:

    class SubSubModel(BaseModel):
        a: str
        b: str

    class SubModel(BaseModel):
        c: str
        d: List[SubSubModel]

    class Model(BaseModel):
        e: str
        f: SubModel
    m: Model = Model(e='e', f=SubModel(c='foo', d=[SubSubModel(a='a', b='b'), SubSubModel(a='c', b='e')]))
    m2: Model = deprecated_copy(m, include={'f': {'c'}})
    assert hasattr(m.f, 'c')
    assert hasattr(m2.f, 'c')
    assert m2.model_dump() == {'f': {'c': 'foo'}}
    m2 = deprecated_copy(m, include={'e': ..., 'f': {'d': {-1}}})
    assert m2.model_dump() == {'e': 'e', 'f': {'d': [{'a': 'c', 'b': 'e'}]}}

def test_copy_advanced_include_exclude() -> None:

    class SubSubModel(BaseModel):
        a: str
        b: str

    class SubModel(BaseModel):
        c: str
        d: List[SubSubModel]

    class Model(BaseModel):
        e: str
        f: SubModel
    m: Model = Model(e='e', f=SubModel(c='foo', d=[SubSubModel(a='a', b='b'), SubSubModel(a='c', b='e')]))
    m2: Model = deprecated_copy(m, include={'e': ..., 'f': {'d'}}, exclude={'e': ..., 'f': {'d': {0}}})
    assert m2.model_dump() == {'f': {'d': [{'a': 'c', 'b': 'e'}]}}

def test_copy_update(ModelTwo: Any, copy_method: Any) -> None:
    m: Any = ModelTwo(a=24, d=Model(a='12'))
    m2: Any = copy_method(m, update={'a': 'different'})
    assert m.a == 24
    assert m2.a == 'different'
    m_keys: Set[str] = m.model_dump().keys()
    with pytest.warns(UserWarning, match='Expected `float` but got `str`'):
        m2_keys: Set[str] = m2.model_dump().keys()
    assert set(m_keys) == set(m2_keys) == {'a', 'b', 'c', 'd'}
    assert m != m2

def test_copy_update_unset(copy_method: Any) -> None:

    class Foo(BaseModel):
        foo: Optional[str] = None
        bar: Optional[str] = None
    assert copy_method(Foo(foo='hello'), update={'bar': 'world'}).model_dump_json(exclude_unset=True) == '{"foo":"hello","bar":"world"}'

class ExtraModel(BaseModel, extra='allow'):
    pass

def test_copy_deep_extra(copy_method: Any) -> None:

    class Foo(BaseModel, extra='allow'):
        pass
    m: Foo = Foo(extra=[])
    assert copy_method(m).extra is m.extra
    assert copy_method(m, deep=True).extra == m.extra
    assert copy_method(m, deep=True).extra is not m.extra

def test_copy_set_fields(ModelTwo: Any, copy_method: Any) -> None:
    m: Any = ModelTwo(a=24, d=Model(a='12'))
    m2: Any = copy_method(m)
    assert m.model_dump(exclude_unset=True) == {'a': 24.0, 'd': {'a': 12}}
    assert m.model_dump(exclude_unset=True) == m2.model_dump(exclude_unset=True)

def test_simple_pickle() -> None:
    m: Model = Model(a='24')
    b: bytes = pickle.dumps(m)
    m2: Model = pickle.loads(b)
    assert m.a == m2.a == 24
    assert m.b == m2.b == 10
    assert m == m2
    assert m is not m2
    assert tuple(m) == (('a', 24.0), ('b', 10))
    assert tuple(m2) == (('a', 24.0), ('b', 10))

def test_recursive_pickle(create_module: Any) -> None:

    @create_module
    def module() -> None:
        from pydantic import BaseModel, PrivateAttr

        class PickleModel(BaseModel):
            b: int = 10

        class PickleModelTwo(BaseModel):
            _foo_: Dict[str, str] = PrivateAttr({'private'})
            b: int = 10
            c: str = 'foobar'
    m: Any = module.PickleModelTwo(a=24, d=module.PickleModel(a='123.45'))
    m2: Any = pickle.loads(pickle.dumps(m))
    assert m == m2
    assert m.d.a == 123.45
    assert m2.d.a == 123.45
    assert m._foo_ == m2._foo_

def test_pickle_undefined(create_module: Any) -> None:

    @create_module
    def module() -> None:
        from pydantic import BaseModel, PrivateAttr

        class PickleModel(BaseModel):
            b: int = 10

        class PickleModelTwo(BaseModel):
            _foo_: Dict[str, str] = PrivateAttr({'private'})
            b: int = 10
            c: str = 'foobar'
    m: Any = module.PickleModelTwo(a=24, d=module.PickleModel(a='123.45'))
    m2: Any = pickle.loads(pickle.dumps(m))
    assert m2._foo_ == {'private'}
    m._foo_ = PydanticUndefined
    m3: Any = pickle.loads(pickle.dumps(m))
    assert not hasattr(m3, '_foo_')

def test_copy_undefined(ModelTwo: Any, copy_method: Any) -> None:
    m: Any = ModelTwo(a=24, d=Model(a='123.45'))
    m2: Any = copy_method(m)
    assert m2._foo_ == {'private'}
    m._foo_ = PydanticUndefined
    m3: Any = copy_method(m)
    assert not hasattr(m3, '_foo_')

def test_immutable_copy_with_frozen(copy_method: Any) -> None:

    class Model(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int
        b: int
    m: Model = Model(a=40, b=10)
    assert m == copy_method(m)
    assert repr(m) == 'Model(a=40, b=10)'
    m2: Model = copy_method(m, update={'b': 12})
    assert repr(m2) == 'Model(a=40, b=12)'
    with pytest.raises(ValidationError):
        m2.b = 13

def test_pickle_fields_set() -> None:
    m: Model = Model(a=24)
    assert m.model_dump(exclude_unset=True) == {'a': 24}
    m2: Model = pickle.loads(pickle.dumps(m))
    assert m2.model_dump(exclude_unset=True) == {'a': 24}

def test_pickle_preserves_extra() -> None:
    m: ExtraModel = ExtraModel(a=24)
    assert m.model_extra == {'a': 24}
    m2: ExtraModel = pickle.loads(pickle.dumps(m))
    assert m2.model_extra == {'a': 24}

def test_copy_update_exclude() -> None:

    class SubModel(BaseModel):
        a: str
        b: str

    class Model(BaseModel):
        c: str
        d: Dict[str, str]
    m: Model = Model(c='ex', d=dict(a='ax', b='bx'))
    assert m.model_dump() == {'c': 'ex', 'd': {'a': 'ax', 'b': 'bx'}}
    assert deprecated_copy(m, exclude={'c'}).model_dump() == {'d': {'a': 'ax', 'b': 'bx'}}
    with pytest.warns(UserWarning, match='Expected `str` but got `int`'):
        assert deprecated_copy(m, exclude={'c'}, update={'c': 42}).model_dump() == {'c': 42, 'd': {'a': 'ax', 'b': 'bx'}}
    with pytest.warns(PydanticDeprecatedSince20, match='The private method `_calculate_keys` will be removed and should no longer be used.'):
        assert m._calculate_keys(exclude={'x': ...}, include=None, exclude_unset=False) == {'c', 'd'}
        assert m._calculate_keys(exclude={'x': ...}, include=None, exclude_unset=False, update={'c': 42}) == {'d'}

def test_shallow_copy_modify(copy_method: Any) -> None:

    class X(BaseModel):
        val: int
        deep: Dict[str, List[int]]
    x