from inspect import signature
from typing import Optional, get_type_hints, Any, Dict, List
from hypothesis import given, strategies as st
from tests.common.debug import find_any

def use_this_signature(self: Any, a: int, b: Optional[List[Any]] = None, *, x: float, y: str) -> None:
    pass

class Model:
    __annotations__: Dict[str, Any] = get_type_hints(use_this_signature)
    __signature__: Any = signature(use_this_signature)

    def __init__(self, **kwargs: Any) -> None:
        assert set(kwargs) == {'a', 'x', 'y'}
        assert isinstance(kwargs['a'], int)
        assert isinstance(kwargs['x'], float)
        assert isinstance(kwargs['y'], str)

@given(st.builds(Model))
def test_builds_uses_signature_attribute(val: Model) -> None:
    assert isinstance(val, Model)

class ModelForFromType(Model):

    def __init__(self, **kwargs: Any) -> None:
        assert set(kwargs) == {'a', 'b', 'x', 'y'}
        self.b: Optional[List[Any]] = kwargs['b']
        assert self.b is None or isinstance(self.b, list)

@given(st.from_type(ModelForFromType))
def test_from_type_uses_signature_attribute(val: ModelForFromType) -> None:
    assert isinstance(val, ModelForFromType)

def test_from_type_can_be_default_or_annotation() -> None:
    find_any(st.from_type(ModelForFromType), lambda m: m.b is None)
    find_any(st.from_type(ModelForFromType), lambda m: isinstance(m.b, list))

def use_annotations(self: Any, test_a: int, test_b: Optional[List[Any]] = None, *, test_x: float, test_y: List[str]) -> None:
    pass

def use_signature(self: Any, testA: int, testB: Optional[List[Any]] = None, *, testX: float, testY: List[str]) -> None:
    pass

class ModelWithAlias:
    __annotations__: Dict[str, Any] = get_type_hints(use_annotations)
    __signature__: Any = signature(use_signature)

    def __init__(self, **kwargs: Any) -> None:
        assert set(kwargs) == {'testA', 'testX', 'testY'}
        assert isinstance(kwargs['testA'], int)
        assert isinstance(kwargs['testX'], float)
        assert isinstance(kwargs['testY'], list)
        assert all((isinstance(elem, str) for elem in kwargs['testY'])

@given(st.builds(ModelWithAlias))
def test_build_using_different_signature_and_annotations(val: ModelWithAlias) -> None:
    assert isinstance(val, ModelWithAlias)

def use_bad_signature(self: Any, testA: int, *, testX: float) -> None:
    pass

class ModelWithBadAliasSignature:
    __annotations__: Dict[str, Any] = get_type_hints(use_annotations)
    __signature__: Any = signature(use_bad_signature)

    def __init__(self, **kwargs: Any) -> None:
        assert set(kwargs) == {'testX'}
        assert isinstance(kwargs['testX'], float)

@given(st.builds(ModelWithBadAliasSignature))
def test_build_with_non_types_in_signature(val: ModelWithBadAliasSignature) -> None:
    assert isinstance(val, ModelWithBadAliasSignature)

class UnconventionalSignature:

    def __init__(x: Any, self: bool = True) -> None:
        assert not isinstance(x, int)
        x.self = self

def test_build_in_from_type_with_self_named_something_else() -> None:
    find_any(st.from_type(UnconventionalSignature), lambda x: x.self is True)
    find_any(st.from_type(UnconventionalSignature), lambda x: x.self is False)
