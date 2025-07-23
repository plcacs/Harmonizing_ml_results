import inspect
from inspect import Parameter, Signature
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, NewType, Optional, Set, Tuple, Type, TypeVar, Union
from unittest import skipIf
import pytest
from monkeytype.compat import cached_property, make_forward_ref
from monkeytype.stubs import AttributeStub, ClassStub, ExistingAnnotationStrategy, FunctionDefinition, FunctionStub, FunctionKind, ImportBlockStub, ImportMap, ModuleStub, ReplaceTypedDictsWithStubs, StubIndexBuilder, build_module_stubs, get_imports_for_annotation, get_imports_for_signature, render_annotation, render_signature, shrink_traced_types, update_signature_args, update_signature_return
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType, make_typed_dict
from mypy_extensions import TypedDict
from .util import Dummy

UserId = NewType('UserId', int)
T = TypeVar('T')

class TestImportMap:
    def test_merge(self) -> None:
        a = ImportMap()
        a['module.a'] = {'ClassA', 'ClassB'}
        a['module.b'] = {'ClassE', 'ClassF'}
        b = ImportMap()
        b['module.a'] = {'ClassB', 'ClassC'}
        b['module.c'] = {'ClassX', 'ClassY'}
        expected = ImportMap()
        for mod in ('module.a', 'module.b', 'module.c'):
            expected[mod] = a[mod] | b[mod]
        a.merge(b)
        assert a == expected

class TestImportBlockStub:
    def test_single_import(self) -> None:
        imports = ImportMap()
        imports['a.module'] = {'AClass'}
        imports['another.module'] = {'AnotherClass'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from a.module import AClass', 'from another.module import AnotherClass'])
        assert stub.render() == expected

    def test_io_import_single(self) -> None:
        imports = ImportMap()
        imports['_io'] = {'BytesIO'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from io import BytesIO'])
        assert stub.render() == expected

    def test_multiple_imports(self) -> None:
        imports = ImportMap()
        imports['a.module'] = {'AClass', 'AnotherClass', 'AThirdClass'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from a.module import (', '    AClass,', '    AThirdClass,', '    AnotherClass,', ')'])
        assert stub.render() == expected

    def test_multiple_io_imports(self) -> None:
        imports = ImportMap()
        imports['_io'] = {'BytesIO', 'FileIO'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from io import (', '    BytesIO,', '    FileIO,', ')'])
        assert stub.render() == expected

def simple_add(a: Any, b: Any) -> Any:
    return a + b

def strip_modules_helper(d1: Any, d2: Any) -> None:
    pass

def has_optional_param(x: Optional[int] = None) -> None:
    pass

def has_optional_union_param(x: Optional[Union[int, float]]) -> None:
    pass

def has_optional_return() -> Optional[int]:
    return None

def default_none_parameter(x: Optional[int] = None) -> None:
    pass

def has_length_exceeds_120_chars(very_long_name_parameter_1: float, very_long_name_parameter_2: float) -> Optional[float]:
    return None

def has_newtype_param(user_id: UserId) -> None:
    pass

def has_forward_ref() -> Optional['TestFunctionStub']:
    pass

def has_forward_ref_within_generator() -> Generator['TestFunctionStub', None, int]:
    pass

class TestAttributeStub:
    @pytest.mark.parametrize('stub, expected', [
        (AttributeStub(name='foo', typ=int), '    foo: int'),
        (AttributeStub(name='foo', typ=make_forward_ref('Foo')), "    foo: 'Foo'")
    ])
    def test_simple_attribute(self, stub: AttributeStub, expected: str) -> None:
        assert stub.render('    ') == expected

class TestRenderAnnotation:
    @pytest.mark.parametrize('annotation, expected', [
        (make_forward_ref('Foo'), "'Foo'"),
        (List[make_forward_ref('Foo')], "List['Foo']"),
        (List[List[make_forward_ref('Foo')]], "List[List['Foo']]"),
        (Optional[int], 'Optional[int]'),
        (List[Optional[int]], 'List[Optional[int]]'),
        (UserId, 'UserId'),
        (List[UserId], 'List[UserId]'),
        (List[int], 'List[int]'),
        (List[List[int]], 'List[List[int]]'),
        (None, 'None'),
        (List[None], 'List[None]'),
        (int, 'int'),
        (Dummy, 'tests.util.Dummy'),
        (List[Dummy], 'List[tests.util.Dummy]'),
        ('some_string', 'some_string'),
        (Iterable[None], 'Iterable[None]'),
        (List[Iterable[None]], 'List[Iterable[None]]'),
        (Generator[make_forward_ref('Foo'), None, None], "Generator['Foo', None, None]"),
        (List[Generator[make_forward_ref('Foo'), None, None]], "List[Generator['Foo', None, None]]"),
        (T, 'T'),
        (Dict[str, T], 'Dict[str, T]'),
        (Tuple[()], 'Tuple[()]')
    ])
    def test_render_annotation(self, annotation: Any, expected: str) -> None:
        assert render_annotation(annotation) == expected

class TestFunctionStub:
    def test_classmethod(self) -> None:
        stub = FunctionStub('test', inspect.signature(Dummy.a_class_method), FunctionKind.CLASS)
        expected = '\n'.join(['@classmethod', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    def test_staticmethod(self) -> None:
        stub = FunctionStub('test', inspect.signature(Dummy.a_static_method), FunctionKind.STATIC)
        expected = '\n'.join(['@staticmethod', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    def test_property(self) -> None:
        stub = FunctionStub('test', inspect.signature(Dummy.a_property.fget), FunctionKind.PROPERTY)
        expected = '\n'.join(['@property', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    @skipIf(cached_property is None, 'install Django to run this test')
    def test_cached_property(self) -> None:
        stub = FunctionStub('test', inspect.signature(Dummy.a_cached_property.func), FunctionKind.DJANGO_CACHED_PROPERTY)
        expected = '\n'.join(['@cached_property', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    def test_simple(self) -> None:
        for kind in [FunctionKind.MODULE, FunctionKind.INSTANCE]:
            stub = FunctionStub('test', inspect.signature(simple_add), kind)
            expected = 'def test%s: ...' % (render_signature(stub.signature),)
            assert stub.render() == expected

    def test_with_prefix(self) -> None:
        stub = FunctionStub('test', inspect.signature(simple_add), FunctionKind.MODULE)
        expected = '  def test%s: ...' % (render_signature(stub.signature),)
        assert stub.render(prefix='  ') == expected

    def test_strip_modules(self) -> None:
        f = strip_modules_helper
        stub = FunctionStub(f.__name__, inspect.signature(f), FunctionKind.MODULE, [Dummy.__module__])
        expected = 'def strip_modules_helper(d1: Dummy, d2: Dummy) -> None: ...'
        assert stub.render() == expected

    def test_async_function(self) -> None:
        stub = FunctionStub('test', inspect.signature(simple_add), FunctionKind.MODULE, is_async=True)
        expected = 'async def test%s: ...' % (render_signature(stub.signature),)
        assert stub.render() == expected

    def test_optional_parameter_annotation(self) -> None:
        stub = FunctionStub('test', inspect.signature(has_optional_param), FunctionKind.MODULE)
        expected = 'def test(x: Optional[int] = ...) -> None: ...'
        assert stub.render() == expected

    def test_optional_union_parameter_annotation(self) -> None:
        stub = FunctionStub('test', inspect.signature(has_optional_union_param), FunctionKind.MODULE)
        expected = 'def test(x: Optional[Union[int, float]]) -> None: ...'
        assert stub.render() == expected

    def test_optional_return_annotation(self) -> None:
        stub = FunctionStub('test', inspect.signature(has_optional_return), FunctionKind.MODULE)
        expected = 'def test() -> Optional[int]: ...'
        assert stub.render() == expected

    def test_split_parameters_across_multiple_lines(self) -> None:
        stub = FunctionStub('has_length_exceeds_120_chars', inspect.signature(has_length_exceeds_120_chars), FunctionKind.MODULE)
        expected = dedent('        def has_length_exceeds_120_chars(\n            very_long_name_parameter_1: float,\n            very_long_name_parameter_2: float\n        ) -> Optional[float]: ...')
        assert stub.render() == expected
        expected = '\n'.join(['    def has_length_exceeds_120_chars(', '        very_long_name_parameter_1: float,', '        very_long_name_parameter_2: float', '    ) -> Optional[float]: ...'])
        assert stub.render(prefix='    ') == expected

    def test_default_none_parameter_annotation(self) -> None:
        stub = FunctionStub('test', inspect.signature(default_none_parameter), FunctionKind.MODULE)
        expected = 'def test(x: Optional[int] = ...) -> None: ...'
        assert stub.render() == expected

    def test_newtype_parameter_annotation(self) -> None:
        stub = FunctionStub('test', inspect.signature(has_newtype_param), FunctionKind.MODULE)
        expected = 'def test(user_id: UserId) -> None: ...'
        assert stub.render() == expected

    def test_nonetype_annotation(self) -> None:
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': Dict[str, NoneType]}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        stub = FunctionStub('test', sig, FunctionKind.MODULE)
        expected = 'def test(a: Dict[str, None], b) -> int: ...'
        assert stub.render() == expected

    def test_forward_ref_annotation(self) -> None:
        stub = FunctionStub('has_forward_ref', inspect.signature(has_forward_ref), FunctionKind.MODULE)
        expected = "def has_forward_ref() -> Optional['TestFunctionStub']: ..."
        assert stub.render() == expected

    @pytest.mark.xfail(reason='We get Generator[ForwardRef(), ...].')
    def test_forward_ref_annotation_within_generator(self) -> None:
        stub = FunctionStub('foo', inspect.signature(has_forward_ref_within_generator), FunctionKind.MODULE)
        expected = "def foo() -> Generator['TestFunctionStub', None, int]: ..."
        assert stub.render() == expected

def _func_stub_from_callable(func: Callable[..., Any], strip_modules: Optional[List[str]] = None) -> FunctionStub:
    kind = FunctionKind.from_callable(func)
    sig = Signature.from_callable(func)
    return FunctionStub(func.__name__, sig, kind, strip_modules)

class TestClassStub:
    def test_render(self) -> None:
        cm_stub = _func_stub_from_callable(Dummy.a_class_method.__func__)
        im_stub = _func_stub_from_callable(Dummy.an_instance_method)
        class_stub = ClassStub('Test', function_stubs=(cm_stub, im_stub), attribute_stubs=[
            AttributeStub('foo', int),
            AttributeStub('bar', str)
        ])
        expected = '\n'.join([
            'class Test:',
            '    bar: str',
            '    foo: int',
            '    @classmethod',
            '    def a_class_method(cls, foo: Any) -> Optional[frame]: ...',
            '    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...'
        ])
        assert class_stub.render() == expected

class TestReplaceTypedDictsWithStubs:
    SIMPLE_TYPED_DICT_STUB = ClassStub(
        name='FooBarTypedDict__RENAME_ME__(TypedDict)',
        function_stubs=[],
        attribute_stubs=[
            AttributeStub(name='a', typ=int),
            AttributeStub(name='b', typ=str)
        ]
    )
    SIMPLE_TYPED_DICT_STUB2 = ClassStub(
        name='FooBar2TypedDict__RENAME_ME__(TypedDict)',
        function_stubs=[],
        attribute_stubs=[
            AttributeStub(name='a', typ=int),
            AttributeStub(name='b', typ=str)
        ]
    )
    SIMPLE_NON_TOTAL_TYPED_DICT_STUB = ClassStub(
        name='FooBarTypedDict__RENAME_ME__(TypedDict, total=False)',
        function_stubs=[],
        attribute_stubs=[
            AttributeStub(name='a', typ=int),
            AttributeStub(name='b', typ=str)
        ]
    )
    SIMPLE_BASE_AND_SUBCLASS = [
        ClassStub(
            name='FooBarTypedDict__RENAME_ME__(TypedDict)',
            function_stubs=[],
            attribute_stubs=[
                AttributeStub(name='a', typ=int),
                AttributeStub(name='b', typ=str)
            ]
        ),
        ClassStub(
            name='FooBarTypedDict__RENAME_ME__NonTotal(FooBarTypedDict__RENAME_ME__, total=False)',
            function_stubs=[],
            attribute_stubs=[
                AttributeStub(name='c', typ=int)
            ]
        )
    ]

    @pytest.mark.parametrize('typ, expected', [
        (int, (int, [])),
        (List[int], (List[int], [])),
        (Set[int], (Set[int], [])),
        (Dict[str, int], (Dict[str, int], [])),
        (Tuple[str, int], (Tuple[str, int], [])),
        (List[List[Dict[str, int]]], (List[List[Dict[str, int]]], [])),
        (List[List[make_typed_dict(required_fields={'a': int, 'b': str})]], 
         (List[List[make_forward_ref('FooBarTypedDict__RENAME_ME__')]], [SIMPLE_TYPED_DICT_STUB])),
        (Dict[str, make_typed_dict(required_fields={'a': int, 'b': str})], 
         (Dict[str, make_forward_ref('FooBar2TypedDict__RENAME_ME__')], [SIMPLE_TYPED_DICT_STUB2])),
        (make_typed_dict(required_fields={'a': int, 'b': str}), 
         (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [SIMPLE_TYPED_DICT_STUB])),
        (make_typed_dict(optional_fields={'a': int, 'b': str}), 
         (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [SIMPLE_NON_TOTAL_TYPED_DICT_STUB])),
        (make_typed_dict(required_fields={'a': int, 'b': str}, optional_fields={'c': int}), 
         (make_forward_ref('FooBarTypedDict__RENAME_ME__NonTotal'), SIMPLE_BASE_AND_SUBCLASS)),
        (TypedDict('GenuineTypedDict', {'a': int, 'b': str}), 
         (TypedDict('GenuineTypedDict', {'a': int, 'b': str}), []),
    ])
    def test_replace_typed_dict_with_stubs(self, typ: Any, expected: Tuple[Any, List[ClassStub]]) -> None:
        rewritten_type, stubs = ReplaceTypedDictsWithStubs.rewrite_and_get_stubs(typ, class_name_hint='foo_bar')
        actual = (rewritten_type, stubs)
        assert actual == expected

typed_dict_import_map = ImportMap()
typed_dict_import_map['mypy_extensions'] = {'TypedDict'}
module_stub_for_method_with_typed_dict = {
    'tests.util': ModuleStub(
        function_stubs=(),
        class_stubs=[
            ClassStub(
                name='Dummy',
                function_stubs=[
                    FunctionStub(
                        name='an_instance_method',
                        signature=Signature(
                            parameters=[
                                Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Parameter.empty),
                                Parameter(name='foo', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=make_forward_ref('FooTypedDict__RENAME_ME__')),
                                Parameter(name='bar', kind=Parameter.POSITIONAL_