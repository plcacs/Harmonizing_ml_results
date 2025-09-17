#!/usr/bin/env python3
import inspect
from inspect import Parameter, Signature
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, NewType, Optional, Set, Tuple, Type, Union
from unittest import skipIf
import pytest
from monkeytype.compat import cached_property, make_forward_ref
from monkeytype.stubs import (
    AttributeStub,
    ClassStub,
    ExistingAnnotationStrategy,
    FunctionDefinition,
    FunctionStub,
    FunctionKind,
    ImportBlockStub,
    ImportMap,
    ModuleStub,
    ReplaceTypedDictsWithStubs,
    StubIndexBuilder,
    build_module_stubs,
    get_imports_for_annotation,
    get_imports_for_signature,
    render_annotation,
    render_signature,
    shrink_traced_types,
    update_signature_args,
    update_signature_return,
)
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType, make_typed_dict
from mypy_extensions import TypedDict
from .util import Dummy

UserId = NewType('UserId', int)
T = TypeVar('T')

class TestImportMap:
    def test_merge(self) -> None:
        a: ImportMap = ImportMap()
        a['module.a'] = {'ClassA', 'ClassB'}
        a['module.b'] = {'ClassE', 'ClassF'}
        b: ImportMap = ImportMap()
        b['module.a'] = {'ClassB', 'ClassC'}
        b['module.c'] = {'ClassX', 'ClassY'}
        expected: ImportMap = ImportMap()
        for mod in ('module.a', 'module.b', 'module.c'):
            expected[mod] = a[mod] | b[mod]
        a.merge(b)
        assert a == expected

class TestImportBlockStub:
    def test_single_import(self) -> None:
        """Single imports should be on one line"""
        imports: ImportMap = ImportMap()
        imports['a.module'] = {'AClass'}
        imports['another.module'] = {'AnotherClass'}
        stub: ImportBlockStub = ImportBlockStub(imports)
        expected: str = '\n'.join([
            'from a.module import AClass',
            'from another.module import AnotherClass'
        ])
        assert stub.render() == expected

    def test_io_import_single(self) -> None:
        """Single _io imports should convert to io"""
        imports: ImportMap = ImportMap()
        imports['_io'] = {'BytesIO'}
        stub: ImportBlockStub = ImportBlockStub(imports)
        expected: str = '\n'.join([
            'from io import BytesIO'
        ])
        assert stub.render() == expected

    def test_multiple_imports(self) -> None:
        """Multiple imports from a single module should each be on their own line"""
        imports: ImportMap = ImportMap()
        imports['a.module'] = {'AClass', 'AnotherClass', 'AThirdClass'}
        stub: ImportBlockStub = ImportBlockStub(imports)
        expected: str = '\n'.join([
            'from a.module import (',
            '    AClass,',
            '    AThirdClass,',
            '    AnotherClass,',
            ')'
        ])
        assert stub.render() == expected

    def test_multiple_io_imports(self) -> None:
        """Multiple imports from single _io module should be convert to io import"""
        imports: ImportMap = ImportMap()
        imports['_io'] = {'BytesIO', 'FileIO'}
        stub: ImportBlockStub = ImportBlockStub(imports)
        expected: str = '\n'.join([
            'from io import (',
            '    BytesIO,',
            '    FileIO,',
            ')'
        ])
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
        stub: FunctionStub = FunctionStub('test', inspect.signature(Dummy.a_class_method), FunctionKind.CLASS)
        expected: str = '\n'.join([
            '@classmethod',
            'def test%s: ...' % (render_signature(stub.signature),)
        ])
        assert stub.render() == expected

    def test_staticmethod(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(Dummy.a_static_method), FunctionKind.STATIC)
        expected: str = '\n'.join([
            '@staticmethod',
            'def test%s: ...' % (render_signature(stub.signature),)
        ])
        assert stub.render() == expected

    def test_property(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(Dummy.a_property.fget), FunctionKind.PROPERTY)
        expected: str = '\n'.join([
            '@property',
            'def test%s: ...' % (render_signature(stub.signature),)
        ])
        assert stub.render() == expected

    @skipIf(cached_property is None, 'install Django to run this test')
    def test_cached_property(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(Dummy.a_cached_property.func), FunctionKind.DJANGO_CACHED_PROPERTY)
        expected: str = '\n'.join([
            '@cached_property',
            'def test%s: ...' % (render_signature(stub.signature),)
        ])
        assert stub.render() == expected

    def test_simple(self) -> None:
        for kind in [FunctionKind.MODULE, FunctionKind.INSTANCE]:
            stub: FunctionStub = FunctionStub('test', inspect.signature(simple_add), kind)
            expected: str = 'def test%s: ...' % (render_signature(stub.signature),)
            assert stub.render() == expected

    def test_with_prefix(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(simple_add), FunctionKind.MODULE)
        expected: str = '  def test%s: ...' % (render_signature(stub.signature),)
        assert stub.render(prefix='  ') == expected

    def test_strip_modules(self) -> None:
        """We should strip modules from annotations in the signature"""
        to_strip: List[str] = [Dummy.__module__]
        f: Callable[..., Any] = strip_modules_helper
        stub: FunctionStub = FunctionStub(f.__name__, inspect.signature(f), FunctionKind.MODULE, to_strip)
        expected: str = 'def strip_modules_helper(d1: Dummy, d2: Dummy) -> None: ...'
        assert stub.render() == expected

    def test_async_function(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(simple_add), FunctionKind.MODULE, is_async=True)
        expected: str = 'async def test%s: ...' % (render_signature(stub.signature),)
        assert stub.render() == expected

    def test_optional_parameter_annotation(self) -> None:
        """Optional should always be included in parameter annotations, even if the default value is None"""
        stub: FunctionStub = FunctionStub('test', inspect.signature(has_optional_param), FunctionKind.MODULE)
        expected: str = 'def test(x: Optional[int] = ...) -> None: ...'
        assert stub.render() == expected

    def test_optional_union_parameter_annotation(self) -> None:
        """Optional[Union[X, Y]] should always be rendered as such, not Union[X, Y, None]"""
        stub: FunctionStub = FunctionStub('test', inspect.signature(has_optional_union_param), FunctionKind.MODULE)
        expected: str = 'def test(x: Optional[Union[int, float]]) -> None: ...'
        assert stub.render() == expected

    def test_optional_return_annotation(self) -> None:
        """Optional should always be included in return annotations"""
        stub: FunctionStub = FunctionStub('test', inspect.signature(has_optional_return), FunctionKind.MODULE)
        expected: str = 'def test() -> Optional[int]: ...'
        assert stub.render() == expected

    def test_split_parameters_across_multiple_lines(self) -> None:
        """When single-line length exceeds 120 characters, parameters should be split into multiple lines."""
        stub: FunctionStub = FunctionStub('has_length_exceeds_120_chars', inspect.signature(has_length_exceeds_120_chars), FunctionKind.MODULE)
        expected: str = dedent('        def has_length_exceeds_120_chars(\n            very_long_name_parameter_1: float,\n            very_long_name_parameter_2: float\n        ) -> Optional[float]: ...')
        assert stub.render() == expected
        expected_with_prefix: str = '\n'.join([
            '    def has_length_exceeds_120_chars(',
            '        very_long_name_parameter_1: float,',
            '        very_long_name_parameter_2: float',
            '    ) -> Optional[float]: ...'
        ])
        assert stub.render(prefix='    ') == expected_with_prefix

    def test_default_none_parameter_annotation(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(default_none_parameter), FunctionKind.MODULE)
        expected: str = 'def test(x: Optional[int] = ...) -> None: ...'
        assert stub.render() == expected

    def test_newtype_parameter_annotation(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(has_newtype_param), FunctionKind.MODULE)
        expected: str = 'def test(user_id: UserId) -> None: ...'
        assert stub.render() == expected

    def test_nonetype_annotation(self) -> None:
        """NoneType should always be rendered as None"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': Dict[str, NoneType]}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        stub: FunctionStub = FunctionStub('test', sig, FunctionKind.MODULE)
        expected: str = 'def test(a: Dict[str, None], b) -> int: ...'
        assert stub.render() == expected

    def test_forward_ref_annotation(self) -> None:
        """Forward refs should be rendered as strings, not _ForwardRef(...)."""
        stub: FunctionStub = FunctionStub('has_forward_ref', inspect.signature(has_forward_ref), FunctionKind.MODULE)
        expected: str = "def has_forward_ref() -> Optional['TestFunctionStub']: ..."
        assert stub.render() == expected

    @pytest.mark.xfail(reason='We get Generator[ForwardRef(), ...].')
    def test_forward_ref_annotation_within_generator(self) -> None:
        stub: FunctionStub = FunctionStub('foo', inspect.signature(has_forward_ref_within_generator), FunctionKind.MODULE)
        expected: str = "def foo() -> Generator['TestFunctionStub', None, int]: ..."
        assert stub.render() == expected

def _func_stub_from_callable(func: Callable[..., Any], strip_modules: Optional[List[str]] = None) -> FunctionStub:
    kind: FunctionKind = FunctionKind.from_callable(func)
    sig: Signature = Signature.from_callable(func)
    return FunctionStub(func.__name__, sig, kind, strip_modules)

class TestClassStub:
    def test_render(self) -> None:
        cm_stub: FunctionStub = _func_stub_from_callable(Dummy.a_class_method.__func__)
        im_stub: FunctionStub = _func_stub_from_callable(Dummy.an_instance_method)
        class_stub: ClassStub = ClassStub(
            'Test',
            function_stubs=(cm_stub, im_stub),
            attribute_stubs=[AttributeStub('foo', int), AttributeStub('bar', str)]
        )
        expected: str = '\n'.join([
            'class Test:',
            '    bar: str',
            '    foo: int',
            '    @classmethod',
            '    def a_class_method(cls, foo: Any) -> Optional[frame]: ...',
            '    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...'
        ])
        assert class_stub.render() == expected

class TestReplaceTypedDictsWithStubs:
    SIMPLE_TYPED_DICT_STUB: ClassStub = ClassStub(
        name='FooBarTypedDict__RENAME_ME__(TypedDict)',
        function_stubs=[],
        attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)]
    )
    SIMPLE_TYPED_DICT_STUB2: ClassStub = ClassStub(
        name='FooBar2TypedDict__RENAME_ME__(TypedDict)',
        function_stubs=[],
        attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)]
    )
    SIMPLE_NON_TOTAL_TYPED_DICT_STUB: ClassStub = ClassStub(
        name='FooBarTypedDict__RENAME_ME__(TypedDict, total=False)',
        function_stubs=[],
        attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)]
    )
    SIMPLE_BASE_AND_SUBCLASS: List[ClassStub] = [
        ClassStub(
            name='FooBarTypedDict__RENAME_ME__(TypedDict)',
            function_stubs=[],
            attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)]
        ),
        ClassStub(
            name='FooBarTypedDict__RENAME_ME__NonTotal(FooBarTypedDict__RENAME_ME__, total=False)',
            function_stubs=[],
            attribute_stubs=[AttributeStub(name='c', typ=int)]
        )
    ]

    @pytest.mark.parametrize('typ, expected', [
        (int, (int, [])),
        (List[int], (List[int], [])),
        (Set[int], (Set[int], [])),
        (Dict[str, int], (Dict[str, int], [])),
        (Tuple[str, int], (Tuple[str, int], [])),
        (List[List[Dict[str, int]]], (List[List[Dict[str, int]]], [])),
        (List[List[Dict[str, int]]], (List[List[Dict[str, int]]], [])),
        (List[List[make_typed_dict(required_fields={'a': int, 'b': str})]],
         (List[List[make_forward_ref('FooBarTypedDict__RENAME_ME__')]], [TestReplaceTypedDictsWithStubs.SIMPLE_TYPED_DICT_STUB])),
        (Dict[str, make_typed_dict(required_fields={'a': int, 'b': str})],
         (Dict[str, make_forward_ref('FooBar2TypedDict__RENAME_ME__')], [TestReplaceTypedDictsWithStubs.SIMPLE_TYPED_DICT_STUB2])),
        (Set[make_typed_dict(required_fields={'a': int, 'b': str})],
         (Set[make_forward_ref('FooBarTypedDict__RENAME_ME__')], [TestReplaceTypedDictsWithStubs.SIMPLE_TYPED_DICT_STUB])),
        (Tuple[int, make_typed_dict(required_fields={'a': int, 'b': str})],
         (Tuple[int, make_forward_ref('FooBar2TypedDict__RENAME_ME__')], [TestReplaceTypedDictsWithStubs.SIMPLE_TYPED_DICT_STUB2])),
        (make_typed_dict(required_fields={'a': int, 'b': str}),
         (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [TestReplaceTypedDictsWithStubs.SIMPLE_TYPED_DICT_STUB])),
        (make_typed_dict(optional_fields={'a': int, 'b': str}),
         (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [TestReplaceTypedDictsWithStubs.SIMPLE_NON_TOTAL_TYPED_DICT_STUB])),
        (make_typed_dict(required_fields={'a': int, 'b': str}, optional_fields={'c': int}),
         (make_forward_ref('FooBarTypedDict__RENAME_ME__NonTotal'), TestReplaceTypedDictsWithStubs.SIMPLE_BASE_AND_SUBCLASS)),
        (TypedDict('GenuineTypedDict', {'a': int, 'b': str}),
         (TypedDict('GenuineTypedDict', {'a': int, 'b': str}), [])),
        (make_typed_dict(required_fields={'a': int, 'b': make_typed_dict(required_fields={'a': int, 'b': str})}),
         (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [
             ClassStub(name='BTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)]),
             ClassStub(name='FooBarTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=make_forward_ref('BTypedDict__RENAME_ME__'))])
         ])),
        (Tuple[make_typed_dict(required_fields={'a': int}), make_typed_dict(required_fields={'b': str})],
         (Tuple[make_forward_ref('FooBarTypedDict__RENAME_ME__'), make_forward_ref('FooBar2TypedDict__RENAME_ME__')], [
             ClassStub(name='FooBarTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int)]),
             ClassStub(name='FooBar2TypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='b', typ=str)])
         ]))
    ])
    def test_replace_typed_dict_with_stubs(self, typ: Any, expected: Any) -> None:
        rewritten_type, stubs = ReplaceTypedDictsWithStubs.rewrite_and_get_stubs(typ, class_name_hint='foo_bar')
        actual: Tuple[Any, Any] = (rewritten_type, stubs)
        assert actual == expected

typed_dict_import_map: ImportMap = ImportMap()
typed_dict_import_map['mypy_extensions'] = {'TypedDict'}
module_stub_for_method_with_typed_dict: Dict[str, ModuleStub] = {
    'tests.util': ModuleStub(
        function_stubs=(),
        class_stubs=[ClassStub(
            name='Dummy',
            function_stubs=[FunctionStub(
                name='an_instance_method',
                signature=Signature(
                    parameters=[
                        Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Parameter.empty),
                        Parameter(name='foo', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=make_forward_ref('FooTypedDict__RENAME_ME__')),
                        Parameter(name='bar', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
                    ],
                    return_annotation=make_forward_ref('DummyAnInstanceMethodTypedDict__RENAME_ME__')
                ),
                kind=FunctionKind.INSTANCE,
                strip_modules=['mypy_extensions'],
                is_async=False
            )]
        )],
        imports_stub=ImportBlockStub(typed_dict_import_map),
        typed_dict_class_stubs=[
            ClassStub(
                name='FooTypedDict__RENAME_ME__(TypedDict)',
                function_stubs=[],
                attribute_stubs=[AttributeStub('a', int), AttributeStub('b', str)]
            ),
            ClassStub(
                name='DummyAnInstanceMethodTypedDict__RENAME_ME__(TypedDict)',
                function_stubs=[],
                attribute_stubs=[AttributeStub('c', int)]
            )
        ]
    )
}

class TestModuleStub:
    def test_render(self) -> None:
        cm_stub: FunctionStub = _func_stub_from_callable(Dummy.a_class_method)
        im_stub: FunctionStub = _func_stub_from_callable(Dummy.an_instance_method)
        sig_stub: FunctionStub = _func_stub_from_callable(Dummy.has_complex_signature)
        func_stubs: Tuple[FunctionStub, ...] = (cm_stub, im_stub, sig_stub)
        test_stub: ClassStub = ClassStub('Test', function_stubs=func_stubs)
        test2_stub: ClassStub = ClassStub('Test2', function_stubs=func_stubs)
        other_class_stubs: Iterable[ClassStub] = module_stub_for_method_with_typed_dict['tests.util'].class_stubs.values()
        class_stubs: Tuple[ClassStub, ...] = (*other_class_stubs, test_stub, test2_stub)
        typed_dict_class_stubs: List[ClassStub] = module_stub_for_method_with_typed_dict['tests.util'].typed_dict_class_stubs
        mod_stub: ModuleStub = ModuleStub(function_stubs=func_stubs, class_stubs=class_stubs, typed_dict_class_stubs=typed_dict_class_stubs)
        expected: str = '\n'.join([
            'class DummyAnInstanceMethodTypedDict__RENAME_ME__(TypedDict):',
            '    c: int',
            '',
            '',
            'class FooTypedDict__RENAME_ME__(TypedDict):',
            '    a: int',
            '    b: str',
            '',
            '',
            '@classmethod',
            'def a_class_method(foo: Any) -> Optional[frame]: ...',
            '',
            '',
            'def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...',
            '',
            '',
            'def has_complex_signature(',
            '    self,',
            '    a: Any,',
            '    b: Any,',
            '    /,',
            '    c: Any,',
            '    d: Any = ...,',
            '    *e: Any,',
            '    f: Any,',
            '    g: Any = ...,',
            '    **h: Any',
            ') -> Optional[frame]: ...',
            '',
            '',
            'class Dummy:',
            '    def an_instance_method(',
            '        self,',
            "        foo: 'FooTypedDict__RENAME_ME__',",
            '        bar: int',
            "    ) -> 'DummyAnInstanceMethodTypedDict__RENAME_ME__': ...",
            '',
            '',
            'class Test:',
            '    @classmethod',
            '    def a_class_method(foo: Any) -> Optional[frame]: ...',
            '    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...',
            '    def has_complex_signature(',
            '        self,',
            '        a: Any,',
            '        b: Any,',
            '        /,',
            '        c: Any,',
            '        d: Any = ...,',
            '        *e: Any,',
            '        f: Any,',
            '        g: Any = ...,',
            '        **h: Any',
            '    ) -> Optional[frame]: ...',
            '',
            '',
            'class Test2:',
            '    @classmethod',
            '    def a_class_method(foo: Any) -> Optional[frame]: ...',
            '    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...',
            '    def has_complex_signature(',
            '        self,',
            '        a: Any,',
            '        b: Any,',
            '        /,',
            '        c: Any,',
            '        d: Any = ...,',
            '        *e: Any,',
            '        f: Any,',
            '        g: Any = ...,',
            '        **h: Any',
            '    ) -> Optional[frame]: ...'
        ])
        assert mod_stub.render() == expected

    def test_render_nested_typed_dict(self) -> None:
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            Dummy.an_instance_method,
            {'foo': make_typed_dict(required_fields={'z': make_typed_dict(required_fields={'a': int, 'b': str}), 'b': str}),
             'bar': int},
            int,
            None,
            existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        entries: List[FunctionDefinition] = [function]
        expected: str = '\n'.join([
            'from mypy_extensions import TypedDict',
            '',
            '',
            'class FooTypedDict__RENAME_ME__(TypedDict):',
            '    b: str',
            "    z: 'ZTypedDict__RENAME_ME__'",
            '',
            '',
            'class ZTypedDict__RENAME_ME__(TypedDict):',
            '    a: int',
            '    b: str',
            '',
            '',
            'class Dummy:',
            "    def an_instance_method(self, foo: 'FooTypedDict__RENAME_ME__', bar: int) -> int: ..."
        ])
        self.maxDiff = None  # type: ignore
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_return_typed_dict(self) -> None:
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            Dummy.an_instance_method,
            {'foo': int, 'bar': int},
            make_typed_dict(required_fields={'a': int, 'b': str}),
            None,
            existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        entries: List[FunctionDefinition] = [function]
        expected: str = '\n'.join([
            'from mypy_extensions import TypedDict',
            '',
            '',
            'class DummyAnInstanceMethodTypedDict__RENAME_ME__(TypedDict):',
            '    a: int',
            '    b: str',
            '',
            '',
            'class Dummy:',
            "    def an_instance_method(self, foo: int, bar: int) -> 'DummyAnInstanceMethodTypedDict__RENAME_ME__': ..."
        ])
        self.maxDiff = None  # type: ignore
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_yield_typed_dict(self) -> None:
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            Dummy.an_instance_method,
            {'foo': int, 'bar': int},
            int,
            yield_type=make_typed_dict(required_fields={'a': int, 'b': str}),
            existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        entries: List[FunctionDefinition] = [function]
        expected: str = '\n'.join([
            'from mypy_extensions import TypedDict',
            'from typing import Generator',
            '',
            '',
            'class DummyAnInstanceMethodYieldTypedDict__RENAME_ME__(TypedDict):',
            '    a: int',
            '    b: int',
            '',
            '',
            'class Dummy:',
            '    def an_instance_method(',
            '        self,',
            '        foo: int,',
            '        bar: int',
            "    ) -> Generator['DummyAnInstanceMethodYieldTypedDict__RENAME_ME__', None, int]: ..."
        ])
        self.maxDiff = None  # type: ignore
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_typed_dict_in_list(self) -> None:
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            Dummy.an_instance_method,
            {'foo': List[make_typed_dict(required_fields={'a': int})], 'bar': int},
            int,
            None,
            existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        entries: List[FunctionDefinition] = [function]
        expected: str = '\n'.join([
            'from mypy_extensions import TypedDict',
            'from typing import List',
            '',
            '',
            'class FooTypedDict__RENAME_ME__(TypedDict):',
            '    a: int',
            '',
            '',
            'class Dummy:',
            "    def an_instance_method(self, foo: List['FooTypedDict__RENAME_ME__'], bar: int) -> int: ..."
        ])
        self.maxDiff = None  # type: ignore
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_typed_dict_base_and_subclass(self) -> None:
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            Dummy.an_instance_method,
            {'foo': make_typed_dict(required_fields={'a': int}, optional_fields={'b': str}), 'bar': int},
            int,
            None,
            existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        entries: List[FunctionDefinition] = [function]
        expected: str = '\n'.join([
            'from mypy_extensions import TypedDict',
            '',
            '',
            'class FooTypedDict__RENAME_ME__(TypedDict):',
            '    a: int',
            '',
            '',
            'class FooTypedDict__RENAME_ME__NonTotal(FooTypedDict__RENAME_ME__, total=False):',
            '    b: str',
            '',
            '',
            'class Dummy:',
            "    def an_instance_method(self, foo: 'FooTypedDict__RENAME_ME__NonTotal', bar: int) -> int: ..."
        ])
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_return_empty_tuple(self) -> None:
        """Regression test for #190."""
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            Dummy.an_instance_method,
            {'foo': int, 'bar': int},
            Tuple[()],
            None,
            existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        entries: List[FunctionDefinition] = [function]
        expected: str = '\n'.join([
            'from typing import Tuple',
            '',
            '',
            'class Dummy:',
            '    def an_instance_method(self, foo: int, bar: int) -> Tuple[()]: ...'
        ])
        self.maxDiff = None  # type: ignore
        assert build_module_stubs(entries)['tests.util'].render() == expected

class TestBuildModuleStubs:
    def test_build_module_stubs(self) -> None:
        entries: List[FunctionDefinition] = [
            FunctionDefinition.from_callable(Dummy.a_static_method),
            FunctionDefinition.from_callable(Dummy.a_class_method.__func__),
            FunctionDefinition.from_callable(Dummy.an_instance_method),
            FunctionDefinition.from_callable(simple_add)
        ]
        simple_add_stub: FunctionStub = _func_stub_from_callable(simple_add)
        to_strip: List[str] = ['typing']
        dummy_stub: ClassStub = ClassStub(
            'Dummy',
            function_stubs=[
                _func_stub_from_callable(Dummy.a_class_method.__func__, to_strip),
                _func_stub_from_callable(Dummy.an_instance_method, to_strip),
                _func_stub_from_callable(Dummy.a_static_method, to_strip)
            ]
        )
        imports: Dict[str, Set[str]] = {'typing': {'Any', 'Optional'}}
        expected: Dict[str, ModuleStub] = {
            'tests.test_stubs': ModuleStub(function_stubs=[simple_add_stub]),
            'tests.util': ModuleStub(class_stubs=[dummy_stub], imports_stub=ImportBlockStub(imports))
        }
        self.maxDiff = None  # type: ignore
        assert build_module_stubs(entries) == expected

    def test_build_module_stubs_typed_dict_parameter(self) -> None:
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            Dummy.an_instance_method,
            {'foo': make_typed_dict(required_fields={'a': int, 'b': str}), 'bar': int},
            int,
            None,
            existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        entries: List[FunctionDefinition] = [function]
        expected: Dict[str, ModuleStub] = module_stub_for_method_with_typed_dict
        self.maxDiff = None  # type: ignore
        assert build_module_stubs(entries) == expected

def untyped_helper(x: Any, y: Any) -> None:
    pass

class TestStubIndexBuilder:
    def test_ignore_non_matching_functions(self) -> None:
        b: StubIndexBuilder = StubIndexBuilder('foo.bar', max_typed_dict_size=0)
        b.log(CallTrace(untyped_helper, {'x': int, 'y': str}))
        assert len(b.index) == 0

    def test_build_index(self) -> None:
        idxb: StubIndexBuilder = StubIndexBuilder('tests', max_typed_dict_size=0)
        idxb.log(CallTrace(untyped_helper, {'x': int, 'y': str}, str))
        sig: Signature = Signature.from_callable(untyped_helper)
        sig = sig.replace(parameters=[
            Parameter('x', Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter('y', Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
        ], return_annotation=str)
        mod_stub: ModuleStub = ModuleStub(function_stubs=[FunctionStub('untyped_helper', sig, FunctionKind.MODULE)])
        expected: Dict[str, ModuleStub] = {'tests.test_stubs': mod_stub}
        assert idxb.get_stubs() == expected

class UpdateSignatureHelper:
    @staticmethod
    def has_annos(a: Any, b: Any) -> int:
        return 0

    @classmethod
    def a_class_method(cls) -> None:
        pass

    def an_instance_method(self) -> None:
        pass

class TestUpdateSignatureArgs:
    def test_update_arg(self) -> None:
        """Update arg annotations from types"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'b': int}, False)
        params = [
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
        ]
        assert sig == Signature(parameters=params, return_annotation=int)

    def test_update_arg_with_anno(self) -> None:
        """Leave existing arg annotations alone"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': str}, False)
        expected_sig: Signature = Signature(parameters=[
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)
        ], return_annotation=int)
        assert sig == expected_sig

    def test_update_self(self) -> None:
        """Don't annotate first arg of instance methods"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.an_instance_method)
        sig = update_signature_args(sig, {'self': UpdateSignatureHelper}, True)
        expected_sig: Signature = Signature(parameters=[Parameter('self', Parameter.POSITIONAL_OR_KEYWORD)])
        assert sig == expected_sig

    def test_update_class(self) -> None:
        """Don't annotate the first arg of classmethods"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.a_class_method.__func__)
        sig = update_signature_args(sig, {'cls': Type[UpdateSignatureHelper]}, True)
        expected_sig: Signature = Signature(parameters=[Parameter('cls', Parameter.POSITIONAL_OR_KEYWORD)])
        assert sig == expected_sig

    def test_update_arg_ignore_existing_anno(self) -> None:
        """Update stubs only bases on traces."""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': str, 'b': bool}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        params = [
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=bool)
        ]
        assert sig == Signature(parameters=params, return_annotation=int)

    def test_update_self_ignore_existing_anno(self) -> None:
        """Don't annotate first arg of instance methods if asked to ignore"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.an_instance_method)
        sig = update_signature_args(sig, {'self': UpdateSignatureHelper}, has_self=True, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        expected_sig: Signature = Signature(parameters=[Parameter('self', Parameter.POSITIONAL_OR_KEYWORD)])
        assert sig == expected_sig

    def test_update_arg_ignore_existing_anno_None(self) -> None:
        """Update arg annotations from types"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': None, 'b': int}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        params = [
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=inspect.Parameter.empty),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
        ]
        assert sig == Signature(parameters=params, return_annotation=int)

    def test_update_arg_avoid_incompatible_anno(self) -> None:
        """Can generate stub with no annotations where they already exist in the source."""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': int, 'b': int}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.OMIT)
        params = [
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=inspect.Parameter.empty),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
        ]
        assert sig == Signature(parameters=params, return_annotation=int)

class TestUpdateSignatureReturn:
    def test_update_return(self) -> None:
        """Update return annotations from types"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=str)
        assert sig == Signature(return_annotation=str)

    def test_update_return_with_anno(self) -> None:
        """Leave existing return annotations alone"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_return(sig, return_type=str)
        expected_sig: Signature = Signature(parameters=[
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)
        ], return_annotation=int)
        assert sig == expected_sig

    def test_avoid_incompatible_return(self) -> None:
        """Generate stub for application with no annotation where source has one"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_return(sig, return_type=str, existing_annotation_strategy=ExistingAnnotationStrategy.OMIT)
        expected_sig: Signature = Signature(parameters=[
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)
        ])
        assert sig == expected_sig

    def test_update_return_with_anno_ignored(self) -> None:
        """Leave existing return annotations alone"""
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_return(sig, return_type=str, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        expected_sig: Signature = Signature(parameters=[
            Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)
        ], return_annotation=str)
        assert sig == expected_sig

    def test_update_yield(self) -> None:
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, yield_type=int)
        assert sig == Signature(return_annotation=Iterator[int])
        sig = update_signature_return(sig, return_type=NoneType, yield_type=int)
        assert sig == Signature(return_annotation=Iterator[int])

    def test_update_yield_and_return(self) -> None:
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=str, yield_type=int)
        assert sig == Signature(return_annotation=Generator[int, NoneType, str])

    def test_update_yield_none_and_return(self) -> None:
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=str, yield_type=NoneType)
        assert sig == Signature(return_annotation=Generator[NoneType, NoneType, str])

    def test_update_yield_and_return_none(self) -> None:
        sig: Signature = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=NoneType, yield_type=str)
        assert sig == Signature(return_annotation=Iterator[str])

def a_module_func() -> None:
    pass

async def an_async_func() -> None:
    pass

class TestFunctionKind:
    cases: List[Tuple[Callable[..., Any], FunctionKind]] = [
        (Dummy.a_static_method, FunctionKind.STATIC),
        (Dummy.a_class_method.__func__, FunctionKind.CLASS),
        (Dummy.an_instance_method, FunctionKind.INSTANCE),
        (Dummy.a_property.fget, FunctionKind.PROPERTY),
        (a_module_func, FunctionKind.MODULE)
    ]
    if cached_property:
        cases.append((Dummy.a_cached_property.func, FunctionKind.DJANGO_CACHED_PROPERTY))

    @pytest.mark.parametrize('func, expected', cases)
    def test_from_callable(self, func: Callable[..., Any], expected: FunctionKind) -> None:
        assert FunctionKind.from_callable(func) == expected

class TestFunctionDefinition:
    cases_has_self: List[Tuple[Callable[..., Any], bool]] = [
        (Dummy.a_static_method, False),
        (Dummy.a_class_method.__func__, True),
        (Dummy.an_instance_method, True),
        (Dummy.a_property.fget, True),
        (a_module_func, False)
    ]
    if cached_property:
        cases_has_self.append((Dummy.a_cached_property.func, True))

    @pytest.mark.parametrize('func, expected', cases_has_self)
    def test_has_self(self, func: Callable[..., Any], expected: bool) -> None:
        defn: FunctionDefinition = FunctionDefinition.from_callable(func)
        assert defn.has_self == expected

    cases_from_callable: List[Tuple[Callable[..., Any], FunctionDefinition]] = [
        (Dummy.a_static_method, FunctionDefinition('tests.util', 'Dummy.a_static_method', FunctionKind.STATIC, Signature.from_callable(Dummy.a_static_method))),
        (Dummy.a_class_method.__func__, FunctionDefinition('tests.util', 'Dummy.a_class_method', FunctionKind.CLASS, Signature.from_callable(Dummy.a_class_method.__func__))),
        (Dummy.an_instance_method, FunctionDefinition('tests.util', 'Dummy.an_instance_method', FunctionKind.INSTANCE, Signature.from_callable(Dummy.an_instance_method))),
        (Dummy.a_property.fget, FunctionDefinition('tests.util', 'Dummy.a_property', FunctionKind.PROPERTY, Signature.from_callable(Dummy.a_property.fget))),
        (a_module_func, FunctionDefinition('tests.test_stubs', 'a_module_func', FunctionKind.MODULE, Signature.from_callable(a_module_func))),
        (an_async_func, FunctionDefinition('tests.test_stubs', 'an_async_func', FunctionKind.MODULE, Signature.from_callable(a_module_func), is_async=True))
    ]
    if cached_property:
        cases_from_callable.append((Dummy.a_cached_property.func, FunctionDefinition('tests.util', 'Dummy.a_cached_property', FunctionKind.DJANGO_CACHED_PROPERTY, Signature.from_callable(Dummy.a_cached_property.func))))

    @pytest.mark.parametrize('func, expected', cases_from_callable)
    def test_from_callable(self, func: Callable[..., Any], expected: FunctionDefinition) -> None:
        defn: FunctionDefinition = FunctionDefinition.from_callable(func)
        assert defn == expected

    cases_from_callable_and_traced: List[Tuple[Callable[..., Any], Dict[str, Any], Any, Optional[Any], FunctionDefinition]] = [
        (
            Dummy.an_instance_method,
            {'foo': int, 'bar': List[str]},
            int,
            None,
            FunctionDefinition(
                'tests.util',
                'Dummy.an_instance_method',
                FunctionKind.INSTANCE,
                Signature(
                    parameters=[
                        Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Parameter.empty),
                        Parameter(name='foo', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                        Parameter(name='bar', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=List[str])
                    ],
                    return_annotation=int
                ),
                False,
                []
            )
        ),
        (
            Dummy.an_instance_method,
            {'foo': make_typed_dict(required_fields={'a': int, 'b': str}),
             'bar': make_typed_dict(required_fields={'c': int})},
            int,
            None,
            FunctionDefinition(
                'tests.util',
                'Dummy.an_instance_method',
                FunctionKind.INSTANCE,
                Signature(
                    parameters=[
                        Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Parameter.empty),
                        Parameter(name='foo', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=make_forward_ref('FooTypedDict__RENAME_ME__')),
                        Parameter(name='bar', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=make_forward_ref('BarTypedDict__RENAME_ME__'))
                    ],
                    return_annotation=int
                ),
                False,
                [
                    ClassStub(
                        name='FooTypedDict__RENAME_ME__(TypedDict)',
                        function_stubs=[],
                        attribute_stubs=[AttributeStub('a', int), AttributeStub('b', str)]
                    ),
                    ClassStub(
                        name='BarTypedDict__RENAME_ME__(TypedDict)',
                        function_stubs=[],
                        attribute_stubs=[AttributeStub('c', int)]
                    )
                ]
            )
        )
    ]
    @pytest.mark.parametrize('func, arg_types, return_type, yield_type, expected', cases_from_callable_and_traced)
    def test_from_callable_and_traced_types(
        self,
        func: Callable[..., Any],
        arg_types: Dict[str, Any],
        return_type: Any,
        yield_type: Optional[Any],
        expected: FunctionDefinition
    ) -> None:
        function: FunctionDefinition = FunctionDefinition.from_callable_and_traced_types(
            func, arg_types, return_type, yield_type, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE
        )
        assert function == expected

def tie_helper(a: Any, b: Any) -> None:
    pass

class TestShrinkTracedTypes:
    def test_shrink_args(self) -> None:
        traces: List[CallTrace] = [
            CallTrace(tie_helper, {'a': str, 'b': int}),
            CallTrace(tie_helper, {'a': str, 'b': NoneType})
        ]
        assert shrink_traced_types(traces, max_typed_dict_size=0) == ({'a': str, 'b': Optional[int]}, None, None)

    def test_shrink_return(self) -> None:
        traces: List[CallTrace] = [
            CallTrace(tie_helper, {}, NoneType),
            CallTrace(tie_helper, {}, str)
        ]
        assert shrink_traced_types(traces, max_typed_dict_size=0) == ({}, Optional[str], None)

    def test_shrink_yield(self) -> None:
        traces: List[CallTrace] = [
            CallTrace(tie_helper, {}, yield_type=int),
            CallTrace(tie_helper, {}, yield_type=str)
        ]
        assert shrink_traced_types(traces, max_typed_dict_size=0) == ({}, None, Union[int, str])

class Parent:
    class Child:
        pass

class TestGetImportsForAnnotation:
    @pytest.mark.parametrize('anno', [inspect.Parameter.empty, inspect.Signature.empty, 'not a type', int])
    def test_no_imports(self, anno: Any) -> None:
        """We shouldn't import any builtins, non-types, or empty annos"""
        assert get_imports_for_annotation(anno) == {}

    @pytest.mark.parametrize('anno, expected', [
        (Any, {'typing': {'Any'}}),
        (Union[int, str], {'typing': {'Union'}})
    ])
    def test_special_case_types(self, anno: Any, expected: Dict[str, Set[str]]) -> None:
        """Any and Union do not have module/qualname and need to be treated specially"""
        assert get_imports_for_annotation(anno) == expected

    def test_callable(self) -> None:
        assert get_imports_for_annotation(Callable) == {'typing': {'Callable'}}

    def test_user_defined_class(self) -> None:
        assert get_imports_for_annotation(Dummy) == {'tests.util': {'Dummy'}}

    @pytest.mark.parametrize('anno, expected', [
        (Dict[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Dict'}}),
        (List[Dummy], {'tests.util': {'Dummy'}, 'typing': {'List'}}),
        (Set[Dummy], {'tests.util': {'Dummy'}, 'typing': {'Set'}}),
        (Tuple[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Tuple'}}),
        (Type[Dummy], {'tests.util': {'Dummy'}, 'typing': {'Type'}}),
        (Union[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Union'}})
    ])
    def test_container_types(self, anno: Any, expected: Dict[str, Set[str]]) -> None:
        """We need to descend into container types"""
        assert get_imports_for_annotation(anno) == expected

    def test_nested_class(self) -> None:
        assert get_imports_for_annotation(Parent.Child) == {Parent.__module__: {'Parent'}}

class TestGetImportsForSignature:
    def test_default_none_parameter_imports(self) -> None:
        stub: FunctionStub = FunctionStub('test', inspect.signature(default_none_parameter), FunctionKind.MODULE)
        expected: Dict[str, Set[str]] = {'typing': {'Optional'}}
        assert get_imports_for_signature(stub.signature) == expected
