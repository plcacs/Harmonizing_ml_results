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
    """Test the ImportMap class."""

    def test_merge(self) -> None:
        """Test merging two ImportMaps."""
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
    """Test the ImportBlockStub class."""

    def test_single_import(self) -> None:
        """Test rendering a single import."""
        imports = ImportMap()
        imports['a.module'] = {'AClass'}
        imports['another.module'] = {'AnotherClass'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from a.module import AClass', 'from another.module import AnotherClass'])
        assert stub.render() == expected

    def test_io_import_single(self) -> None:
        """Test rendering a single _io import."""
        imports = ImportMap()
        imports['_io'] = {'BytesIO'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from io import BytesIO'])
        assert stub.render() == expected

    def test_multiple_imports(self) -> None:
        """Test rendering multiple imports from a single module."""
        imports = ImportMap()
        imports['a.module'] = {'AClass', 'AnotherClass', 'AThirdClass'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from a.module import (', '    AClass,', '    AThirdClass,', '    AnotherClass,', ')'])
        assert stub.render() == expected

    def test_multiple_io_imports(self) -> None:
        """Test rendering multiple imports from single _io module."""
        imports = ImportMap()
        imports['_io'] = {'BytesIO', 'FileIO'}
        stub = ImportBlockStub(imports)
        expected = '\n'.join(['from io import (', '    BytesIO,', '    FileIO,', ')'])
        assert stub.render() == expected

def simple_add(a: int, b: int) -> int:
    """Simple function that adds two numbers."""
    return a + b

def strip_modules_helper(d1: Dummy, d2: Dummy) -> None:
    """Helper function for testing module stripping."""
    pass

def has_optional_param(x: Optional[int] = None) -> None:
    """Function with an optional parameter."""
    pass

def has_optional_union_param(x: Optional[Union[int, float]]) -> None:
    """Function with an optional union parameter."""
    pass

def has_optional_return() -> Optional[int]:
    """Function with an optional return type."""
    return None

def default_none_parameter(x: Optional[int] = None) -> None:
    """Function with an optional parameter and default value."""
    pass

def has_length_exceeds_120_chars(very_long_name_parameter_1: float, very_long_name_parameter_2: float) -> Optional[float]:
    """Function with a long parameter name."""
    return None

def has_newtype_param(user_id: UserId) -> None:
    """Function with a NewType parameter."""
    pass

def has_forward_ref() -> Optional['TestFunctionStub']:
    """Function with a forward reference."""
    pass

def has_forward_ref_within_generator() -> Generator['TestFunctionStub', None, int]:
    """Function with a forward reference within a generator."""
    pass

class TestAttributeStub:
    """Test the AttributeStub class."""

    @pytest.mark.parametrize('stub, expected', [(AttributeStub(name='foo', typ=int), '    foo: int'), (AttributeStub(name='foo', typ=make_forward_ref('Foo')), "    foo: 'Foo'")])
    def test_simple_attribute(self, stub: AttributeStub, expected: str) -> None:
        """Test rendering a simple attribute."""
        assert stub.render('    ') == expected

class TestRenderAnnotation:
    """Test the render_annotation function."""

    @pytest.mark.parametrize('annotation, expected', [(make_forward_ref('Foo'), "'Foo'"), (List[make_forward_ref('Foo')], "List['Foo']"), (List[List[make_forward_ref('Foo')]], "List[List['Foo']]"), (Optional[int], 'Optional[int]'), (List[Optional[int]], 'List[Optional[int]]'), (UserId, 'UserId'), (List[UserId], 'List[UserId]'), (List[int], 'List[int]'), (List[List[int]], 'List[List[int]]'), (None, 'None'), (List[None], 'List[None]'), (int, 'int'), (Dummy, 'tests.util.Dummy'), (List[Dummy], 'List[tests.util.Dummy]'), ('some_string', 'some_string'), (Iterable[None], 'Iterable[None]'), (List[Iterable[None]], 'List[Iterable[None]]'), (Generator[make_forward_ref('Foo'), None, None], "Generator['Foo', None, None]"), (List[Generator[make_forward_ref('Foo'), None, None]], "List[Generator['Foo', None, None]]"), (T, 'T'), (Dict[str, T], 'Dict[str, T]'), (Tuple[()], 'Tuple[()]')])
    def test_render_annotation(self, annotation: Any, expected: str) -> None:
        """Test rendering an annotation."""
        assert render_annotation(annotation) == expected

class TestFunctionStub:
    """Test the FunctionStub class."""

    def test_classmethod(self) -> None:
        """Test rendering a classmethod."""
        stub = FunctionStub('test', inspect.signature(Dummy.a_class_method), FunctionKind.CLASS)
        expected = '\n'.join(['@classmethod', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    def test_staticmethod(self) -> None:
        """Test rendering a staticmethod."""
        stub = FunctionStub('test', inspect.signature(Dummy.a_static_method), FunctionKind.STATIC)
        expected = '\n'.join(['@staticmethod', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    def test_property(self) -> None:
        """Test rendering a property."""
        stub = FunctionStub('test', inspect.signature(Dummy.a_property.fget), FunctionKind.PROPERTY)
        expected = '\n'.join(['@property', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    @skipIf(cached_property is None, 'install Django to run this test')
    def test_cached_property(self) -> None:
        """Test rendering a cached property."""
        stub = FunctionStub('test', inspect.signature(Dummy.a_cached_property.func), FunctionKind.DJANGO_CACHED_PROPERTY)
        expected = '\n'.join(['@cached_property', 'def test%s: ...' % (render_signature(stub.signature),)])
        assert stub.render() == expected

    def test_simple(self) -> None:
        """Test rendering a simple function."""
        for kind in [FunctionKind.MODULE, FunctionKind.INSTANCE]:
            stub = FunctionStub('test', inspect.signature(simple_add), kind)
            expected = 'def test%s: ...' % (render_signature(stub.signature),)
            assert stub.render() == expected

    def test_with_prefix(self) -> None:
        """Test rendering a function with a prefix."""
        stub = FunctionStub('test', inspect.signature(simple_add), FunctionKind.MODULE)
        expected = '  def test%s: ...' % (render_signature(stub.signature),)
        assert stub.render(prefix='  ') == expected

    def test_strip_modules(self) -> None:
        """Test stripping modules from annotations."""
        to_strip = [Dummy.__module__]
        f = strip_modules_helper
        stub = FunctionStub(f.__name__, inspect.signature(f), FunctionKind.MODULE, to_strip)
        expected = 'def strip_modules_helper(d1: Dummy, d2: Dummy) -> None: ...'
        assert stub.render() == expected

    def test_async_function(self) -> None:
        """Test rendering an async function."""
        stub = FunctionStub('test', inspect.signature(simple_add), FunctionKind.MODULE, is_async=True)
        expected = 'async def test%s: ...' % (render_signature(stub.signature),)
        assert stub.render() == expected

    def test_optional_parameter_annotation(self) -> None:
        """Test rendering an optional parameter annotation."""
        stub = FunctionStub('test', inspect.signature(has_optional_param), FunctionKind.MODULE)
        expected = 'def test(x: Optional[int] = ...) -> None: ...'
        assert stub.render() == expected

    def test_optional_union_parameter_annotation(self) -> None:
        """Test rendering an optional union parameter annotation."""
        stub = FunctionStub('test', inspect.signature(has_optional_union_param), FunctionKind.MODULE)
        expected = 'def test(x: Optional[Union[int, float]]) -> None: ...'
        assert stub.render() == expected

    def test_optional_return_annotation(self) -> None:
        """Test rendering an optional return annotation."""
        stub = FunctionStub('test', inspect.signature(has_optional_return), FunctionKind.MODULE)
        expected = 'def test() -> Optional[int]: ...'
        assert stub.render() == expected

    def test_split_parameters_across_multiple_lines(self) -> None:
        """Test splitting parameters across multiple lines."""
        stub = FunctionStub('has_length_exceeds_120_chars', inspect.signature(has_length_exceeds_120_chars), FunctionKind.MODULE)
        expected = dedent('        def has_length_exceeds_120_chars(\n            very_long_name_parameter_1: float,\n            very_long_name_parameter_2: float\n        ) -> Optional[float]: ...')
        assert stub.render() == expected
        expected = '\n'.join(['    def has_length_exceeds_120_chars(', '        very_long_name_parameter_1: float,', '        very_long_name_parameter_2: float', '    ) -> Optional[float]: ...'])
        assert stub.render(prefix='    ') == expected

    def test_default_none_parameter_annotation(self) -> None:
        """Test rendering a default None parameter annotation."""
        stub = FunctionStub('test', inspect.signature(default_none_parameter), FunctionKind.MODULE)
        expected = 'def test(x: Optional[int] = ...) -> None: ...'
        assert stub.render() == expected

    def test_newtype_parameter_annotation(self) -> None:
        """Test rendering a NewType parameter annotation."""
        stub = FunctionStub('test', inspect.signature(has_newtype_param), FunctionKind.MODULE)
        expected = 'def test(user_id: UserId) -> None: ...'
        assert stub.render() == expected

    def test_nonetype_annotation(self) -> None:
        """Test rendering a NoneType annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': Dict[str, NoneType]}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        stub = FunctionStub('test', sig, FunctionKind.MODULE)
        expected = 'def test(a: Dict[str, None], b) -> int: ...'
        assert stub.render() == expected

    def test_forward_ref_annotation(self) -> None:
        """Test rendering a forward reference annotation."""
        stub = FunctionStub('has_forward_ref', inspect.signature(has_forward_ref), FunctionKind.MODULE)
        expected = "def has_forward_ref() -> Optional['TestFunctionStub']: ..."
        assert stub.render() == expected

    @pytest.mark.xfail(reason='We get Generator[ForwardRef(), ...].')
    def test_forward_ref_annotation_within_generator(self) -> None:
        """Test rendering a forward reference annotation within a generator."""
        stub = FunctionStub('foo', inspect.signature(has_forward_ref_within_generator), FunctionKind.MODULE)
        expected = "def foo() -> Generator['TestFunctionStub', None, int]: ..."
        assert stub.render() == expected

def _func_stub_from_callable(func: Callable, strip_modules: List[str] = None) -> FunctionStub:
    """Helper function to create a FunctionStub from a callable."""
    kind = FunctionKind.from_callable(func)
    sig = Signature.from_callable(func)
    return FunctionStub(func.__name__, sig, kind, strip_modules)

class TestClassStub:
    """Test the ClassStub class."""

    def test_render(self) -> None:
        """Test rendering a class."""
        cm_stub = _func_stub_from_callable(Dummy.a_class_method.__func__)
        im_stub = _func_stub_from_callable(Dummy.an_instance_method)
        class_stub = ClassStub('Test', function_stubs=(cm_stub, im_stub), attribute_stubs=[AttributeStub('foo', int), AttributeStub('bar', str)])
        expected = '\n'.join(['class Test:', '    bar: str', '    foo: int', '    @classmethod', '    def a_class_method(cls, foo: Any) -> Optional[frame]: ...', '    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...'])
        assert class_stub.render() == expected

class TestReplaceTypedDictsWithStubs:
    """Test the ReplaceTypedDictsWithStubs class."""

    SIMPLE_TYPED_DICT_STUB = ClassStub(name='FooBarTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)])
    SIMPLE_TYPED_DICT_STUB2 = ClassStub(name='FooBar2TypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)])
    SIMPLE_NON_TOTAL_TYPED_DICT_STUB = ClassStub(name='FooBarTypedDict__RENAME_ME__(TypedDict, total=False)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)])
    SIMPLE_BASE_AND_SUBCLASS = [ClassStub(name='FooBarTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)]), ClassStub(name='FooBarTypedDict__RENAME_ME__NonTotal(FooBarTypedDict__RENAME_ME__, total=False)', function_stubs=[], attribute_stubs=[AttributeStub(name='c', typ=int)])]

    @pytest.mark.parametrize('typ, expected', [(int, (int, [])), (List[int], (List[int], [])), (Set[int], (Set[int], [])), (Dict[str, int], (Dict[str, int], [])), (Tuple[str, int], (Tuple[str, int], [])), (List[List[Dict[str, int]]], (List[List[Dict[str, int]]], [])), (List[List[Dict[str, int]]], (List[List[Dict[str, int]]], [])), (List[List[make_typed_dict(required_fields={'a': int, 'b': str})]], (List[List[make_forward_ref('FooBarTypedDict__RENAME_ME__')]], [SIMPLE_TYPED_DICT_STUB])), (Dict[str, make_typed_dict(required_fields={'a': int, 'b': str})], (Dict[str, make_forward_ref('FooBar2TypedDict__RENAME_ME__')], [SIMPLE_TYPED_DICT_STUB2])), (Set[make_typed_dict(required_fields={'a': int, 'b': str})], (Set[make_forward_ref('FooBarTypedDict__RENAME_ME__')], [SIMPLE_TYPED_DICT_STUB])), (Tuple[int, make_typed_dict(required_fields={'a': int, 'b': str})], (Tuple[int, make_forward_ref('FooBar2TypedDict__RENAME_ME__')], [SIMPLE_TYPED_DICT_STUB2])), (make_typed_dict(required_fields={'a': int, 'b': str}), (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [SIMPLE_TYPED_DICT_STUB])), (make_typed_dict(optional_fields={'a': int, 'b': str}), (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [SIMPLE_NON_TOTAL_TYPED_DICT_STUB])), (make_typed_dict(required_fields={'a': int, 'b': str}, optional_fields={'c': int}), (make_forward_ref('FooBarTypedDict__RENAME_ME__NonTotal'), SIMPLE_BASE_AND_SUBCLASS)), (TypedDict('GenuineTypedDict', {'a': int, 'b': str}), (TypedDict('GenuineTypedDict', {'a': int, 'b': str}), [])), (make_typed_dict(required_fields={'a': int, 'b': make_typed_dict(required_fields={'a': int, 'b': str})}), (make_forward_ref('FooBarTypedDict__RENAME_ME__'), [ClassStub(name='BTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=str)]), ClassStub(name='FooBarTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int), AttributeStub(name='b', typ=make_forward_ref('BTypedDict__RENAME_ME__'))])])), (Tuple[make_typed_dict(required_fields={'a': int}), make_typed_dict(required_fields={'b': str})], (Tuple[make_forward_ref('FooBarTypedDict__RENAME_ME__'), make_forward_ref('FooBar2TypedDict__RENAME_ME__')], [ClassStub(name='FooBarTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='a', typ=int)]), ClassStub(name='FooBar2TypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub(name='b', typ=str)])]))])
    def test_replace_typed_dict_with_stubs(self, typ: Any, expected: Tuple[Any, List[ClassStub]]) -> None:
        """Test replacing TypedDicts with stubs."""
        rewritten_type, stubs = ReplaceTypedDictsWithStubs.rewrite_and_get_stubs(typ, class_name_hint='foo_bar')
        actual = (rewritten_type, stubs)
        assert actual == expected

typed_dict_import_map = ImportMap()
typed_dict_import_map['mypy_extensions'] = {'TypedDict'}
module_stub_for_method_with_typed_dict = {'tests.util': ModuleStub(function_stubs=(), class_stubs=[ClassStub(name='Dummy', function_stubs=[FunctionStub(name='an_instance_method', signature=Signature(parameters=[Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Parameter.empty), Parameter(name='foo', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=make_forward_ref('FooTypedDict__RENAME_ME__')), Parameter(name='bar', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=int)], return_annotation=make_forward_ref('DummyAnInstanceMethodTypedDict__RENAME_ME__')), kind=FunctionKind.INSTANCE, strip_modules=['mypy_extensions'], is_async=False)])], imports_stub=ImportBlockStub(typed_dict_import_map), typed_dict_class_stubs=[ClassStub(name='FooTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub('a', int), AttributeStub('b', str)]), ClassStub(name='DummyAnInstanceMethodTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub('c', int)])])}

class TestModuleStub:
    """Test the ModuleStub class."""

    def test_render(self) -> None:
        """Test rendering a module."""
        cm_stub = _func_stub_from_callable(Dummy.a_class_method)
        im_stub = _func_stub_from_callable(Dummy.an_instance_method)
        sig_stub = _func_stub_from_callable(Dummy.has_complex_signature)
        func_stubs = (cm_stub, im_stub, sig_stub)
        test_stub = ClassStub('Test', function_stubs=func_stubs)
        test2_stub = ClassStub('Test2', function_stubs=func_stubs)
        other_class_stubs = module_stub_for_method_with_typed_dict['tests.util'].class_stubs.values()
        class_stubs = (*other_class_stubs, test_stub, test2_stub)
        typed_dict_class_stubs = module_stub_for_method_with_typed_dict['tests.util'].typed_dict_class_stubs
        mod_stub = ModuleStub(function_stubs=func_stubs, class_stubs=class_stubs, typed_dict_class_stubs=typed_dict_class_stubs)
        expected = '\n'.join(['class DummyAnInstanceMethodTypedDict__RENAME_ME__(TypedDict):', '    c: int', '', '', 'class FooTypedDict__RENAME_ME__(TypedDict):', '    a: int', '    b: str', '', '', '@classmethod', 'def a_class_method(foo: Any) -> Optional[frame]: ...', '', '', 'def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...', '', '', 'def has_complex_signature(', '    self,', '    a: Any,', '    b: Any,', '    /,', '    c: Any,', '    d: Any = ...,', '    *e: Any,', '    f: Any,', '    g: Any = ...,', '    **h: Any', ') -> Optional[frame]: ...', '', '', 'class Dummy:', '    def an_instance_method(', '        self,', "        foo: 'FooTypedDict__RENAME_ME__',", '        bar: int', "    ) -> 'DummyAnInstanceMethodTypedDict__RENAME_ME__': ...", '', '', 'class Test:', '    @classmethod', '    def a_class_method(foo: Any) -> Optional[frame]: ...', '    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...', '    def has_complex_signature(', '        self,', '        a: Any,', '        b: Any,', '        /,', '        c: Any,', '        d: Any = ...,', '        *e: Any,', '        f: Any,', '        g: Any = ...,', '        **h: Any', '    ) -> Optional[frame]: ...', '', '', 'class Test2:', '    @classmethod', '    def a_class_method(foo: Any) -> Optional[frame]: ...', '    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]: ...', '    def has_complex_signature(', '        self,', '        a: Any,', '        b: Any,', '        /,', '        c: Any,', '        d: Any = ...,', '        *e: Any,', '        f: Any,', '        g: Any = ...,', '        **h: Any', '    ) -> Optional[frame]: ...'])
        assert mod_stub.render() == expected

    def test_render_nested_typed_dict(self) -> None:
        """Test rendering a nested TypedDict."""
        function = FunctionDefinition.from_callable_and_traced_types(Dummy.an_instance_method, {'foo': make_typed_dict(required_fields={'z': make_typed_dict(required_fields={'a': int, 'b': str}), 'b': str}), 'bar': int}, int, None, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        entries = [function]
        expected = '\n'.join(['from mypy_extensions import TypedDict', '', '', 'class FooTypedDict__RENAME_ME__(TypedDict):', '    b: str', "    z: 'ZTypedDict__RENAME_ME__'", '', '', 'class ZTypedDict__RENAME_ME__(TypedDict):', '    a: int', '    b: str', '', '', 'class Dummy:', "    def an_instance_method(self, foo: 'FooTypedDict__RENAME_ME__', bar: int) -> int: ..."])
        self.maxDiff = None
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_return_typed_dict(self) -> None:
        """Test rendering a return TypedDict."""
        function = FunctionDefinition.from_callable_and_traced_types(Dummy.an_instance_method, {'foo': int, 'bar': int}, make_typed_dict(required_fields={'a': int, 'b': str}), yield_type=None, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        entries = [function]
        expected = '\n'.join(['from mypy_extensions import TypedDict', '', '', 'class DummyAnInstanceMethodTypedDict__RENAME_ME__(TypedDict):', '    a: int', '    b: str', '', '', 'class Dummy:', "    def an_instance_method(self, foo: int, bar: int) -> 'DummyAnInstanceMethodTypedDict__RENAME_ME__': ..."])
        self.maxDiff = None
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_yield_typed_dict(self) -> None:
        """Test rendering a yield TypedDict."""
        function = FunctionDefinition.from_callable_and_traced_types(Dummy.an_instance_method, {'foo': int, 'bar': int}, int, yield_type=make_typed_dict(required_fields={'a': int, 'b': str}), existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        entries = [function]
        expected = '\n'.join(['from mypy_extensions import TypedDict', 'from typing import Generator', '', '', 'class DummyAnInstanceMethodYieldTypedDict__RENAME_ME__(TypedDict):', '    a: int', '    b: str', '', '', 'class Dummy:', '    def an_instance_method(', '        self,', '        foo: int,', '        bar: int', "    ) -> Generator['DummyAnInstanceMethodYieldTypedDict__RENAME_ME__', None, int]: ..."])
        self.maxDiff = None
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_typed_dict_in_list(self) -> None:
        """Test rendering a TypedDict in a list."""
        function = FunctionDefinition.from_callable_and_traced_types(Dummy.an_instance_method, {'foo': List[make_typed_dict(required_fields={'a': int})], 'bar': int}, int, None, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        entries = [function]
        expected = '\n'.join(['from mypy_extensions import TypedDict', 'from typing import List', '', '', 'class FooTypedDict__RENAME_ME__(TypedDict):', '    a: int', '', '', 'class Dummy:', "    def an_instance_method(self, foo: List['FooTypedDict__RENAME_ME__'], bar: int) -> int: ..."])
        self.maxDiff = None
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_typed_dict_base_and_subclass(self) -> None:
        """Test rendering a TypedDict base and subclass."""
        function = FunctionDefinition.from_callable_and_traced_types(Dummy.an_instance_method, {'foo': make_typed_dict(required_fields={'a': int}, optional_fields={'b': str}), 'bar': int}, int, None, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        entries = [function]
        expected = '\n'.join(['from mypy_extensions import TypedDict', '', '', 'class FooTypedDict__RENAME_ME__(TypedDict):', '    a: int', '', '', 'class FooTypedDict__RENAME_ME__NonTotal(FooTypedDict__RENAME_ME__, total=False):', '    b: str', '', '', 'class Dummy:', "    def an_instance_method(self, foo: 'FooTypedDict__RENAME_ME__NonTotal', bar: int) -> int: ..."])
        assert build_module_stubs(entries)['tests.util'].render() == expected

    def test_render_return_empty_tuple(self) -> None:
        """Test rendering an empty tuple return."""
        function = FunctionDefinition.from_callable_and_traced_types(Dummy.an_instance_method, {'foo': int, 'bar': int}, Tuple[()], yield_type=None, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        entries = [function]
        expected = '\n'.join(['from typing import Tuple', '', '', 'class Dummy:', '    def an_instance_method(self, foo: int, bar: int) -> Tuple[()]: ...'])
        self.maxDiff = None
        assert build_module_stubs(entries)['tests.util'].render() == expected

class TestBuildModuleStubs:
    """Test the build_module_stubs function."""

    def test_build_module_stubs(self) -> None:
        """Test building module stubs."""
        entries = [FunctionDefinition.from_callable(Dummy.a_static_method), FunctionDefinition.from_callable(Dummy.a_class_method.__func__), FunctionDefinition.from_callable(Dummy.an_instance_method), FunctionDefinition.from_callable(simple_add)]
        simple_add_stub = _func_stub_from_callable(simple_add)
        to_strip = ['typing']
        dummy_stub = ClassStub('Dummy', function_stubs=[_func_stub_from_callable(Dummy.a_class_method.__func__, to_strip), _func_stub_from_callable(Dummy.an_instance_method, to_strip), _func_stub_from_callable(Dummy.a_static_method, to_strip)])
        imports = {'typing': {'Any', 'Optional'}}
        expected = {'tests.test_stubs': ModuleStub(function_stubs=[simple_add_stub]), 'tests.util': ModuleStub(class_stubs=[dummy_stub], imports_stub=ImportBlockStub(imports))}
        self.maxDiff = None
        assert build_module_stubs(entries) == expected

    def test_build_module_stubs_typed_dict_parameter(self) -> None:
        """Test building module stubs with a TypedDict parameter."""
        function = FunctionDefinition.from_callable_and_traced_types(Dummy.an_instance_method, {'foo': make_typed_dict(required_fields={'a': int, 'b': str}), 'bar': int}, make_typed_dict(required_fields={'c': int}), None, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        entries = [function]
        expected = module_stub_for_method_with_typed_dict
        self.maxDiff = None
        assert build_module_stubs(entries) == expected

def untyped_helper(x: int, y: str) -> str:
    """Helper function for testing the StubIndexBuilder."""
    pass

class TestStubIndexBuilder:
    """Test the StubIndexBuilder class."""

    def test_ignore_non_matching_functions(self) -> None:
        """Test ignoring non-matching functions."""
        b = StubIndexBuilder('foo.bar', max_typed_dict_size=0)
        b.log(CallTrace(untyped_helper, {'x': int, 'y': str}))
        assert len(b.index) == 0

    def test_build_index(self) -> None:
        """Test building the index."""
        idxb = StubIndexBuilder('tests', max_typed_dict_size=0)
        idxb.log(CallTrace(untyped_helper, {'x': int, 'y': str}, str))
        sig = Signature.from_callable(untyped_helper)
        sig = sig.replace(parameters=[Parameter('x', Parameter.POSITIONAL_OR_KEYWORD, annotation=int), Parameter('y', Parameter.POSITIONAL_OR_KEYWORD, annotation=str)], return_annotation=str)
        mod_stub = ModuleStub(function_stubs=[FunctionStub('untyped_helper', sig, FunctionKind.MODULE)])
        expected = {'tests.test_stubs': mod_stub}
        assert idxb.get_stubs() == expected

class UpdateSignatureHelper:
    """Helper class for testing signature updates."""

    @staticmethod
    def has_annos(a: int, b: int) -> int:
        """Function for testing signature updates."""
        return 0

    @classmethod
    def a_class_method(cls: Type['UpdateSignatureHelper']) -> None:
        """Classmethod for testing signature updates."""
        pass

    def an_instance_method(self, foo: Any, bar: Any) -> Optional[frame]:
        """Instancemethod for testing signature updates."""
        pass

class TestUpdateSignatureArgs:
    """Test the update_signature_args function."""

    def test_update_arg(self) -> None:
        """Test updating an argument annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'b': int}, False)
        params = [Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=int)]
        assert sig == Signature(parameters=params, return_annotation=int)

    def test_update_arg_with_anno(self) -> None:
        """Test updating an argument annotation when the existing annotation is not empty."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': str}, False)
        expected = Signature(parameters=[Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)], return_annotation=int)
        assert sig == expected

    def test_update_self(self) -> None:
        """Test updating the self argument annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.an_instance_method)
        sig = update_signature_args(sig, {'self': UpdateSignatureHelper}, True)
        expected = Signature(parameters=[Parameter('self', Parameter.POSITIONAL_OR_KEYWORD)])
        assert sig == expected

    def test_update_class(self) -> None:
        """Test updating the class argument annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.a_class_method.__func__)
        sig = update_signature_args(sig, {'cls': Type[UpdateSignatureHelper]}, True)
        expected = Signature(parameters=[Parameter('cls', Parameter.POSITIONAL_OR_KEYWORD)])
        assert sig == expected

    def test_update_arg_ignore_existing_anno(self) -> None:
        """Test updating an argument annotation when the existing annotation is not empty and ignoring it."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': str, 'b': bool}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        params = [Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=str), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=bool)]
        assert sig == Signature(parameters=params, return_annotation=int)

    def test_update_self_ignore_existing_anno(self) -> None:
        """Test updating the self argument annotation when the existing annotation is not empty and ignoring it."""
        sig = Signature.from_callable(UpdateSignatureHelper.an_instance_method)
        sig = update_signature_args(sig, {'self': UpdateSignatureHelper}, has_self=True, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        expected = Signature(parameters=[Parameter('self', Parameter.POSITIONAL_OR_KEYWORD)])
        assert sig == expected

    def test_update_arg_ignore_existing_anno_None(self) -> None:
        """Test updating an argument annotation when the existing annotation is empty and ignoring it."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': None, 'b': int}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        params = [Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=inspect.Parameter.empty), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=int)]
        assert sig == Signature(parameters=params, return_annotation=int)

    def test_update_arg_avoid_incompatible_anno(self) -> None:
        """Test updating an argument annotation when the existing annotation is not empty and avoiding an incompatible annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_args(sig, {'a': int, 'b': int}, has_self=False, existing_annotation_strategy=ExistingAnnotationStrategy.OMIT)
        params = [Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=inspect.Parameter.empty), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD, annotation=int)]
        assert sig == Signature(parameters=params, return_annotation=int)

class TestUpdateSignatureReturn:
    """Test the update_signature_return function."""

    def test_update_return(self) -> None:
        """Test updating the return annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=str)
        assert sig == Signature(return_annotation=str)

    def test_update_return_with_anno(self) -> None:
        """Test updating the return annotation when the existing annotation is not empty."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_return(sig, return_type=str)
        expected = Signature(parameters=[Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)], return_annotation=int)
        assert sig == expected

    def test_avoid_incompatible_return(self) -> None:
        """Test updating the return annotation when the existing annotation is not empty and avoiding an incompatible annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_return(sig, return_type=str, existing_annotation_strategy=ExistingAnnotationStrategy.OMIT)
        expected = Signature(parameters=[Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)])
        assert sig == expected

    def test_update_return_with_anno_ignored(self) -> None:
        """Test updating the return annotation when the existing annotation is not empty and ignoring it."""
        sig = Signature.from_callable(UpdateSignatureHelper.has_annos)
        sig = update_signature_return(sig, return_type=str, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        expected = Signature(parameters=[Parameter('a', Parameter.POSITIONAL_OR_KEYWORD, annotation=int), Parameter('b', Parameter.POSITIONAL_OR_KEYWORD)], return_annotation=str)
        assert sig == expected

    def test_update_yield(self) -> None:
        """Test updating the yield annotation."""
        sig = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, yield_type=int)
        assert sig == Signature(return_annotation=Iterator[int])
        sig = update_signature_return(sig, return_type=NoneType, yield_type=int)
        assert sig == Signature(return_annotation=Iterator[int])

    def test_update_yield_and_return(self) -> None:
        """Test updating the yield and return annotations."""
        sig = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=str, yield_type=int)
        assert sig == Signature(return_annotation=Generator[int, NoneType, str])

    def test_update_yield_none_and_return(self) -> None:
        """Test updating the yield and return annotations when the yield type is None."""
        sig = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=str, yield_type=NoneType)
        assert sig == Signature(return_annotation=Generator[NoneType, NoneType, str])

    def test_update_yield_and_return_none(self) -> None:
        """Test updating the yield and return annotations when the return type is None."""
        sig = Signature.from_callable(UpdateSignatureHelper.a_class_method)
        sig = update_signature_return(sig, return_type=NoneType, yield_type=str)
        assert sig == Signature(return_annotation=Iterator[str])

def a_module_func() -> None:
    """Function for testing the FunctionKind class."""
    pass

async def an_async_func() -> None:
    """Async function for testing the FunctionKind class."""
    pass

class TestFunctionKind:
    """Test the FunctionKind class."""

    cases = [(Dummy.a_static_method, FunctionKind.STATIC), (Dummy.a_class_method.__func__, FunctionKind.CLASS), (Dummy.an_instance_method, FunctionKind.INSTANCE), (Dummy.a_property.fget, FunctionKind.PROPERTY), (a_module_func, FunctionKind.MODULE)]
    if cached_property:
        cases.append((Dummy.a_cached_property.func, FunctionKind.DJANGO_CACHED_PROPERTY))

    @pytest.mark.parametrize('func, expected', cases)
    def test_from_callable(self, func: Callable, expected: FunctionKind) -> None:
        """Test getting the function kind from a callable."""
        assert FunctionKind.from_callable(func) == expected

class TestFunctionDefinition:
    """Test the FunctionDefinition class."""

    cases = [(Dummy.a_static_method, False), (Dummy.a_class_method.__func__, True), (Dummy.an_instance_method, True), (Dummy.a_property.fget, True), (a_module_func, False)]
    if cached_property:
        cases.append((Dummy.a_cached_property.func, True))

    @pytest.mark.parametrize('func, expected', cases)
    def test_has_self(self, func: Callable, expected: bool) -> None:
        """Test whether the function has a self argument."""
        defn = FunctionDefinition.from_callable(func)
        assert defn.has_self == expected
    cases = [(Dummy.a_static_method, FunctionDefinition('tests.util', 'Dummy.a_static_method', FunctionKind.STATIC, Signature.from_callable(Dummy.a_static_method))), (Dummy.a_class_method.__func__, FunctionDefinition('tests.util', 'Dummy.a_class_method', FunctionKind.CLASS, Signature.from_callable(Dummy.a_class_method.__func__))), (Dummy.an_instance_method, FunctionDefinition('tests.util', 'Dummy.an_instance_method', FunctionKind.INSTANCE, Signature.from_callable(Dummy.an_instance_method))), (Dummy.a_property.fget, FunctionDefinition('tests.util', 'Dummy.a_property', FunctionKind.PROPERTY, Signature.from_callable(Dummy.a_property.fget))), (a_module_func, FunctionDefinition('tests.test_stubs', 'a_module_func', FunctionKind.MODULE, Signature.from_callable(a_module_func))), (an_async_func, FunctionDefinition('tests.test_stubs', 'an_async_func', FunctionKind.MODULE, Signature.from_callable(a_module_func), is_async=True))]
    if cached_property:
        cases.append((Dummy.a_cached_property.func, FunctionDefinition('tests.util', 'Dummy.a_cached_property', FunctionKind.DJANGO_CACHED_PROPERTY, Signature.from_callable(Dummy.a_cached_property.func))))

    @pytest.mark.parametrize('func, expected', cases)
    def test_from_callable(self, func: Callable, expected: FunctionDefinition) -> None:
        """Test getting a FunctionDefinition from a callable."""
        defn = FunctionDefinition.from_callable(func)
        assert defn == expected

    @pytest.mark.parametrize('func, arg_types, return_type, yield_type, expected', [(Dummy.an_instance_method, {'foo': int, 'bar': List[str]}, int, None, FunctionDefinition('tests.util', 'Dummy.an_instance_method', FunctionKind.INSTANCE, Signature(parameters=[Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Parameter.empty), Parameter(name='foo', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=int), Parameter(name='bar', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=List[str])], return_annotation=int), False, [])), (Dummy.an_instance_method, {'foo': make_typed_dict(required_fields={'a': int, 'b': str}), 'bar': make_typed_dict(required_fields={'c': int})}, int, None, FunctionDefinition('tests.util', 'Dummy.an_instance_method', FunctionKind.INSTANCE, Signature(parameters=[Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Parameter.empty), Parameter(name='foo', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=make_forward_ref('FooTypedDict__RENAME_ME__')), Parameter(name='bar', kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=make_forward_ref('BarTypedDict__RENAME_ME__'))], return_annotation=int), False, [ClassStub(name='FooTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub('a', int), AttributeStub('b', str)]), ClassStub(name='BarTypedDict__RENAME_ME__(TypedDict)', function_stubs=[], attribute_stubs=[AttributeStub('c', int)])]))])
    def test_from_callable_and_traced_types(self, func: Callable, arg_types: Dict[str, Any], return_type: Any, yield_type: Any, expected: FunctionDefinition) -> None:
        """Test getting a FunctionDefinition from a callable and traced types."""
        function = FunctionDefinition.from_callable_and_traced_types(func, arg_types, return_type, yield_type, existing_annotation_strategy=ExistingAnnotationStrategy.IGNORE)
        assert function == expected

def tie_helper(a: str, b: Optional[int]) -> str:
    """Helper function for testing the shrink_traced_types function."""
    pass

class TestShrinkTracedTypes:
    """Test the shrink_traced_types function."""

    def test_shrink_args(self) -> None:
        """Test shrinking argument types."""
        traces = [CallTrace(tie_helper, {'a': str, 'b': int}), CallTrace(tie_helper, {'a': str, 'b': NoneType})]
        assert shrink_traced_types(traces, max_typed_dict_size=0) == ({'a': str, 'b': Optional[int]}, None, None)

    def test_shrink_return(self) -> None:
        """Test shrinking return types."""
        traces = [CallTrace(tie_helper, {}, NoneType), CallTrace(tie_helper, {}, str)]
        assert shrink_traced_types(traces, max_typed_dict_size=0) == ({}, Optional[str], None)

    def test_shrink_yield(self) -> None:
        """Test shrinking yield types."""
        traces = [CallTrace(tie_helper, {}, yield_type=int), CallTrace(tie_helper, {}, yield_type=str)]
        assert shrink_traced_types(traces, max_typed_dict_size=0) == ({}, None, Union[int, str])

class Parent:
    """Parent class for testing get_imports_for_annotation."""

    class Child:
        """Child class for testing get_imports_for_annotation."""
        pass

class TestGetImportsForAnnotation:
    """Test the get_imports_for_annotation function."""

    @pytest.mark.parametrize('anno', [inspect.Parameter.empty, inspect.Signature.empty, 'not a type', int])
    def test_no_imports(self, anno: Any) -> None:
        """Test getting no imports for a non-type annotation."""
        assert get_imports_for_annotation(anno) == {}

    @pytest.mark.parametrize('anno, expected', [(Any, {'typing': {'Any'}}), (Union[int, str], {'typing': {'Union'}})])
    def test_special_case_types(self, anno: Any, expected: Dict[str, Set[str]]) -> None:
        """Test getting imports for special case types."""
        assert get_imports_for_annotation(anno) == expected

    def test_callable(self) -> None:
        """Test getting imports for a callable."""
        assert get_imports_for_annotation(Callable) == {'typing': {'Callable'}}

    def test_user_defined_class(self) -> None:
        """Test getting imports for a user-defined class."""
        assert get_imports_for_annotation(Dummy) == {'tests.util': {'Dummy'}}

    @pytest.mark.parametrize('anno, expected', [(Dict[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Dict'}}, (Dict[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Dict'}})), (List[Dummy], {'tests.util': {'Dummy'}, 'typing': {'List'}}, (List[Dummy], {'tests.util': {'Dummy'}, 'typing': {'List'}})), (Set[Dummy], {'tests.util': {'Dummy'}, 'typing': {'Set'}}, (Set[Dummy], {'tests.util': {'Dummy'}, 'typing': {'Set'}})), (Tuple[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Tuple'}}, (Tuple[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Tuple'}})), (Type[Dummy], {'tests.util': {'Dummy'}, 'typing': {'Type'}}, (Type[Dummy], {'tests.util': {'Dummy'}, 'typing': {'Type'}})), (Union[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Union'}}, (Union[str, Dummy], {'tests.util': {'Dummy'}, 'typing': {'Union'}}))])
    def test_container_types(self, anno: Any, expected: Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]) -> None:
        """Test getting imports for container types."""
        assert get_imports_for_annotation(anno) == expected

    def test_nested_class(self) -> None:
        """Test getting imports for a nested class."""
        assert get_imports_for_annotation(Parent.Child) == {Parent.__module__: {'Parent'}}

class TestGetImportsForSignature:
    """Test the get_imports_for_signature function."""

    def test_default_none_parameter_imports(self) -> None:
        """Test getting imports for a default None parameter."""
        stub = FunctionStub('test', inspect.signature(default_none_parameter), FunctionKind.MODULE)
        expected = {'typing': {'Optional'}}
        assert get_imports_for_signature(stub.signature) == expected
