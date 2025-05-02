import asyncio
import collections
import enum
import inspect
import logging
import re
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    ForwardRef,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

from monkeytype.compat import cached_property, is_any, is_forward_ref, is_generic, is_union, make_forward_ref, qualname_of_generic
from monkeytype.tracing import CallTrace, CallTraceLogger
from monkeytype.typing import GenericTypeRewriter, NoneType, NoOpRewriter, TypeRewriter, field_annotations, make_generator, make_iterator, shrink_types
from monkeytype.util import get_name_in_module, pascal_case

logger = logging.getLogger(__name__)

T = TypeVar('T')
FunctionKindT = TypeVar('FunctionKindT', bound='FunctionKind')
ExistingAnnotationStrategyT = TypeVar('ExistingAnnotationStrategyT', bound='ExistingAnnotationStrategy')
StubT = TypeVar('StubT', bound='Stub')
FunctionDefinitionT = TypeVar('FunctionDefinitionT', bound='FunctionDefinition')

class FunctionKind(enum.Enum):
    MODULE = 0
    CLASS = 1
    INSTANCE = 2
    STATIC = 3
    PROPERTY = 4
    DJANGO_CACHED_PROPERTY = 5

    @classmethod
    def from_callable(cls: Type[FunctionKindT], func: Callable[..., Any]) -> FunctionKindT:
        if '.' not in func.__qualname__:
            return cast(FunctionKindT, FunctionKind.MODULE)
        func_or_desc = get_name_in_module(func.__module__, func.__qualname__, inspect.getattr_static)
        if isinstance(func_or_desc, classmethod):
            return cast(FunctionKindT, FunctionKind.CLASS)
        elif isinstance(func_or_desc, staticmethod):
            return cast(FunctionKindT, FunctionKind.STATIC)
        elif isinstance(func_or_desc, property):
            return cast(FunctionKindT, FunctionKind.PROPERTY)
        elif cached_property and isinstance(func_or_desc, cached_property):
            return cast(FunctionKindT, FunctionKind.DJANGO_CACHED_PROPERTY)
        return cast(FunctionKindT, FunctionKind.INSTANCE)

class ExistingAnnotationStrategy(enum.Enum):
    """Strategies for handling existing annotations in the source."""
    REPLICATE = 0
    IGNORE = 1
    OMIT = 2

class ImportMap(DefaultDict[str, Set[str]]):
    """A mapping of module name to the set of names to be imported."""

    def __init__(self) -> None:
        super().__init__(set)

    def merge(self, other: 'ImportMap') -> None:
        for module, names in other.items():
            self[module].update(names)

def _get_import_for_qualname(qualname: str) -> str:
    return qualname.split('.')[0]

def get_imports_for_annotation(anno: Any) -> ImportMap:
    """Return the imports (module, name) needed for the type in the annotation"""
    imports = ImportMap()
    if anno is inspect.Parameter.empty or anno is inspect.Signature.empty or (not (isinstance(anno, type) or is_any(anno) or is_union(anno) or is_generic(anno))) or (anno.__module__ == 'builtins'):
        return imports
    if is_any(anno):
        imports['typing'].add('Any')
    elif _is_optional(anno):
        imports['typing'].add('Optional')
        elem_type = _get_optional_elem(anno)
        elem_imports = get_imports_for_annotation(elem_type)
        imports.merge(elem_imports)
    elif is_generic(anno):
        if is_union(anno):
            imports['typing'].add('Union')
        else:
            imports[anno.__module__].add(_get_import_for_qualname(qualname_of_generic(anno)))
        elem_types = getattr(anno, '__args__', None) or []
        for et in elem_types:
            elem_imports = get_imports_for_annotation(et)
            imports.merge(elem_imports)
    else:
        name = _get_import_for_qualname(anno.__qualname__)
        imports[anno.__module__].add(name)
    return imports

def get_imports_for_signature(sig: inspect.Signature) -> ImportMap:
    """Return the imports (module, name) needed for all types in annotations"""
    imports = ImportMap()
    for param in sig.parameters.values():
        param_imports = get_imports_for_annotation(param.annotation)
        if not _is_optional(param.annotation) and param.default is None:
            imports['typing'].add('Optional')
        imports.merge(param_imports)
    return_imports = get_imports_for_annotation(sig.return_annotation)
    imports.merge(return_imports)
    return imports

def update_signature_args(
    sig: inspect.Signature,
    arg_types: Dict[str, Any],
    has_self: bool,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE
) -> inspect.Signature:
    """Update argument annotations with the supplied types"""
    params = []
    for arg_idx, name in enumerate(sig.parameters):
        param = sig.parameters[name]
        typ = arg_types.get(name)
        typ = inspect.Parameter.empty if typ is None else typ
        is_self = has_self and arg_idx == 0
        annotated = param.annotation is not inspect.Parameter.empty
        if annotated and existing_annotation_strategy == ExistingAnnotationStrategy.OMIT:
            param = param.replace(annotation=inspect.Parameter.empty)
        if not is_self and (existing_annotation_strategy == ExistingAnnotationStrategy.IGNORE or not annotated):
            param = param.replace(annotation=typ)
        params.append(param)
    return sig.replace(parameters=params)

def update_signature_return(
    sig: inspect.Signature,
    return_type: Optional[Any] = None,
    yield_type: Optional[Any] = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE
) -> inspect.Signature:
    """Update return annotation with the supplied types"""
    anno = sig.return_annotation
    if anno is not inspect.Signature.empty:
        if existing_annotation_strategy == ExistingAnnotationStrategy.OMIT:
            return sig.replace(return_annotation=inspect.Signature.empty)
        if existing_annotation_strategy == ExistingAnnotationStrategy.REPLICATE:
            return sig
    if yield_type is not None and (return_type is None or return_type == NoneType):
        anno = make_iterator(yield_type)
    elif yield_type is not None and return_type is not None:
        anno = make_generator(yield_type, NoneType, return_type)
    elif return_type is not None:
        anno = return_type
    return sig.replace(return_annotation=anno)

def shrink_traced_types(
    traces: Iterable[CallTrace],
    max_typed_dict_size: int
) -> Tuple[Dict[str, Any], Optional[Any], Optional[Any]]:
    """Merges the traced types and returns the minimally equivalent types"""
    arg_types = collections.defaultdict(set)
    return_types = set()
    yield_types = set()
    for t in traces:
        for arg, typ in t.arg_types.items():
            arg_types[arg].add(typ)
        if t.return_type is not None:
            return_types.add(t.return_type)
        if t.yield_type is not None:
            yield_types.add(t.yield_type)
    shrunken_arg_types = {name: shrink_types(ts, max_typed_dict_size) for name, ts in arg_types.items()}
    return_type = shrink_types(return_types, max_typed_dict_size) if return_types else None
    yield_type = shrink_types(yield_types, max_typed_dict_size) if yield_types else None
    return (shrunken_arg_types, return_type, yield_type)

def get_typed_dict_class_name(parameter_name: str) -> str:
    """Return the name for a TypedDict class generated for parameter `parameter_name`."""
    return f'{pascal_case(parameter_name)}TypedDict__RENAME_ME__'

class Stub(metaclass=ABCMeta):
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    @abstractmethod
    def render(self) -> str:
        pass

class ImportBlockStub(Stub):
    def __init__(self, imports: Optional[ImportMap] = None) -> None:
        self.imports = imports if imports else ImportMap()

    def render(self) -> str:
        imports = []
        for module in sorted(self.imports.keys()):
            names = sorted(self.imports[module])
            if module == '_io':
                module = module[1:]
            if len(names) == 1:
                imports.append('from %s import %s' % (module, names[0]))
            else:
                stanza = ['from %s import (' % (module,)]
                stanza.extend(['    %s,' % (name,) for name in names])
                stanza.append(')')
                imports.append('\n'.join(stanza))
        return '\n'.join(imports)

    def __repr__(self) -> str:
        return 'ImportBlockStub(%s)' % (repr(self.imports),)

def _is_optional(anno: Any) -> bool:
    """Is the supplied annotation an instance of the 'virtual' Optional type?

    Optional isn't really a type. It's an alias to Union[T, NoneType]
    """
    return is_union(anno) and NoneType in anno.__args__

def _get_optional_elem(anno: Any) -> Any:
    """Get the non-null type from an optional."""
    if not _is_optional(anno):
        raise TypeError("Supplied annotation isn't an optional")
    elems = tuple((e for e in anno.__args__ if e is not NoneType))
    if len(elems) == 1:
        return elems[0]
    return Union[elems]

class RenderAnnotation(GenericTypeRewriter[str]):
    """Render annotation recursively."""

    def make_anonymous_typed_dict(self, required_fields: Dict[str, Any], optional_fields: Dict[str, Any]) -> str:
        raise Exception(f'Should not receive an anonymous TypedDict in RenderAnnotation, but was called with required_fields={required_fields}, optional_fields={optional_fields}.')

    def make_builtin_typed_dict(self, name: str, annotations: Dict[str, Any], total: bool) -> str:
        raise Exception(f'Should not receive a TypedDict type in RenderAnnotation, but was called with name={name}, annotations={annotations}, total={total}.')

    def generic_rewrite(self, typ: Any) -> str:
        if hasattr(typ, '__supertype__'):
            rendered = str(typ.__name__)
        elif is_forward_ref(typ):
            rendered = repr(typ.__forward_arg__)
        elif isinstance(typ, NoneType) or typ is NoneType:
            rendered = 'None'
        elif is_generic(typ):
            rendered = repr(typ)
        elif isinstance(typ, type):
            if typ.__module__ in ('builtins',):
                rendered = typ.__qualname__
            else:
                rendered = typ.__module__ + '.' + typ.__qualname__
        elif isinstance(typ, str):
            rendered = typ
        else:
            rendered = repr(typ)
        return rendered

    def rewrite_container_type(self, container_type: Any) -> str:
        return repr(container_type)

    def rewrite_malformed_container(self, container: Any) -> str:
        return repr(container)

    def rewrite_type_variable(self, type_variable: Any) -> str:
        rendered = str(type_variable)
        tilde_prefix = '~'
        return rendered[len(tilde_prefix):] if rendered.startswith(tilde_prefix) else rendered

    def make_builtin_tuple(self, elements: List[str]) -> str:
        elems = list(elements)
        return ', '.join(elems) if elems else '()'

    def make_container_type(self, container_type: str, elements: str) -> str:
        return f'{container_type}[{elements}]'

    def rewrite_Union(self, union: Any) -> str:
        if _is_optional(union):
            elem_type = _get_optional_elem(union)
            return 'Optional[' + self.rewrite(elem_type) + ']'
        return self._rewrite_container(Union, union)

    def rewrite(self, typ: Any) -> str:
        rendered = super().rewrite(typ)
        if getattr(typ, '__module__', None) == 'typing':
            rendered = rendered.replace('typing.', '')
        rendered = rendered.replace('NoneType', 'None')
        return rendered

def render_annotation(anno: Any) -> str:
    """Convert an annotation into its stub representation."""
    return RenderAnnotation().rewrite(anno)

def render_parameter(param: inspect.Parameter) -> str:
    """Convert a parameter into its stub representation.

    NB: This is copied almost entirely from https://github.com/python/cpython/blob/3.6/Lib/inspect.py
    with the modification that it calls our own rendering functions for annotations.

    TODO: push a patch upstream so we don't have to do this on Python 3.x.
    """
    kind = param.kind
    formatted = param.name
    if param.annotation is not inspect.Parameter.empty:
        anno = param.annotation
        if not _is_optional(anno) and param.default is None:
            anno = Optional[anno]
        rendered = render_annotation(anno)
        formatted = '{}: {}'.format(formatted, rendered)
    if param.default is not inspect.Parameter.empty:
        formatted = '{} = ...'.format(formatted)
    if kind == inspect.Parameter.VAR_POSITIONAL:
        formatted = '*' + formatted
    elif kind == inspect.Parameter.VAR_KEYWORD:
        formatted = '**' + formatted
    return formatted

def render_signature(sig: inspect.Signature, max_line_len: Optional[int] = None, prefix: str = '') -> str:
    """Convert a signature into its stub representation.

    NB: This is copied almost entirely from https://github.com/python/cpython/blob/3.6/Lib/inspect.py
    with the modification that it calls our own rendering functions for annotations.

    TODO: push a patch upstream so we don't have to do this on Python 3.x.
    """
    formatted_params = []
    render_pos_only_separator = False
    render_kw_only_separator = True
    for param in sig.parameters.values():
        formatted = render_parameter(param)
        kind = param.kind
        if kind == inspect.Parameter.POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            formatted_params.append('/')
            render_pos_only_separator = False
        if kind == inspect.Parameter.VAR_POSITIONAL:
            render_kw_only_separator = False
        elif kind == inspect.Parameter.KEYWORD_ONLY and render_kw_only_separator:
            formatted_params.append('*')
            render_kw_only_separator = False
        formatted_params.append(formatted)
    if render_pos_only_separator:
        formatted_params.append('/')
    rendered_return = ''
    if sig.return_annotation is not inspect.Signature.empty:
        anno = render_annotation(sig.return_annotation)
        rendered_return = ' -> {}'.format(anno)
    rendered_single_line = '({})'.format(', '.join(formatted_params)) + rendered_return
    if max_line_len is None or len(rendered_single_line) <= max_line_len:
        return rendered_single_line
    rendered_multi_lines = ['(']
    for i, f_param in enumerate(formatted_params):
        line = '    ' + f_param
        if i != len(formatted_params) - 1:
            line += ','
        rendered_multi_lines.append(prefix + line)
    rendered_multi_lines.append(prefix + ')' + rendered_return)
    return '\n'.join(rendered_multi_lines)

class AttributeStub(Stub):
    def __init__(self, name: str, typ: Any) -> None:
        self.name = name
        self.typ = typ

    def render(self, prefix: str = '') -> str:
        return f'{prefix}{self.name}: {render_annotation(self.typ)}'

    def __repr__(self) -> str:
        return f'AttributeStub({self.name}, {self.typ})'

class FunctionStub(Stub):
    def __init__(
        self,
        name: str,
        signature: inspect.Signature,
        kind: FunctionKind,
        strip_modules: Optional[List[str]] = None,
        is_async: bool = False
    ) -> None:
        self.name = name
        self.signature = signature
        self.kind = kind
        self.strip_modules = strip_modules or []
        self.is_async = is_async

    def render(self, prefix: str = '') -> str:
        s = prefix
        if self.is_async:
            s += 'async '
        s += 'def ' + self.name
        s += render_signature(self.signature, 120 - len(s), prefix) + ': ...'
        for module in self.strip_modules:
            s = s.replace(module + '.', '')
        if self.kind == FunctionKind.CLASS:
            s = prefix + '@classmethod\n' + s
        elif self.kind == FunctionKind.STATIC:
            s = prefix + '@staticmethod\n' + s
        elif self.kind == FunctionKind.PROPERTY:
            s = prefix + '@property\n' + s
        elif self.kind == FunctionKind.DJANGO_CACHED_PROPERTY:
            s = prefix + '@cached_property\n' + s
        return s

    def __repr__(self) -> str:
        return 'FunctionStub(%s,