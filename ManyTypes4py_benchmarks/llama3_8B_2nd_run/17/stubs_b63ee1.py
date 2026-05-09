import asyncio
import collections
import enum
import inspect
import logging
import re
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, DefaultDict, Dict, ForwardRef, Iterable, List, Optional, Set, Tuple, Union
from monkeytype.compat import cached_property, is_any, is_forward_ref, is_generic, is_union, make_forward_ref, qualname_of_generic
from monkeytype.tracing import CallTrace, CallTraceLogger
from monkeytype.typing import GenericTypeRewriter, NoneType, NoOpRewriter, TypeRewriter, field_annotations, make_generator, make_iterator, shrink_types
from monkeytype.util import get_name_in_module, pascal_case

class FunctionKind(enum.Enum):
    """Strategies for handling existing annotations in the source."""
    MODULE = 0
    CLASS = 1
    INSTANCE = 2
    STATIC = 3
    PROPERTY = 4
    DJANGO_CACHED_PROPERTY = 5

    @classmethod
    def from_callable(cls, func: Callable) -> 'FunctionKind':
        if '.' not in func.__qualname__:
            return FunctionKind.MODULE
        func_or_desc = get_name_in_module(func.__module__, func.__qualname__, inspect.getattr_static)
        if isinstance(func_or_desc, classmethod):
            return FunctionKind.CLASS
        elif isinstance(func_or_desc, staticmethod):
            return FunctionKind.STATIC
        elif isinstance(func_or_desc, property):
            return FunctionKind.PROPERTY
        elif cached_property and isinstance(func_or_desc, cached_property):
            return FunctionKind.DJANGO_CACHED_PROPERTY
        return FunctionKind.INSTANCE

class ImportMap(DefaultDict[Any, Any]):
    """A mapping of module name to the set of names to be imported."""

    def __init__(self):
        super().__init__(set)

    def merge(self, other: 'ImportMap') -> None:
        for module, names in other.items():
            self[module].update(names)

def _get_import_for_qualname(qualname: str) -> str:
    return qualname.split('.')[0]

def get_imports_for_annotation(anno: Any) -> 'ImportMap':
    """Return the imports (module, name) needed for the type in the annotation"""
    imports: 'ImportMap' = ImportMap()
    if anno is inspect.Parameter.empty or anno is inspect.Signature.empty or (not (isinstance(anno, type) or is_any(anno) or is_union(anno) or is_generic(anno)) or (anno.__module__ == 'builtins')):
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

def get_imports_for_signature(sig: inspect.Signature) -> 'ImportMap':
    """Return the imports (module, name) needed for all types in annotations"""
    imports: 'ImportMap' = ImportMap()
    for param in sig.parameters.values():
        param_imports = get_imports_for_annotation(param.annotation)
        if not _is_optional(param.annotation) and param.default is None:
            imports['typing'].add('Optional')
        imports.merge(param_imports)
    return_imports = get_imports_for_annotation(sig.return_annotation)
    imports.merge(return_imports)
    return imports

def update_signature_args(sig: inspect.Signature, arg_types: Dict[str, Any], has_self: bool, existing_annotation_strategy: 'ExistingAnnotationStrategy') -> inspect.Signature:
    """Update argument annotations with the supplied types"""
    params: List[inspect.Parameter] = []
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

def update_signature_return(sig: inspect.Signature, return_type: Optional[Any], yield_type: Optional[Any], existing_annotation_strategy: 'ExistingAnnotationStrategy') -> inspect.Signature:
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

def shrink_traced_types(traces: Iterable[CallTrace], max_typed_dict_size: int) -> Tuple[Dict[str, Any], Optional[Any], Optional[Any]]:
    """Merges the traced types and returns the minimally equivalent types"""
    arg_types: DefaultDict[str, Set[Any]] = collections.defaultdict(set)
    return_types: Set[Any] = set()
    yield_types: Set[Any] = set()
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
    return shrunken_arg_types, return_type, yield_type

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

    def make_builtin_tuple(self, elements: Iterable[Any]) -> str:
        elems = list(elements)
        return ', '.join(elems) if elems else '()'

    def make_container_type(self, container_type: Any, elements: Iterable[Any]) -> str:
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

def render_signature(sig: inspect.Signature, max_line_len: Optional[int], prefix: str) -> str:
    """Convert a signature into its stub representation.

    NB: This is copied almost entirely from https://github.com/python/cpython/blob/3.6/Lib/inspect.py
    with the modification that it calls our own rendering functions for annotations.

    TODO: push a patch upstream so we don't have to do this on Python 3.x.
    """
    formatted_params = []
    render_pos_only_separator = False
    render_kw_only_separator = True
    for param in sig