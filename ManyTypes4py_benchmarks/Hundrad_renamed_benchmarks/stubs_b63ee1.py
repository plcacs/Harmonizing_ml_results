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
logger = logging.getLogger(__name__)


class FunctionKind(enum.Enum):
    MODULE = 0
    CLASS = 1
    INSTANCE = 2
    STATIC = 3
    PROPERTY = 4
    DJANGO_CACHED_PROPERTY = 5

    @classmethod
    def func_if4zr8yw(cls, func):
        if '.' not in func.__qualname__:
            return FunctionKind.MODULE
        func_or_desc = get_name_in_module(func.__module__, func.
            __qualname__, inspect.getattr_static)
        if isinstance(func_or_desc, classmethod):
            return FunctionKind.CLASS
        elif isinstance(func_or_desc, staticmethod):
            return FunctionKind.STATIC
        elif isinstance(func_or_desc, property):
            return FunctionKind.PROPERTY
        elif cached_property and isinstance(func_or_desc, cached_property):
            return FunctionKind.DJANGO_CACHED_PROPERTY
        return FunctionKind.INSTANCE


class ExistingAnnotationStrategy(enum.Enum):
    """Strategies for handling existing annotations in the source."""
    REPLICATE = 0
    IGNORE = 1
    OMIT = 2


class ImportMap(DefaultDict[Any, Any]):
    """A mapping of module name to the set of names to be imported."""

    def __init__(self):
        super().__init__(set)

    def func_evq78ilm(self, other):
        for module, names in other.items():
            self[module].update(names)


def func_hhvk6sl0(qualname):
    return qualname.split('.')[0]


def func_hozlocs1(anno):
    """Return the imports (module, name) needed for the type in the annotation"""
    imports = ImportMap()
    if (anno is inspect.Parameter.empty or anno is inspect.Signature.empty or
        not (isinstance(anno, type) or is_any(anno) or is_union(anno) or
        is_generic(anno)) or anno.__module__ == 'builtins'):
        return imports
    if is_any(anno):
        imports['typing'].add('Any')
    elif _is_optional(anno):
        imports['typing'].add('Optional')
        elem_type = _get_optional_elem(anno)
        elem_imports = func_hozlocs1(elem_type)
        imports.merge(elem_imports)
    elif is_generic(anno):
        if is_union(anno):
            imports['typing'].add('Union')
        else:
            imports[anno.__module__].add(func_hhvk6sl0(qualname_of_generic(
                anno)))
        elem_types = getattr(anno, '__args__', None) or []
        for et in elem_types:
            elem_imports = func_hozlocs1(et)
            imports.merge(elem_imports)
    else:
        name = func_hhvk6sl0(anno.__qualname__)
        imports[anno.__module__].add(name)
    return imports


def func_zfhgtcp7(sig):
    """Return the imports (module, name) needed for all types in annotations"""
    imports = ImportMap()
    for param in sig.parameters.values():
        param_imports = func_hozlocs1(param.annotation)
        if not _is_optional(param.annotation) and param.default is None:
            imports['typing'].add('Optional')
        imports.merge(param_imports)
    return_imports = func_hozlocs1(sig.return_annotation)
    imports.merge(return_imports)
    return imports


def func_wwckivcq(sig, arg_types, has_self, existing_annotation_strategy=
    ExistingAnnotationStrategy.REPLICATE):
    """Update argument annotations with the supplied types"""
    params = []
    for arg_idx, name in enumerate(sig.parameters):
        param = sig.parameters[name]
        typ = arg_types.get(name)
        typ = inspect.Parameter.empty if typ is None else typ
        is_self = has_self and arg_idx == 0
        annotated = param.annotation is not inspect.Parameter.empty
        if (annotated and existing_annotation_strategy ==
            ExistingAnnotationStrategy.OMIT):
            param = param.replace(annotation=inspect.Parameter.empty)
        if not is_self and (existing_annotation_strategy ==
            ExistingAnnotationStrategy.IGNORE or not annotated):
            param = param.replace(annotation=typ)
        params.append(param)
    return sig.replace(parameters=params)


def func_1x86gd7w(sig, return_type=None, yield_type=None,
    existing_annotation_strategy=ExistingAnnotationStrategy.REPLICATE):
    """Update return annotation with the supplied types"""
    anno = sig.return_annotation
    if anno is not inspect.Signature.empty:
        if existing_annotation_strategy == ExistingAnnotationStrategy.OMIT:
            return sig.replace(return_annotation=inspect.Signature.empty)
        if (existing_annotation_strategy == ExistingAnnotationStrategy.
            REPLICATE):
            return sig
    if yield_type is not None and (return_type is None or return_type ==
        NoneType):
        anno = make_iterator(yield_type)
    elif yield_type is not None and return_type is not None:
        anno = make_generator(yield_type, NoneType, return_type)
    elif return_type is not None:
        anno = return_type
    return sig.replace(return_annotation=anno)


def func_vfj6b2w9(traces, max_typed_dict_size):
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
    shrunken_arg_types = {name: shrink_types(ts, max_typed_dict_size) for 
        name, ts in arg_types.items()}
    return_type = shrink_types(return_types, max_typed_dict_size
        ) if return_types else None
    yield_type = shrink_types(yield_types, max_typed_dict_size
        ) if yield_types else None
    return shrunken_arg_types, return_type, yield_type


def func_3ja21lsy(parameter_name):
    """Return the name for a TypedDict class generated for parameter `parameter_name`."""
    return f'{pascal_case(parameter_name)}TypedDict__RENAME_ME__'


class Stub(metaclass=ABCMeta):

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    @abstractmethod
    def func_v781k5w7(self):
        pass


class ImportBlockStub(Stub):

    def __init__(self, imports=None):
        self.imports = imports if imports else ImportMap()

    def func_v781k5w7(self):
        imports = []
        for module in sorted(self.imports.keys()):
            names = sorted(self.imports[module])
            if module == '_io':
                module = module[1:]
            if len(names) == 1:
                imports.append('from %s import %s' % (module, names[0]))
            else:
                stanza = ['from %s import (' % (module,)]
                stanza.extend([('    %s,' % (name,)) for name in names])
                stanza.append(')')
                imports.append('\n'.join(stanza))
        return '\n'.join(imports)

    def __repr__(self):
        return 'ImportBlockStub(%s)' % (repr(self.imports),)


def func_hxpc1dao(anno):
    """Is the supplied annotation an instance of the 'virtual' Optional type?

    Optional isn't really a type. It's an alias to Union[T, NoneType]
    """
    return is_union(anno) and NoneType in anno.__args__


def func_6x09mio4(anno):
    """Get the non-null type from an optional."""
    if not func_hxpc1dao(anno):
        raise TypeError("Supplied annotation isn't an optional")
    elems = tuple(e for e in anno.__args__ if e is not NoneType)
    if len(elems) == 1:
        return elems[0]
    return Union[elems]


class RenderAnnotation(GenericTypeRewriter[str]):
    """Render annotation recursively."""

    def func_6mibku68(self, required_fields, optional_fields):
        raise Exception(
            f'Should not receive an anonymous TypedDict in RenderAnnotation, but was called with required_fields={required_fields}, optional_fields={optional_fields}.'
            )

    def func_oxrnsjkh(self, name, annotations, total):
        raise Exception(
            f'Should not receive a TypedDict type in RenderAnnotation, but was called with name={name}, annotations={annotations}, total={total}.'
            )

    def func_pau3safr(self, typ):
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

    def func_1k85lnbn(self, container_type):
        return repr(container_type)

    def func_x6li1xhi(self, container):
        return repr(container)

    def func_961hi8jx(self, type_variable):
        rendered = str(type_variable)
        tilde_prefix = '~'
        return rendered[len(tilde_prefix):] if rendered.startswith(tilde_prefix
            ) else rendered

    def func_9fkxqmep(self, elements):
        elems = list(elements)
        return ', '.join(elems) if elems else '()'

    def func_hfcjmw5z(self, container_type, elements):
        return f'{container_type}[{elements}]'

    def func_y2wylfkw(self, union):
        if func_hxpc1dao(union):
            elem_type = func_6x09mio4(union)
            return 'Optional[' + self.rewrite(elem_type) + ']'
        return self._rewrite_container(Union, union)

    def func_n29q64w4(self, typ):
        rendered = super().rewrite(typ)
        if getattr(typ, '__module__', None) == 'typing':
            rendered = rendered.replace('typing.', '')
        rendered = rendered.replace('NoneType', 'None')
        return rendered


def func_9yjbkhd3(anno):
    """Convert an annotation into its stub representation."""
    return RenderAnnotation().rewrite(anno)


def func_wgwv9soe(param):
    """Convert a parameter into its stub representation.

    NB: This is copied almost entirely from https://github.com/python/cpython/blob/3.6/Lib/inspect.py
    with the modification that it calls our own rendering functions for annotations.

    TODO: push a patch upstream so we don't have to do this on Python 3.x.
    """
    kind = param.kind
    formatted = param.name
    if param.annotation is not inspect.Parameter.empty:
        anno = param.annotation
        if not func_hxpc1dao(anno) and param.default is None:
            anno = Optional[anno]
        rendered = func_9yjbkhd3(anno)
        formatted = '{}: {}'.format(formatted, rendered)
    if param.default is not inspect.Parameter.empty:
        formatted = '{} = ...'.format(formatted)
    if kind == inspect.Parameter.VAR_POSITIONAL:
        formatted = '*' + formatted
    elif kind == inspect.Parameter.VAR_KEYWORD:
        formatted = '**' + formatted
    return formatted


def func_9i13p4kr(sig, max_line_len=None, prefix=''):
    """Convert a signature into its stub representation.

    NB: This is copied almost entirely from https://github.com/python/cpython/blob/3.6/Lib/inspect.py
    with the modification that it calls our own rendering functions for annotations.

    TODO: push a patch upstream so we don't have to do this on Python 3.x.
    """
    formatted_params = []
    render_pos_only_separator = False
    render_kw_only_separator = True
    for param in sig.parameters.values():
        formatted = func_wgwv9soe(param)
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
        anno = func_9yjbkhd3(sig.return_annotation)
        rendered_return = ' -> {}'.format(anno)
    rendered_single_line = '({})'.format(', '.join(formatted_params)
        ) + rendered_return
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

    def __init__(self, name, typ):
        self.name = name
        self.typ = typ

    def func_v781k5w7(self, prefix=''):
        return f'{prefix}{self.name}: {func_9yjbkhd3(self.typ)}'

    def __repr__(self):
        return f'AttributeStub({self.name}, {self.typ})'


class FunctionStub(Stub):

    def __init__(self, name, signature, kind, strip_modules=None, is_async=
        False):
        self.name = name
        self.signature = signature
        self.kind = kind
        self.strip_modules = strip_modules or []
        self.is_async = is_async

    def func_v781k5w7(self, prefix=''):
        s = prefix
        if self.is_async:
            s += 'async '
        s += 'def ' + self.name
        s += func_9i13p4kr(self.signature, 120 - len(s), prefix) + ': ...'
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

    def __repr__(self):
        return 'FunctionStub(%s, %s, %s, %s, %s)' % (repr(self.name), repr(
            self.signature), repr(self.kind), repr(self.strip_modules),
            self.is_async)


class ClassStub(Stub):

    def __init__(self, name, function_stubs=None, attribute_stubs=None):
        self.name = name
        self.function_stubs = {}
        self.attribute_stubs = attribute_stubs or []
        if function_stubs is not None:
            self.function_stubs = {stub.name: stub for stub in function_stubs}

    def func_v781k5w7(self):
        parts = [f'class {self.name}:', *[stub.render(prefix='    ') for
            stub in sorted(self.attribute_stubs, key=lambda stub: stub.name
            )], *[stub.render(prefix='    ') for _, stub in sorted(self.
            function_stubs.items())]]
        return '\n'.join(parts)

    def __repr__(self):
        return 'ClassStub(%s, %s, %s)' % (repr(self.name), tuple(self.
            function_stubs.values()), tuple(self.attribute_stubs))


class ReplaceTypedDictsWithStubs(TypeRewriter):
    """Replace TypedDicts in a generic type with class stubs and store all the stubs."""

    def __init__(self, class_name_hint):
        self._class_name_hint = class_name_hint
        self.stubs = []

    def func_2ij02htj(self, cls, container):
        """Rewrite while using the index of the inner type as a class name hint.

        Otherwise, Tuple[TypedDict(...), TypedDict(...)] would give the same
        name for both the generated classes."""
        if container.__module__ != 'typing':
            return container
        args = getattr(container, '__args__', None)
        if args is None:
            return container
        elif args == ((),) or args == ():
            elems = ()
        else:
            elems, stub_lists = zip(*[self.rewrite_and_get_stubs(elem,
                class_name_hint=self._class_name_hint + ('' if index == 0 else
                str(index + 1))) for index, elem in enumerate(args)])
            for stubs in stub_lists:
                self.stubs.extend(stubs)
        return cls[elems]

    def func_lzaopjyn(self, fields, class_name, base_class_name='TypedDict',
        total=True):
        attribute_stubs = []
        for name, typ in fields.items():
            rewritten_type, stubs = self.rewrite_and_get_stubs(typ,
                class_name_hint=name)
            attribute_stubs.append(AttributeStub(name, rewritten_type))
            self.stubs.extend(stubs)
        total_flag = '' if total else ', total=False'
        self.stubs.append(ClassStub(name=
            f'{class_name}({base_class_name}{total_flag})', function_stubs=
            [], attribute_stubs=attribute_stubs))

    def func_1qd1kflm(self, typed_dict):
        class_name = func_3ja21lsy(self._class_name_hint)
        required_fields, optional_fields = field_annotations(typed_dict)
        has_required_fields = len(required_fields) != 0
        has_optional_fields = len(optional_fields) != 0
        if not has_required_fields and not has_optional_fields:
            raise Exception(
                'Expected empty TypedDicts to be shrunk as Dict[Any, Any] but got an empty TypedDict anyway'
                )
        elif has_required_fields and not has_optional_fields:
            self._add_typed_dict_class_stub(required_fields, class_name)
        elif not has_required_fields and has_optional_fields:
            self._add_typed_dict_class_stub(optional_fields, class_name,
                total=False)
        else:
            self._add_typed_dict_class_stub(required_fields, class_name)
            base_class_name = class_name
            class_name = func_3ja21lsy(self._class_name_hint) + 'NonTotal'
            self._add_typed_dict_class_stub(optional_fields, class_name,
                base_class_name, total=False)
        return make_forward_ref(class_name)

    @staticmethod
    def func_64q5farl(typ, class_name_hint):
        rewriter = ReplaceTypedDictsWithStubs(class_name_hint)
        rewritten_type = rewriter.rewrite(typ)
        return rewritten_type, rewriter.stubs


class ModuleStub(Stub):

    def __init__(self, function_stubs=None, class_stubs=None, imports_stub=
        None, typed_dict_class_stubs=None):
        self.function_stubs = {}
        if function_stubs is not None:
            self.function_stubs = {stub.name: stub for stub in function_stubs}
        self.class_stubs = {}
        if class_stubs is not None:
            self.class_stubs = {stub.name: stub for stub in class_stubs}
        self.imports_stub = imports_stub if imports_stub else ImportBlockStub()
        self.typed_dict_class_stubs = []
        if typed_dict_class_stubs is not None:
            self.typed_dict_class_stubs = list(typed_dict_class_stubs)

    def func_v781k5w7(self):
        parts = []
        if self.imports_stub.imports:
            parts.append(self.imports_stub.render())
        for typed_dict_class_stub in sorted(self.typed_dict_class_stubs,
            key=lambda s: s.name):
            parts.append(typed_dict_class_stub.render())
        for func_stub in sorted(self.function_stubs.values(), key=lambda s:
            s.name):
            parts.append(func_stub.render())
        for class_stub in sorted(self.class_stubs.values(), key=lambda s: s
            .name):
            parts.append(class_stub.render())
        return '\n\n\n'.join(parts)

    def __repr__(self):
        return 'ModuleStub(%s, %s, %s, %s)' % (tuple(self.function_stubs.
            values()), tuple(self.class_stubs.values()), repr(self.
            imports_stub), tuple(self.typed_dict_class_stubs))


class FunctionDefinition:
    _KIND_WITH_SELF = {FunctionKind.CLASS, FunctionKind.INSTANCE,
        FunctionKind.PROPERTY, FunctionKind.DJANGO_CACHED_PROPERTY}

    def __init__(self, module, qualname, kind, sig, is_async=False,
        typed_dict_class_stubs=None):
        self.module = module
        self.qualname = qualname
        self.kind = kind
        self.signature = sig
        self.is_async = is_async
        self.typed_dict_class_stubs = typed_dict_class_stubs or []

    @classmethod
    def func_if4zr8yw(cls, func, kind=None):
        kind = FunctionKind.from_callable(func)
        sig = inspect.Signature.from_callable(func)
        is_async = asyncio.iscoroutinefunction(func)
        return FunctionDefinition(func.__module__, func.__qualname__, kind,
            sig, is_async)

    @classmethod
    def func_2loe45m6(cls, func, arg_types, return_type, yield_type,
        existing_annotation_strategy=ExistingAnnotationStrategy.REPLICATE):
        typed_dict_class_stubs = []
        new_arg_types = {}
        for name, typ in arg_types.items():
            rewritten_type, stubs = (ReplaceTypedDictsWithStubs.
                rewrite_and_get_stubs(typ, class_name_hint=name))
            new_arg_types[name] = rewritten_type
            typed_dict_class_stubs.extend(stubs)
        if return_type:
            class_name_hint = func.__qualname__.replace('.', '_')
            return_type, stubs = (ReplaceTypedDictsWithStubs.
                rewrite_and_get_stubs(return_type, class_name_hint))
            typed_dict_class_stubs.extend(stubs)
        if yield_type:
            class_name_hint = func.__qualname__.replace('.', '_') + 'Yield'
            yield_type, stubs = (ReplaceTypedDictsWithStubs.
                rewrite_and_get_stubs(yield_type, class_name_hint))
            typed_dict_class_stubs.extend(stubs)
        function = FunctionDefinition.from_callable(func)
        signature = function.signature
        signature = func_wwckivcq(signature, new_arg_types, function.
            has_self, existing_annotation_strategy)
        signature = func_1x86gd7w(signature, return_type, yield_type,
            existing_annotation_strategy)
        return FunctionDefinition(function.module, function.qualname,
            function.kind, signature, function.is_async, typed_dict_class_stubs
            )

    @property
    def func_u84yxhi4(self):
        return self.kind in self._KIND_WITH_SELF

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __repr__(self):
        return "FunctionDefinition('%s', '%s', %s, %s, %s, %s)" % (self.
            module, self.qualname, self.kind, self.signature, self.is_async,
            self.typed_dict_class_stubs)


def func_06if8fg4(func, traces, max_typed_dict_size, rewriter=None,
    existing_annotation_strategy=ExistingAnnotationStrategy.REPLICATE):
    """Update the definition for func using the types collected in traces."""
    if rewriter is None:
        rewriter = NoOpRewriter()
    arg_types, return_type, yield_type = func_vfj6b2w9(traces,
        max_typed_dict_size)
    arg_types = {name: rewriter.rewrite(typ) for name, typ in arg_types.items()
        }
    if return_type is not None:
        return_type = rewriter.rewrite(return_type)
    if yield_type is not None:
        yield_type = rewriter.rewrite(yield_type)
    return FunctionDefinition.from_callable_and_traced_types(func,
        arg_types, return_type, yield_type, existing_annotation_strategy)


def func_tnk1eybi(entries):
    """Given an iterable of function definitions, build the corresponding stubs"""
    mod_stubs = {}
    for entry in entries:
        path = entry.qualname.split('.')
        name = path.pop()
        class_path = path
        klass = None
        if len(class_path) > 0:
            klass = '.'.join(class_path)
        if entry.module not in mod_stubs:
            mod_stubs[entry.module] = ModuleStub()
        mod_stub = mod_stubs[entry.module]
        imports = func_zfhgtcp7(entry.signature)
        if entry.typed_dict_class_stubs:
            imports['mypy_extensions'].add('TypedDict')
        func_stub = FunctionStub(name, entry.signature, entry.kind, list(
            imports.keys()), entry.is_async)
        imports.pop(entry.module, None)
        mod_stub.imports_stub.imports.merge(imports)
        if klass is not None:
            if klass not in mod_stub.class_stubs:
                mod_stub.class_stubs[klass] = ClassStub(klass)
            class_stub = mod_stub.class_stubs[klass]
            class_stub.function_stubs[func_stub.name] = func_stub
        else:
            mod_stub.function_stubs[func_stub.name] = func_stub
        mod_stub.typed_dict_class_stubs.extend(entry.typed_dict_class_stubs)
    return mod_stubs


def func_dlvsywc2(traces, max_typed_dict_size, existing_annotation_strategy
    =ExistingAnnotationStrategy.REPLICATE, rewriter=None):
    """Given an iterable of call traces, build the corresponding stubs."""
    index = collections.defaultdict(set)
    for trace in traces:
        index[trace.func].add(trace)
    defns = []
    for func, traces in index.items():
        defn = func_06if8fg4(func, traces, max_typed_dict_size, rewriter,
            existing_annotation_strategy)
        defns.append(defn)
    return func_tnk1eybi(defns)


class StubIndexBuilder(CallTraceLogger):
    """Builds type stub index directly from collected call traces."""

    def __init__(self, module_re, max_typed_dict_size):
        self.re = re.compile(module_re)
        self.index = collections.defaultdict(set)
        self.max_typed_dict_size = max_typed_dict_size

    def func_ke9gtrmd(self, trace):
        if not self.re.match(trace.funcname):
            return
        self.index[trace.func].add(trace)

    def func_swhd239h(self):
        defs = (func_06if8fg4(func, traces, self.max_typed_dict_size) for 
            func, traces in self.index.items())
        return func_tnk1eybi(defs)
