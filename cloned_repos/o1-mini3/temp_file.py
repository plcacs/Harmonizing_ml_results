"""
Imitate the parser representation.
"""
import re
from functools import partial, wraps
from inspect import Parameter
from pathlib import Path
from typing import Optional, Callable, Any, Tuple, List, Union, Generator, Dict
from jedi import debug
from jedi.inference.utils import to_list
from jedi.cache import memoize_method
from jedi.inference.filters import AbstractFilter
from jedi.inference.names import AbstractNameDefinition, ValueNameMixin, ParamNameInterface
from jedi.inference.base_value import Value, ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValue
from jedi.inference.compiled.access import _sentinel
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.helpers import reraise_getitem_errors
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext
from jedi.inference.compiled import create_from_access_path, create_cached_compiled_value, create_from_name


class CheckAttribute:
    """Raises :exc:`AttributeError` if the attribute X is not available."""

    def __init__(self, check_name: Optional[str]=None) ->None:
        self.check_name: Optional[str] = check_name

    def __call__(self, func: Callable[..., Any]) ->'CheckAttribute':
        self.func: Callable[..., Any] = func
        if self.check_name is None:
            self.check_name = func.__name__[2:]
        return self

    def __get__(self, instance: Optional[Any], owner: type) ->Optional[Callable
        [..., Any]]:
        if instance is None:
            return self
        instance.access_handle.getattr_paths(self.check_name)
        return partial(self.func, instance)


class CompiledValue(Value):
    access_handle: Any

    def __init__(self, inference_state: Any, access_handle: Any,
        parent_context: Optional['CompiledContext']=None) ->None:
        super().__init__(inference_state, parent_context)
        self.access_handle = access_handle

    def py__call__(self, arguments: Any) ->ValueSet:
        return_annotation = self.access_handle.get_return_annotation()
        if return_annotation is not None:
            return create_from_access_path(self.inference_state,
                return_annotation).execute_annotation()
        try:
            self.access_handle.getattr_paths('__call__')
        except AttributeError:
            return super().py__call__(arguments)
        else:
            if self.access_handle.is_class():
                from jedi.inference.value import CompiledInstance
                return ValueSet([CompiledInstance(self.inference_state,
                    self.parent_context, self, arguments)])
            else:
                return ValueSet(self._execute_function(arguments))

    @CheckAttribute()
    def py__class__(self) ->Any:
        return create_from_access_path(self.inference_state, self.
            access_handle.py__class__())

    @CheckAttribute()
    def py__mro__(self) ->Any:
        return (self,) + tuple(create_from_access_path(self.inference_state,
            access) for access in self.access_handle.py__mro__accesses())

    @CheckAttribute()
    def py__bases__(self) ->Tuple[Any, ...]:
        return tuple(create_from_access_path(self.inference_state, access) for
            access in self.access_handle.py__bases__())

    def get_qualified_names(self) ->Optional[Tuple[str, ...]]:
        return self.access_handle.get_qualified_names()

    def py__bool__(self) ->bool:
        return self.access_handle.py__bool__()

    def is_class(self) ->bool:
        return self.access_handle.is_class()

    def is_function(self) ->bool:
        return self.access_handle.is_function()

    def is_module(self) ->bool:
        return self.access_handle.is_module()

    def is_compiled(self) ->bool:
        return True

    def is_stub(self) ->bool:
        return False

    def is_instance(self) ->bool:
        return self.access_handle.is_instance()

    def py__doc__(self) ->Optional[str]:
        return self.access_handle.py__doc__()

    @to_list
    def get_param_names(self) ->Generator['ParamNameInterface', None, None]:
        try:
            signature_params = self.access_handle.get_signature_params()
        except ValueError:
            params_str, ret = self._parse_function_doc()
            if not params_str:
                tokens: List[str] = []
            else:
                tokens = params_str.split(',')
            if self.access_handle.ismethoddescriptor():
                tokens.insert(0, 'self')
            for p in tokens:
                name, _, default = p.strip().partition('=')
                yield UnresolvableParamName(self, name, default)
        else:
            for signature_param in signature_params:
                yield SignatureParamName(self, signature_param)

    def get_signatures(self) ->List[BuiltinSignature]:
        _, return_string = self._parse_function_doc()
        return [BuiltinSignature(self, return_string)]

    def __repr__(self) ->str:
        return '<%s: %s>' % (self.__class__.__name__, self.access_handle.
            get_repr())

    @memoize_method
    def _parse_function_doc(self) ->Tuple[str, str]:
        doc = self.py__doc__()
        if doc is None:
            return '', ''
        return _parse_function_doc(doc)

    @property
    def api_type(self) ->str:
        return self.access_handle.get_api_type()

    def get_filters(self, is_instance: bool=False, origin_scope: Optional[
        Any]=None) ->Generator[AbstractFilter, None, None]:
        yield self._ensure_one_filter(is_instance)

    @memoize_method
    def _ensure_one_filter(self, is_instance: bool) ->'CompiledValueFilter':
        return CompiledValueFilter(self.inference_state, self, is_instance)

    def py__simple_getitem__(self, index: Any) ->ValueSet:
        with reraise_getitem_errors(IndexError, KeyError, TypeError):
            try:
                access = self.access_handle.py__simple_getitem__(index)
            except AttributeError:
                return super().py__simple_getitem__(index)
        if access is None:
            return super().py__simple_getitem__(index)
        return ValueSet([create_from_access_path(self.inference_state, access)]
            )

    def py__getitem__(self, index_value_set: Any, contextualized_node: Any
        ) ->ValueSet:
        all_access_paths = self.access_handle.py__getitem__all_values()
        if all_access_paths is None:
            return super().py__getitem__(index_value_set, contextualized_node)
        return ValueSet(create_from_access_path(self.inference_state,
            access) for access in all_access_paths)

    def py__iter__(self, contextualized_node: Optional[Any]=None) ->Generator[
        Any, None, None]:
        if not self.access_handle.has_iter():
            yield from super().py__iter__(contextualized_node)
        access_path_list = self.access_handle.py__iter__list()
        if access_path_list is None:
            return
        for access in access_path_list:
            yield LazyKnownValue(create_from_access_path(self.
                inference_state, access))

    def py__name__(self) ->Optional[str]:
        return self.access_handle.py__name__()

    @property
    def name(self) ->'CompiledValueName':
        name = self.py__name__()
        if name is None:
            name = self.access_handle.get_repr()
        return CompiledValueName(self, name)

    def _execute_function(self, params: Any) ->Generator[Any, None, None]:
        from jedi.inference import docstrings
        from jedi.inference.compiled import builtin_from_name
        if self.api_type != 'function':
            return
        for name in self._parse_function_doc()[1].split():
            try:
                self.inference_state.builtins_module.access_handle.getattr_paths(
                    name)
            except AttributeError:
                continue
            else:
                bltn_obj = builtin_from_name(self.inference_state, name)
                yield from self.inference_state.execute(bltn_obj, params)
        yield from docstrings.infer_return_types(self)

    def get_safe_value(self, default: Any=_sentinel) ->Any:
        try:
            return self.access_handle.get_safe_value()
        except ValueError:
            if default == _sentinel:
                raise
            return default

    def execute_operation(self, other: 'CompiledValue', operator: str
        ) ->ValueSet:
        try:
            return ValueSet([create_from_access_path(self.inference_state,
                self.access_handle.execute_operation(other.access_handle,
                operator))])
        except TypeError:
            return NO_VALUES

    def execute_annotation(self) ->ValueSet:
        if self.access_handle.get_repr() == 'None':
            return ValueSet([self])
        name, args = self.access_handle.get_annotation_name_and_args()
        arguments = [ValueSet([create_from_access_path(self.inference_state,
            path)]) for path in args]
        if name == 'Union':
            return ValueSet.from_sets(arg.execute_annotation() for arg in
                arguments)
        elif name:
            return ValueSet([v.with_generics(arguments) for v in self.
                inference_state.typing_module.py__getattribute__(name)]
                ).execute_annotation()
        return super().execute_annotation()

    def negate(self) ->'CompiledValue':
        return create_from_access_path(self.inference_state, self.
            access_handle.negate())

    def get_metaclasses(self) ->ValueSet:
        return NO_VALUES

    def _as_context(self) ->'CompiledContext':
        return CompiledContext(self)

    @property
    def array_type(self) ->Optional[str]:
        return self.access_handle.get_array_type()

    def get_key_values(self) ->List['CompiledValue']:
        return [create_from_access_path(self.inference_state, k) for k in
            self.access_handle.get_key_paths()]

    def get_type_hint(self, add_class_info: bool=True) ->Optional[str]:
        if self.access_handle.get_repr() in ('None', "<class 'NoneType'>"):
            return 'None'
        return None


class CompiledModule(CompiledValue):
    file_io: Optional[Any] = None

    def _as_context(self) ->'CompiledModuleContext':
        return CompiledModuleContext(self)

    def py__path__(self) ->Optional[List[Path]]:
        return self.access_handle.py__path__()

    def is_package(self) ->bool:
        return self.py__path__() is not None

    @property
    def string_names(self) ->Tuple[str, ...]:
        name = self.py__name__()
        if name is None:
            return ()
        return tuple(name.split('.'))

    def py__file__(self) ->Optional[Path]:
        return self.access_handle.py__file__()


class CompiledName(AbstractNameDefinition):
    _inference_state: Any
    _parent_value: 'CompiledValue'
    string_name: str
    parent_context: 'CompiledContext'

    def __init__(self, inference_state: Any, parent_value: 'CompiledValue',
        name: str) ->None:
        self._inference_state = inference_state
        self.parent_context = parent_value.as_context()
        self._parent_value = parent_value
        self.string_name = name

    def py__doc__(self) ->Optional[str]:
        return self.infer_compiled_value().py__doc__()

    def _get_qualified_names(self) ->Optional[Tuple[str, ...]]:
        parent_qualified_names = self.parent_context.get_qualified_names()
        if parent_qualified_names is None:
            return None
        return parent_qualified_names + (self.string_name,)

    def get_defining_qualified_value(self) ->Optional['Value']:
        context = self.parent_context
        if context.is_module() or context.is_class():
            return context.get_value()
        return None

    def __repr__(self) ->str:
        try:
            name = self.parent_context.name
        except AttributeError:
            name = None
        return '<%s: (%s).%s>' % (self.__class__.__name__, name, self.
            string_name)

    @property
    def api_type(self) ->str:
        return self.infer_compiled_value().api_type

    def infer(self) ->ValueSet:
        return ValueSet([self.infer_compiled_value()])

    @memoize_method
    def infer_compiled_value(self) ->'CompiledValue':
        return create_from_name(self._inference_state, self._parent_value,
            self.string_name)


class SignatureParamName(ParamNameInterface, AbstractNameDefinition):
    parent_context: 'CompiledContext'
    _signature_param: Any
    string_name: str

    def __init__(self, compiled_value: 'CompiledValue', signature_param: Any
        ) ->None:
        self.parent_context = compiled_value.parent_context
        self._signature_param = signature_param

    @property
    def string_name(self) ->str:
        return self._signature_param.name

    def to_string(self) ->str:
        s = self._kind_string() + self.string_name
        if self._signature_param.has_annotation:
            s += ': ' + self._signature_param.annotation_string
        if self._signature_param.has_default:
            s += '=' + self._signature_param.default_string
        return s

    def get_kind(self) ->int:
        return getattr(Parameter, self._signature_param.kind_name)

    def infer(self) ->ValueSet:
        p = self._signature_param
        inference_state = self.parent_context.inference_state
        values: ValueSet = NO_VALUES
        if p.has_default:
            values = ValueSet([create_from_access_path(inference_state, p.
                default)])
        if p.has_annotation:
            annotation = create_from_access_path(inference_state, p.annotation)
            values |= annotation.execute_with_values()
        return values


class UnresolvableParamName(ParamNameInterface, AbstractNameDefinition):
    parent_context: 'CompiledContext'
    string_name: str
    _default: str

    def __init__(self, compiled_value: 'CompiledValue', name: str, default: str
        ) ->None:
        self.parent_context = compiled_value.parent_context
        self.string_name = name
        self._default = default

    def get_kind(self) ->int:
        return Parameter.POSITIONAL_ONLY

    def to_string(self) ->str:
        string = self.string_name
        if self._default:
            string += '=' + self._default
        return string

    def infer(self) ->ValueSet:
        return NO_VALUES


class CompiledValueName(ValueNameMixin, AbstractNameDefinition):
    string_name: str
    _value: 'CompiledValue'
    parent_context: 'CompiledContext'

    def __init__(self, value: 'CompiledValue', name: str) ->None:
        self.string_name = name
        self._value = value
        self.parent_context = value.parent_context


class EmptyCompiledName(AbstractNameDefinition):
    parent_context: 'CompiledContext'
    string_name: str
    """
    Accessing some names will raise an exception. To avoid not having any
    completions, just give Jedi the option to return this object. It infers to
    nothing.
    """

    def __init__(self, inference_state: Any, name: str) ->None:
        self.parent_context = inference_state.builtins_module
        self.string_name = name

    def infer(self) ->ValueSet:
        return NO_VALUES


class CompiledValueFilter(AbstractFilter):
    _inference_state: Any
    compiled_value: 'CompiledValue'
    is_instance: bool

    def __init__(self, inference_state: Any, compiled_value:
        'CompiledValue', is_instance: bool=False) ->None:
        self._inference_state = inference_state
        self.compiled_value = compiled_value
        self.is_instance = is_instance

    def get(self, name: str) ->List[AbstractNameDefinition]:
        access_handle = self.compiled_value.access_handle
        return self._get(name, lambda name, safe: access_handle.
            is_allowed_getattr(name, safe=safe), lambda name: name in
            access_handle.dir(), check_has_attribute=True)

    def _get(self, name: str, allowed_getattr_callback, in_dir_callback:
        Callable[[str], bool], check_has_attribute: bool=False) ->List[
        AbstractNameDefinition]:
        """
        To remove quite a few access calls we introduced the callback here.
        """
        if self._inference_state.allow_descriptor_getattr:
            pass
        has_attribute, is_descriptor = allowed_getattr_callback(name, safe=
            not self._inference_state.allow_descriptor_getattr)
        if check_has_attribute and not has_attribute:
            return []
        if (is_descriptor or not has_attribute
            ) and not self._inference_state.allow_descriptor_getattr:
            return [self._get_cached_name(name, is_empty=True)]
        if self.is_instance and not in_dir_callback(name):
            return []
        return [self._get_cached_name(name)]

    @memoize_method
    def _get_cached_name(self, name: str, is_empty: bool=False
        ) ->AbstractNameDefinition:
        if is_empty:
            return EmptyCompiledName(self._inference_state, name)
        else:
            return self._create_name(name)

    def values(self) ->List[AbstractNameDefinition]:
        from jedi.inference.compiled import builtin_from_name
        names: List[AbstractNameDefinition] = []
        needs_type_completions: bool
        dir_infos: Dict[str, bool]
        needs_type_completions, dir_infos = (self.compiled_value.
            access_handle.get_dir_infos())
        for name in dir_infos:
            names += self._get(name, lambda name, safe: dir_infos[name], lambda
                name: name in dir_infos)
        if not self.is_instance and needs_type_completions:
            for filter in builtin_from_name(self._inference_state, 'type'
                ).get_filters():
                names += filter.values()
        return names

    def _create_name(self, name: str) ->AbstractNameDefinition:
        return CompiledName(self._inference_state, self.compiled_value, name)

    def __repr__(self) ->str:
        return '<%s: %s>' % (self.__class__.__name__, self.compiled_value)


docstr_defaults: Dict[str, str] = {'floating point number': 'float',
    'character': 'str', 'integer': 'int', 'dictionary': 'dict', 'string': 'str'
    }


def _parse_function_doc(doc: str) ->Tuple[str, str]:
    """
    Takes a function and returns the params and return value as a tuple.
    This is nothing more than a docstring parser.

    TODO docstrings like utime(path, (atime, mtime)) and a(b [, b]) -> None
    TODO docstrings like 'tuple of integers'
    """
    try:
        count = 0
        start = doc.index('(')
        for i, s in enumerate(doc[start:]):
            if s == '(':
                count += 1
            elif s == ')':
                count -= 1
            if count == 0:
                end = start + i
                break
        param_str = doc[start + 1:end]
    except (ValueError, UnboundLocalError):
        debug.dbg('no brackets found - no param')
        end = 0
        param_str = ''
    else:

        def change_options(m: re.Match):
            args = m.group(1).split(',')
            for i, a in enumerate(args):
                if a and '=' not in a:
                    args[i] += '=None'
            return ','.join(args)
        while True:
            param_str, changes = re.subn(' ?\\[([^\\[\\]]+)\\]',
                change_options, param_str)
            if changes == 0:
                break
    param_str = param_str.replace('-', '_')
    r = re.search('-[>-]* ', doc[end:end + 7])
    if r is None:
        ret = ''
    else:
        index = end + r.end()
        pattern = re.compile('(,\\n|[^\\n-])+')
        ret_str = pattern.match(doc, index).group(0).strip()
        ret_str = re.sub('[nN]ew (.*)', '\\1()', ret_str)
        ret = docstr_defaults.get(ret_str, ret_str)
    return param_str, ret


def create_from_name(inference_state: Any, compiled_value: 'CompiledValue',
    name: str):
    access_paths = compiled_value.access_handle.getattr_paths(name, default
        =None)
    value: Optional['CompiledValue'] = None
    for access_path in access_paths:
        value = create_cached_compiled_value(inference_state, access_path,
            parent_context=None if value is None else value.as_context())
    return value


def _normalize_create_args(func: Callable[..., 'CompiledValue']) ->Callable[
    ..., 'CompiledValue']:
    """The cache doesn't care about keyword vs. normal args."""

    @wraps(func)
    def wrapper(inference_state: Any, obj: Any, parent_context: Optional[
        'CompiledContext']=None) ->'CompiledValue':
        return func(inference_state, obj, parent_context)
    return wrapper


def create_from_access_path(inference_state: Any, access_path: Any):
    value: Optional['CompiledValue'] = None
    for name, access in access_path.accesses:
        value = create_cached_compiled_value(inference_state, access,
            parent_context=None if value is None else value.as_context())
    return value


@_normalize_create_args
@inference_state_function_cache()
def create_cached_compiled_value(inference_state: Any, access_handle: Any,
    parent_context: Optional['CompiledContext']) ->'CompiledValue':
    assert not isinstance(parent_context, CompiledValue)
    if parent_context is None:
        cls = CompiledModule
    else:
        cls = CompiledValue
    return cls(inference_state, access_handle, parent_context)
