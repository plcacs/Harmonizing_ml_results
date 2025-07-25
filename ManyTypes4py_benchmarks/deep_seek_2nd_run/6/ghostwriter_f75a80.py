from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    ForwardRef,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
)
import ast
import builtins
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Iterable as CollectionsIterable, Mapping as CollectionsMapping
from itertools import permutations, zip_longest
from keyword import iskeyword as _iskeyword
from string import ascii_lowercase
from textwrap import dedent, indent
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, ForwardRef, NamedTuple, Optional, TypeVar, Union, get_args, get_origin
import black
from hypothesis import Verbosity, find, settings, strategies as st
from hypothesis.errors import InvalidArgument, SmallSearchSpaceWarning
from hypothesis.internal.compat import get_type_hints
from hypothesis.internal.reflection import get_signature, is_mock
from hypothesis.internal.validation import check_type
from hypothesis.provisional import domains
from hypothesis.strategies._internal.collections import ListStrategy
from hypothesis.strategies._internal.core import BuildsStrategy
from hypothesis.strategies._internal.deferred import DeferredStrategy
from hypothesis.strategies._internal.flatmapped import FlatMapStrategy
from hypothesis.strategies._internal.lazy import LazyStrategy, unwrap_strategies
from hypothesis.strategies._internal.strategies import (
    FilteredStrategy,
    MappedStrategy,
    OneOfStrategy,
    SampledFromStrategy,
)
from hypothesis.strategies._internal.types import _global_type_lookup, is_generic_type

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)

IMPORT_SECTION = '\n# This test code was written by the `hypothesis.extra.ghostwriter` module\n# and is provided under the Creative Commons Zero public domain dedication.\n\n{imports}\n'
TEMPLATE = '\n@given({given_args})\ndef test_{test_kind}_{func_name}({arg_names}){return_annotation}:\n{test_body}\n'
SUPPRESS_BLOCK = '\ntry:\n{test_body}\nexcept {exceptions}:\n    reject()\n'.strip()

Except = Union[Type[Exception], Tuple[Type[Exception], ...]]
ImportSet = Set[Union[str, Tuple[str, str]]]
_quietly_settings = settings(database=None, deadline=None, derandomize=True, verbosity=Verbosity.quiet)

def _dedupe_exceptions(exc: Tuple[Type[Exception], ...) -> Tuple[Type[Exception], ...]:
    uniques = list(exc)
    for a, b in permutations(exc, 2):
        if a in uniques and issubclass(a, b):
            uniques.remove(a)
    return tuple(sorted(uniques, key=lambda e: e.__name__))

def _check_except(except_: Except) -> Tuple[Type[Exception], ...]:
    if isinstance(except_, tuple):
        for i, e in enumerate(except_):
            if not isinstance(e, type) or not issubclass(e, Exception):
                raise InvalidArgument(f'Expected an Exception but got except_[{i}]={e!r} (type={_get_qualname(type(e))})')
        return except_
    if not isinstance(except_, type) or not issubclass(except_, Exception):
        raise InvalidArgument(f'Expected an Exception or tuple of exceptions, but got except_={except_!r} (type={_get_qualname(type(except_))})')
    return (except_,)

def _exception_string(except_: Except) -> Tuple[Set[str], str]:
    if not except_:
        return (set(), '')
    exceptions = []
    imports = set()
    for ex in _dedupe_exceptions(except_):
        if ex.__qualname__ in dir(builtins):
            exceptions.append(ex.__qualname__)
        else:
            imports.add(ex.__module__)
            exceptions.append(_get_qualname(ex, include_module=True))
    return (imports, '(' + ', '.join(exceptions) + ')' if len(exceptions) > 1 else exceptions[0])

def _check_style(style: str) -> None:
    if style not in ('pytest', 'unittest'):
        raise InvalidArgument(f"Valid styles are 'pytest' or 'unittest', got {style!r}")

def _exceptions_from_docstring(doc: str) -> Tuple[Type[Exception], ...]:
    """Return a tuple of exceptions that the docstring says may be raised.

    Note that we ignore non-builtin exception types for simplicity, as this is
    used directly in _write_call() and passing import sets around would be really
    really annoying.
    """
    assert isinstance(doc, str), doc
    raises = []
    for excname in re.compile('\\:raises\\s+(\\w+)\\:', re.MULTILINE).findall(doc):
        exc_type = getattr(builtins, excname, None)
        if isinstance(exc_type, type) and issubclass(exc_type, Exception):
            raises.append(exc_type)
    return tuple(_dedupe_exceptions(tuple(raises)))

def _type_from_doc_fragment(token: str) -> Optional[Type]:
    if token == 'integer':
        return int
    if 'numpy' in sys.modules:
        if re.fullmatch('[Aa]rray[-_ ]?like', token):
            return sys.modules['numpy'].ndarray
        elif token == 'dtype':
            return sys.modules['numpy'].dtype
    coll_match = re.fullmatch('(\\w+) of (\\w+)', token)
    if coll_match is not None:
        coll_token, elem_token = coll_match.groups()
        elems = _type_from_doc_fragment(elem_token)
        if elems is None and elem_token.endswith('s'):
            elems = _type_from_doc_fragment(elem_token[:-1])
        if elems is not None and coll_token in ('list', 'sequence', 'collection'):
            return list[elems]
        return _type_from_doc_fragment(coll_token)
    if '.' not in token:
        return getattr(builtins, token, None)
    mod, name = token.rsplit('.', maxsplit=1)
    return getattr(sys.modules.get(mod, None), name, None)

def _strip_typevars(type_: Type) -> Type:
    with contextlib.suppress(Exception):
        if {type(a) for a in get_args(type_)} == {TypeVar}:
            return get_origin(type_)
    return type_

def _strategy_for(param: inspect.Parameter, docstring: str) -> st.SearchStrategy:
    for pattern in (f'^\\s*\\:type\\s+{param.name}\\:\\s+(.+)', f'^\\s*{param.name} \\((.+)\\):', f'^\\s*{param.name} \\: (.+)'):
        match = re.search(pattern, docstring, flags=re.MULTILINE)
        if match is None:
            continue
        doc_type = match.group(1)
        if doc_type.endswith(', optional'):
            doc_type = doc_type[:-len(', optional')]
        doc_type = doc_type.strip('}{')
        elements = []
        types = []
        for token in re.split(',? +or +| *, *', doc_type):
            for prefix in ('default ', 'python '):
                token = token.removeprefix(prefix)
            if not token:
                continue
            try:
                elements.append(ast.literal_eval(token))
                continue
            except (ValueError, SyntaxError):
                t = _type_from_doc_fragment(token)
                if isinstance(t, type) or is_generic_type(t):
                    assert t is not None
                    types.append(_strip_typevars(t))
        if param.default is not inspect.Parameter.empty and param.default not in elements and (not isinstance(param.default, tuple((t for t in types if isinstance(t, type))))):
            with contextlib.suppress(SyntaxError):
                compile(repr(st.just(param.default)), '<string>', 'eval')
                elements.insert(0, param.default)
        if elements or types:
            return (st.sampled_from(elements) if elements else st.nothing()) | (st.one_of(*map(st.from_type, types)) if types else st.nothing())
    if isinstance(param.default, bool):
        return st.booleans()
    if isinstance(param.default, enum.Enum):
        return st.sampled_from(type(param.default))
    if param.default is not inspect.Parameter.empty:
        return st.just(param.default)
    return _guess_strategy_by_argname(name=param.name.lower())

BOOL_NAMES = ('keepdims', 'verbose', 'debug', 'force', 'train', 'training', 'trainable', 'bias', 'shuffle', 'show', 'load', 'pretrained', 'save', 'overwrite', 'normalize', 'reverse', 'success', 'enabled', 'strict', 'copy', 'quiet', 'required', 'inplace', 'recursive', 'enable', 'active', 'create', 'validate', 'refresh', 'use_bias')
POSITIVE_INTEGER_NAMES = ('width', 'size', 'length', 'limit', 'idx', 'stride', 'epoch', 'epochs', 'depth', 'pid', 'steps', 'iteration', 'iterations', 'vocab_size', 'ttl', 'count')
FLOAT_NAMES = ('real', 'imag', 'alpha', 'theta', 'beta', 'sigma', 'gamma', 'angle', 'reward', 'tau', 'temperature')
STRING_NAMES = ('text', 'txt', 'password', 'label', 'prefix', 'suffix', 'desc', 'description', 'str', 'pattern', 'subject', 'reason', 'comment', 'prompt', 'sentence', 'sep')

def _guess_strategy_by_argname(name: str) -> st.SearchStrategy:
    """
    If all else fails, we try guessing a strategy based on common argument names.

    We wouldn't do this in builds() where strict correctness is required, but for
    the ghostwriter we accept "good guesses" since the user would otherwise have
    to change the strategy anyway - from `nothing()` - if we refused to guess.

    A "good guess" is _usually correct_, and _a reasonable mistake_ if not.
    The logic below is therefore based on a manual reading of the builtins and
    some standard-library docs, plus the analysis of about three hundred million
    arguments in https://github.com/HypothesisWorks/hypothesis/issues/3311
    """
    if name in ('function', 'func', 'f'):
        return st.functions()
    if name in ('pred', 'predicate'):
        return st.functions(returns=st.booleans(), pure=True)
    if name in ('iterable',):
        return st.iterables(st.integers()) | st.iterables(st.text())
    if name in ('list', 'lst', 'ls'):
        return st.lists(st.nothing())
    if name in ('object',):
        return st.builds(object)
    if 'uuid' in name:
        return st.uuids().map(str)
    if name.startswith('is_') or name in BOOL_NAMES:
        return st.booleans()
    if name in ('amount', 'threshold', 'number', 'num'):
        return st.integers() | st.floats()
    if name in ('port',):
        return st.integers(0, 2 ** 16 - 1)
    if name.endswith('_size') or (name.endswith('size') and '_' not in name) or re.fullmatch('n(um)?_[a-z_]*s', name) or (name in POSITIVE_INTEGER_NAMES):
        return st.integers(min_value=0)
    if name in ('offset', 'seed', 'dim', 'total', 'priority'):
        return st.integers()
    if name in ('learning_rate', 'dropout', 'dropout_rate', 'epsilon', 'eps', 'prob'):
        return st.floats(0, 1)
    if name in ('lat', 'latitude'):
        return st.floats(-90, 90)
    if name in ('lon', 'longitude'):
        return st.floats(-180, 180)
    if name in ('radius', 'tol', 'tolerance', 'rate'):
        return st.floats(min_value=0)
    if name in FLOAT_NAMES:
        return st.floats()
    if name in ('host', 'hostname'):
        return domains()
    if name in ('email',):
        return st.emails()
    if name in ('word', 'slug', 'api_key'):
        return st.from_regex('\\w+', fullmatch=True)
    if name in ('char', 'character'):
        return st.characters()
    if 'file' in name or 'path' in name or name.endswith('_dir') or (name in ('fname', 'dir', 'dirname', 'directory', 'folder')):
        return st.nothing()
    if name.endswith(('_name', 'label')) or (name.endswith('name') and '_' not in name) or ('string' in name and 'as' not in name) or (name in STRING_NAMES):
        return st.text()
    if re.fullmatch('\\w*[^s]s', name):
        elems = _guess_strategy_by_argname(name[:-1])
        if not elems.is_empty:
            return st.lists(elems)
    return st.nothing()

def _get_params_builtin_fn(func: Callable) -> List[inspect.Parameter]:
    if isinstance(func, (types.BuiltinFunctionType, types.BuiltinMethodType)) and hasattr(func, '__doc__') and isinstance(func.__doc__, str):
        match = re.match(f'^{func.__name__}\\((.+?)\\)', func.__doc__)
        if match is None:
            return []
        args = match.group(1).replace('[', '').replace(']', '')
        params = []
        kind = inspect.Parameter.POSITIONAL_ONLY
        for arg in args.split(', '):
            arg, *_ = arg.partition('=')
            arg = arg.strip()
            if arg == '/':
                kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                continue
            if arg.startswith('*') or arg == '...':
                kind = inspect.Parameter.KEYWORD_ONLY
                continue
            if _iskeyword(arg.lstrip('*')) or not arg.lstrip('*').isidentifier():
                break
            params.append(inspect.Parameter(name=arg, kind=kind))
        return params
    return []

def _get_params_ufunc(func: Callable) -> List[inspect.Parameter]:
    if _is_probably_ufunc(func):
        return [inspect.Parameter(name=name, kind=inspect.Parameter.POSITIONAL_ONLY) for name in ascii_lowercase[:func.nin]]
    return []

def _get_params(func: Callable) -> OrderedDict[str, inspect.Parameter]:
    """Get non-vararg parameters of `func` as an ordered dict."""
    try:
        params = list(get_signature(func).parameters.values())
    except Exception:
        if (params := _get_params_ufunc(func)):
            pass
        elif (params := _get_params_builtin_fn(func)):
            pass
        else:
            raise
    else:
        P = inspect.Parameter
        placeholder = [('args', P.VAR_POSITIONAL), ('kwargs', P.VAR_KEYWORD)]
        if [(p.name, p.kind) for p in params] == placeholder:
            params = _get_params_ufunc(func) or _get_params_builtin_fn(func) or params
    return _params_to_dict(params)

def _params_to_dict(params: List[inspect.Parameter]) -> OrderedDict[str, inspect.Parameter]:
    var_param_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return OrderedDict(((p.name, p) for p in params if p.kind not in var_param_kinds))

@contextlib.contextmanager
def _with_any_registered() -> Iterable[None]:
    if Any in _global_type_lookup:
        yield
    else:
        try:
            _global_type_lookup[Any] = st.builds(object)
            yield
        finally:
            del _global_type_lookup[Any]
            st.from_type.__clear_cache()

def _get_strategies(*funcs: Callable, pass_result_to_next_func: bool = False) -> Dict[str, st.SearchStrategy]:
    """Return a dict of strategies for the union of arguments to `funcs`.

    If `pass_result_to_next_func` is True, assume that the result of each function
    is passed to the next, and therefore skip the first argument of all but the
    first function.

    This dict is used to construct our call to the `@given(...)` decorator.
    """
    assert funcs, 'Must pass at least one function to test.'
    given_strategies = {}
    for i, f in enumerate(funcs):
        params = _get_params(f)
        if pass_result_to_next_func and i >= 1:
            del params[next(iter(params))]
        hints = get_type_hints(f)
        docstring = getattr(f, '__doc__', None) or ''
        builder_args = {k: ... if k in hints else _strategy_for(v, docstring) for k, v in params.items()}
        with _with_any_registered():
            strat = st.builds(f, **builder_args).wrapped_strategy
        if strat.args:
            raise NotImplementedError('Expected to pass everything as kwargs')
        for k, v in strat.kwargs.items():
            if _valid_syntax_repr(v)[1] == 'nothing()' and k in hints:
                v = LazyStrategy(st.from_type, (hints[k],), {})
            if k in given_str