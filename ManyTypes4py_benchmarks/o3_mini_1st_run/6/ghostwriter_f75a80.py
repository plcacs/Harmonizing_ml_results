#!/usr/bin/env python3
"""
Writing tests with Hypothesis frees you from the tedium of deciding on and
writing out specific inputs to test.  Now, the ``hypothesis.extra.ghostwriter``
module can write your test functions for you too!

The idea is to provide **an easy way to start** property-based testing,
**and a seamless transition** to more complex test code - because ghostwritten
tests are source code that you could have written for yourself.

So just pick a function you'd like tested, and feed it to one of the functions
below.  They follow imports, use but do not require type annotations, and
generally do their best to write you a useful test.  You can also use
:ref:`our command-line interface <hypothesis-cli>`::

    $ hypothesis write --help
    Usage: hypothesis write [OPTIONS] FUNC...

      `hypothesis write` writes property-based tests for you!

      Type annotations are helpful but not required for our advanced
      introspection and templating logic.  Try running the examples below to see
      how it works:

          hypothesis write gzip
          hypothesis write numpy.matmul
          hypothesis write pandas.from_dummies
          hypothesis write re.compile --except re.error
          hypothesis write --equivalent ast.literal_eval eval
          hypothesis write --roundtrip json.dumps json.loads
          hypothesis write --style=unittest --idempotent sorted
          hypothesis write --binary-op operator.add

    Options:
      --roundtrip                 start by testing write/read or encode/decode!
      --equivalent                very useful when optimising or refactoring code
      --errors-equivalent         --equivalent, but also allows consistent errors
      --idempotent                check that f(x) == f(f(x))
      --binary-op                 associativity, commutativity, identity element
      --style [pytest|unittest]   pytest-style function, or unittest-style method?
      -e, --except OBJ_NAME       dotted name of exception(s) to ignore
      --annotate / --no-annotate  force ghostwritten tests to be type-annotated
                                  (or not).  By default, match the code to test.
      -h, --help                  Show this message and exit.

.. tip::

    Using a light theme?  Hypothesis respects `NO_COLOR <https://no-color.org/>`__
    and ``DJANGO_COLORS=light``.

.. note::

    The ghostwriter requires :pypi:`black`, but the generated code only
    requires Hypothesis itself.

.. note::

    Legal questions?  While the ghostwriter fragments and logic is under the
    MPL-2.0 license like the rest of Hypothesis, the *output* from the ghostwriter
    is made available under the `Creative Commons Zero (CC0)
    <https://creativecommons.org/public-domain/cc0/>`__
    public domain dedication, so you can use it without any restrictions.
"""
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
from collections.abc import Iterable, Mapping
from itertools import permutations, zip_longest
from keyword import iskeyword as _iskeyword
from string import ascii_lowercase
from textwrap import dedent, indent
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Dict, Iterable as TypingIterable, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, TypeVar, Union, get_args, get_origin

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
from hypothesis.strategies._internal.strategies import FilteredStrategy, MappedStrategy, OneOfStrategy, SampledFromStrategy
from hypothesis.strategies._internal.types import _global_type_lookup, is_generic_type

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)

ImportSet = Set[Union[str, Tuple[str, str]]]
ExceptType = Union[type[Exception], Tuple[type[Exception], ...]]

IMPORT_SECTION: str = (
    "\n# This test code was written by the `hypothesis.extra.ghostwriter` module\n"
    "# and is provided under the Creative Commons Zero public domain dedication.\n\n{imports}\n"
)
TEMPLATE: str = (
    "\n@given({given_args})\n"
    "def test_{test_kind}_{func_name}({arg_names}){return_annotation}:\n"
    "{test_body}\n"
)
SUPPRESS_BLOCK: str = (
    "\ntry:\n{test_body}\nexcept {exceptions}:\n    reject()\n"
).strip()
_quietly_settings = settings(database=None, deadline=None, derandomize=True, verbosity=Verbosity.quiet)

def _dedupe_exceptions(exc: Tuple[type[Exception], ...]) -> Tuple[type[Exception], ...]:
    uniques: List[type[Exception]] = list(exc)
    for a, b in permutations(exc, 2):
        if a in uniques and issubclass(a, b):
            uniques.remove(a)
    return tuple(sorted(uniques, key=lambda e: e.__name__))

def _check_except(except_: Union[type[Exception], Tuple[type[Exception], ...]]) -> Tuple[type[Exception], ...]:
    if isinstance(except_, tuple):
        for i, e in enumerate(except_):
            if not isinstance(e, type) or not issubclass(e, Exception):
                raise InvalidArgument(
                    f'Expected an Exception but got except_[{i}]={e!r} (type={_get_qualname(type(e))})'
                )
        return except_
    if not isinstance(except_, type) or not issubclass(except_, Exception):
        raise InvalidArgument(
            f'Expected an Exception or tuple of exceptions, but got except_={except_!r} (type={_get_qualname(type(except_))})'
        )
    return (except_,)

def _exception_string(except_: Tuple[type[Exception], ...]) -> Tuple[ImportSet, str]:
    if not except_:
        return (set(), '')
    exceptions: List[type[Exception]] = []
    imports: Set[Union[str, Tuple[str, str]]] = set()
    for ex in _dedupe_exceptions(except_):
        if ex.__qualname__ in dir(builtins):
            exceptions.append(ex)
        else:
            imports.add((ex.__module__, ex.__name__))
            exceptions.append(ex)
    exc_names = [ex.__qualname__ for ex in exceptions]
    if len(exc_names) > 1:
        return (imports, '(' + ', '.join(exc_names) + ')')
    else:
        return (imports, exc_names[0])

def _check_style(style: str) -> None:
    if style not in ('pytest', 'unittest'):
        raise InvalidArgument(f"Valid styles are 'pytest' or 'unittest', got {style!r}")

def _exceptions_from_docstring(doc: str) -> Tuple[type[Exception], ...]:
    assert isinstance(doc, str), doc
    raises: List[type[Exception]] = []
    for excname in re.compile(r'\:raises\s+(\w+)\:', re.MULTILINE).findall(doc):
        exc_type = getattr(builtins, excname, None)
        if isinstance(exc_type, type) and issubclass(exc_type, Exception):
            raises.append(exc_type)
    return _dedupe_exceptions(tuple(raises))

def _type_from_doc_fragment(token: str) -> Optional[Any]:
    if token == 'integer':
        return int
    if 'numpy' in sys.modules:
        if re.fullmatch('[Aa]rray[-_ ]?like', token):
            return sys.modules['numpy'].ndarray
        elif token == 'dtype':
            return sys.modules['numpy'].dtype
    coll_match = re.fullmatch(r'(\w+) of (\w+)', token)
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

def _strip_typevars(type_: Any) -> Any:
    with contextlib.suppress(Exception):
        if {type(a) for a in get_args(type_)} == {TypeVar}:
            return get_origin(type_)
    return type_

def _strategy_for(param: inspect.Parameter, docstring: str) -> st.SearchStrategy[Any]:
    for pattern in (f'^\\s*\\:type\\s+{param.name}\\:\\s+(.+)', f'^\\s*{param.name} \\((.+)\\):', f'^\\s*{param.name} \\: (.+)'):
        match = re.search(pattern, docstring, flags=re.MULTILINE)
        if match is None:
            continue
        doc_type = match.group(1)
        if doc_type.endswith(', optional'):
            doc_type = doc_type[:-len(', optional')]
        doc_type = doc_type.strip('}{')
        elements: List[Any] = []
        types_list: List[Any] = []
        for token in re.split(r',? +or +| *, *', doc_type):
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
                    types_list.append(_strip_typevars(t))
        if param.default is not inspect.Parameter.empty and param.default not in elements and (
            not isinstance(param.default, tuple((t for t in types_list if isinstance(t, type))))
        ):
            with contextlib.suppress(SyntaxError):
                compile(repr(st.just(param.default)), '<string>', 'eval')
                elements.insert(0, param.default)
        if elements or types_list:
            strat_elements = st.sampled_from(elements) if elements else st.nothing()
            strat_types = st.one_of(*map(st.from_type, types_list)) if types_list else st.nothing()
            return strat_elements | strat_types
    if isinstance(param.default, bool):
        return st.booleans()
    if isinstance(param.default, enum.Enum):
        return st.sampled_from(type(param.default))
    if param.default is not inspect.Parameter.empty:
        return st.just(param.default)
    return _guess_strategy_by_argname(name=param.name.lower())

BOOL_NAMES: Tuple[str, ...] = (
    'keepdims', 'verbose', 'debug', 'force', 'train', 'training', 'trainable',
    'bias', 'shuffle', 'show', 'load', 'pretrained', 'save', 'overwrite',
    'normalize', 'reverse', 'success', 'enabled', 'strict', 'copy', 'quiet',
    'required', 'inplace', 'recursive', 'enable', 'active', 'create', 'validate',
    'refresh', 'use_bias'
)
POSITIVE_INTEGER_NAMES: Tuple[str, ...] = (
    'width', 'size', 'length', 'limit', 'idx', 'stride', 'epoch', 'epochs',
    'depth', 'pid', 'steps', 'iteration', 'iterations', 'vocab_size', 'ttl', 'count'
)
FLOAT_NAMES: Tuple[str, ...] = (
    'real', 'imag', 'alpha', 'theta', 'beta', 'sigma', 'gamma', 'angle',
    'reward', 'tau', 'temperature'
)
STRING_NAMES: Tuple[str, ...] = (
    'text', 'txt', 'password', 'label', 'prefix', 'suffix', 'desc', 'description',
    'str', 'pattern', 'subject', 'reason', 'comment', 'prompt', 'sentence', 'sep'
)

def _guess_strategy_by_argname(name: str) -> st.SearchStrategy[Any]:
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
    if name.endswith('_size') or (name.endswith('size') and '_' not in name) or re.fullmatch(r'n(um)?_[a-z_]*s', name) or (name in POSITIVE_INTEGER_NAMES):
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
        return st.from_regex(r'\w+', fullmatch=True)
    if name in ('char', 'character'):
        return st.characters()
    if 'file' in name or 'path' in name or name.endswith('_dir') or (name in ('fname', 'dir', 'dirname', 'directory', 'folder')):
        return st.nothing()
    if name.endswith(('_name', 'label')) or (name.endswith('name') and '_' not in name) or ('string' in name and 'as' not in name) or (name in STRING_NAMES):
        return st.text()
    if re.fullmatch(r'\w*[^s]s', name):
        elems = _guess_strategy_by_argname(name[:-1])
        if not elems.is_empty:
            return st.lists(elems)
    return st.nothing()

def _get_params_builtin_fn(func: Callable[..., Any]) -> List[inspect.Parameter]:
    if isinstance(func, (types.BuiltinFunctionType, types.BuiltinMethodType)) and hasattr(func, '__doc__') and isinstance(func.__doc__, str):
        match = re.match(f'^{func.__name__}\\((.+?)\\)', func.__doc__)
        if match is None:
            return []
        args = match.group(1).replace('[', '').replace(']', '')
        params: List[inspect.Parameter] = []
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

def _get_params_ufunc(func: Callable[..., Any]) -> List[inspect.Parameter]:
    if _is_probably_ufunc(func):
        return [inspect.Parameter(name=name, kind=inspect.Parameter.POSITIONAL_ONLY) for name in ascii_lowercase[:func.nin]]
    return []

def _get_params(func: Callable[..., Any]) -> "OrderedDict[str, inspect.Parameter]":
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

def _params_to_dict(params: TypingIterable[inspect.Parameter]) -> "OrderedDict[str, inspect.Parameter]":
    var_param_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return OrderedDict(((p.name, p) for p in params if p.kind not in var_param_kinds))

@contextlib.contextmanager
def _with_any_registered() -> Iterator[None]:
    if Any in _global_type_lookup:
        yield
    else:
        try:
            _global_type_lookup[Any] = st.builds(object)
            yield
        finally:
            del _global_type_lookup[Any]
            st.from_type.__clear_cache()

def _get_strategies(*funcs: Callable[..., Any], pass_result_to_next_func: bool = False) -> Dict[str, st.SearchStrategy[Any]]:
    assert funcs, 'Must pass at least one function'
    given_strategies: Dict[str, st.SearchStrategy[Any]] = {}
    for i, f in enumerate(funcs):
        params = _get_params(f)
        if pass_result_to_next_func and i >= 1:
            del params[next(iter(params))]
        hints = get_type_hints(f)
        docstring: str = getattr(f, '__doc__', None) or ''
        builder_args: Dict[str, Any] = {k: ... if k in hints else _strategy_for(v, docstring) for k, v in params.items()}
        with _with_any_registered():
            strat = st.builds(f, **builder_args).wrapped_strategy
        if strat.args:
            raise NotImplementedError('Expected to pass everything as kwargs')
        for k, v in strat.kwargs.items():
            if _valid_syntax_repr(v)[1] == 'nothing()' and k in hints:
                v = LazyStrategy(st.from_type, (hints[k],), {})
            if k in given_strategies:
                given_strategies[k] |= v
            else:
                given_strategies[k] = v
    if len(funcs) == 1:
        return {name: given_strategies[name] for name in _get_params(func)}
    return dict(sorted(given_strategies.items()))

def _assert_eq(style: str, a: str, b: str) -> str:
    if style == 'unittest':
        return f'self.assertEqual({a}, {b})'
    assert style == 'pytest'
    if a.isidentifier() and b.isidentifier():
        return f'assert {a} == {b}, ({a}, {b})'
    return f'assert {a} == {b}'

def _imports_for_object(obj: Any) -> Set[Union[str, Tuple[str, str]]]:
    if type(obj) is getattr(types, 'UnionType', object()):
        return {imp for mod, _ in set().union(*map(_imports_for_object, obj.__args__))}
    if isinstance(obj, (re.Pattern, re.Match)):
        return {'re'}
    if isinstance(obj, st.SearchStrategy):
        return _imports_for_strategy(obj)
    if isinstance(obj, getattr(sys.modules.get('numpy'), 'dtype', ())):
        return {('numpy', 'dtype')}
    try:
        if is_generic_type(obj):
            if isinstance(obj, TypeVar):
                return {(obj.__module__, obj.__name__)}
            with contextlib.suppress(Exception):
                return set().union(*map(_imports_for_object, obj.__args__))
        if not callable(obj) or obj.__name__ == '<lambda>':
            return set()
        name = _get_qualname(obj).split('.')[0]
        return {(_get_module(obj), name)}
    except Exception:
        return set()

def _imports_for_strategy(strategy: st.SearchStrategy[Any]) -> Set[Union[str, Tuple[str, str]]]:
    if isinstance(strategy, LazyStrategy):
        imports = {imp for arg in set(strategy._LazyStrategy__args) | set(strategy._LazyStrategy__kwargs.values())
                   for imp in _imports_for_object(_strip_typevars(arg))}
        if re.match('from_(type|regex)\\(', repr(strategy)):
            return imports
        elif _get_module(strategy.function).startswith('hypothesis.extra.'):
            module = _get_module(strategy.function).replace('._array_helpers', '.numpy')
            return {(module, strategy.function.__name__)} | imports
    imports: Set[Union[str, Tuple[str, str]]] = set()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SmallSearchSpaceWarning)
        strategy = unwrap_strategies(strategy)
    if isinstance(strategy, MappedStrategy):
        imports |= _imports_for_strategy(strategy.mapped_strategy)
        imports |= _imports_for_object(strategy.pack)
    if isinstance(strategy, FilteredStrategy):
        imports |= _imports_for_strategy(strategy.filtered_strategy)
        for f in strategy.flat_conditions:
            imports |= _imports_for_object(f)
    if isinstance(strategy, FlatMapStrategy):
        imports |= _imports_for_strategy(strategy.flatmapped_strategy)
        imports |= _imports_for_object(strategy.expand)
    if isinstance(strategy, OneOfStrategy):
        for s in strategy.element_strategies:
            imports |= _imports_for_strategy(s)
    if isinstance(strategy, BuildsStrategy):
        imports |= _imports_for_object(strategy.target)
        for s in strategy.args:
            imports |= _imports_for_strategy(s)
        for s in strategy.kwargs.values():
            imports |= _imports_for_strategy(s)
    if isinstance(strategy, SampledFromStrategy):
        for obj in strategy.elements:
            imports |= _imports_for_object(obj)
    if isinstance(strategy, ListStrategy):
        imports |= _imports_for_strategy(strategy.element_strategy)
    return imports

def _valid_syntax_repr(strategy: Any) -> Tuple[Set[Union[str, Tuple[str, str]]], str]:
    if isinstance(strategy, str):
        return (set(), strategy)
    try:
        if isinstance(strategy, DeferredStrategy):
            strategy = strategy.wrapped_strategy
        if isinstance(strategy, OneOfStrategy):
            seen: Set[str] = set()
            elems: List[Any] = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SmallSearchSpaceWarning)
                _ = strategy.element_strategies
            for s in strategy.element_strategies:
                if isinstance(s, SampledFromStrategy) and s.elements == (os.environ,):
                    continue
                if repr(s) not in seen:
                    elems.append(s)
                    seen.add(repr(s))
            strategy = st.one_of(elems or st.nothing())
        if strategy == st.text().wrapped_strategy:
            return (set(), 'text()')
        if isinstance(strategy, LazyStrategy) and strategy.function.__name__ == st.from_type.__name__ and (strategy._LazyStrategy__representation is None):
            strategy._LazyStrategy__args = tuple((_strip_typevars(a) for a in strategy._LazyStrategy__args))
        r = repr(strategy).replace('.filter(_can_hash)', '').replace('hypothesis.strategies.', '')
        r = re.sub(r'(lambda.*?: )(<unknown>)([,)])', r'\1...\3', r)
        compile(r, '<string>', 'eval')
        imports = {i for i in _imports_for_strategy(strategy) if (isinstance(i, str) and i in r) or (isinstance(i, tuple) and i[1] in r)}
        return (imports, r)
    except (SyntaxError, RecursionError, InvalidArgument):
        return (set(), 'nothing()')

KNOWN_FUNCTION_LOCATIONS: Dict[Any, str] = {}

def _get_module_helper(obj: Any) -> str:
    module_name: str = obj.__module__
    if module_name == 'collections.abc':
        return module_name
    dots = [i for i, c in enumerate(module_name) if c == '.'] + [None]
    for idx in dots:
        for candidate in (module_name[:idx].lstrip('_'), module_name[:idx]):
            if getattr(sys.modules.get(candidate), obj.__name__, None) is obj:
                KNOWN_FUNCTION_LOCATIONS[obj] = candidate
                return candidate
    return module_name

def _get_module(obj: Any) -> str:
    if obj in KNOWN_FUNCTION_LOCATIONS:
        return KNOWN_FUNCTION_LOCATIONS[obj]
    try:
        return _get_module_helper(obj)
    except AttributeError:
        if not _is_probably_ufunc(obj):
            raise
    for module_name in sorted(sys.modules, key=lambda n: tuple(n.split('.'))):
        if obj is getattr(sys.modules[module_name], obj.__name__, None):
            KNOWN_FUNCTION_LOCATIONS[obj] = module_name
            return module_name
    raise RuntimeError(f'Could not find module for ufunc {obj.__name__} ({obj!r}')

def _get_qualname(obj: Any, *, include_module: bool = False) -> str:
    qname: str = getattr(obj, '__qualname__', obj.__name__)
    qname = qname.replace('<', '_').replace('>', '_').replace(' ', '')
    if include_module:
        return _get_module(obj) + '.' + qname
    return qname

def _write_call(
    func: Callable[..., Any],
    *pass_variables: Optional[str],
    except_: ExceptType = Exception,
    assign: str = ''
) -> str:
    args = ', '.join(
        (v or p.name) if p.kind is inspect.Parameter.POSITIONAL_ONLY else f'{p.name}={v or p.name}'
        for v, p in zip_longest(pass_variables, _get_params(func).values())
    )
    call = f'{_get_qualname(func, include_module=True)}({args})'
    if assign:
        call = f'{assign} = {call}'
    raises = _exceptions_from_docstring(getattr(func, "__doc__", "") or "")
    exnames = [ex.__name__ for ex in raises if not issubclass(ex, except_)]
    if not exnames:
        return call
    return SUPPRESS_BLOCK.format(
        test_body=indent(call, prefix='    '),
        exceptions='(' + ', '.join(exnames) + ')' if len(exnames) > 1 else exnames[0]
    )

def _st_strategy_names(s: str) -> str:
    names = '|'.join(sorted(st.__all__, key=len, reverse=True))
    return re.sub(pattern=f'\\b(?:{names})\\b[^= ]', repl='st.\\g<0>', string=s)

def _make_test_body(
    *funcs: Callable[..., Any],
    ghost: str,
    test_body: str,
    except_: ExceptType,
    assertions: str = '',
    style: str,
    given_strategies: Optional[Dict[str, st.SearchStrategy[Any]]] = None,
    imports: Optional[Set[Union[str, Tuple[str, str]]]] = None,
    annotate: bool = False
) -> Tuple[ImportSet, str]:
    imports = (imports or set()) | {_get_module(f) for f in funcs}
    with _with_any_registered():
        given_strategies = given_strategies or _get_strategies(*funcs, pass_result_to_next_func=ghost in ('idempotent', 'roundtrip'))
        reprs = [(k, * _valid_syntax_repr(v)) for k, v in given_strategies.items()]
        imports = imports.union(*(imp for _, imp, _ in reprs))
        given_args = ', '.join((f'{k}={v}' for k, _, v in reprs))
    given_args = _st_strategy_names(given_args)
    if except_:
        imp, exc_string = _exception_string(except_)
        imports.update(imp)
        test_body = SUPPRESS_BLOCK.format(test_body=indent(test_body, prefix='    '), exceptions=exc_string)
    if assertions:
        test_body = f'{test_body}\n{assertions}'
    argnames: List[str] = ['self'] if style == 'unittest' else []
    if annotate:
        argnames.extend(list(_annotate_args(given_strategies, funcs, imports)))
    else:
        argnames.extend(list(given_strategies))
    body = TEMPLATE.format(
        given_args=given_args,
        test_kind=ghost,
        func_name='_'.join((_get_qualname(f).replace('.', '_') for f in funcs)),
        arg_names=', '.join(argnames),
        return_annotation=' -> None' if annotate else '',
        test_body=indent(test_body, prefix='    ')
    )
    if style == 'unittest':
        imports.add('unittest')
        body = 'class Test{}{}(unittest.TestCase):\n{}'.format(
            ghost.title(),
            ''.join((_get_qualname(f).replace('.', '').title() for f in funcs)),
            indent(body, '    ')
        )
    return (imports, body)

def _annotate_args(
    argnames: Dict[str, st.SearchStrategy[Any]],
    funcs: Sequence[Callable[..., Any]],
    imports: Set[Union[str, Tuple[str, str]]]
) -> Iterator[str]:
    arg_parameters: DefaultDict[str, Set[Any]] = defaultdict(set)
    for func in funcs:
        try:
            params = tuple(get_signature(func, eval_str=True).parameters.values())
        except Exception:
            pass
        else:
            for key, param in _params_to_dict(params).items():
                if param.annotation != inspect.Parameter.empty:
                    arg_parameters[key].add(param.annotation)
    for argname in argnames:
        parameters = arg_parameters.get(argname)
        annotation = _parameters_to_annotation_name(parameters, {str(i) for i in imports if isinstance(i, str)})
        if annotation is None:
            yield argname
        else:
            yield f'{argname}: {annotation}'

class _AnnotationData(NamedTuple):
    type_name: str
    imports: Set[str]

def _parameters_to_annotation_name(parameters: Optional[Set[Any]], imports: Set[str]) -> Optional[str]:
    if parameters is None:
        return None
    annotations = tuple((annotation for annotation in map(_parameter_to_annotation, parameters) if annotation is not None))
    if not annotations:
        return None
    if len(annotations) == 1:
        type_name, new_imports = annotations[0]
        imports.update(new_imports)
        return type_name
    joined = _join_generics(('typing.Union', {'typing'}), annotations)
    if joined is None:
        return None
    imports.update(joined.imports)
    return joined.type_name

def _join_generics(
    origin_type_data: Optional[Tuple[str, Set[str]]],
    annotations: Iterator[Optional[_AnnotationData]]
) -> Optional[_AnnotationData]:
    if origin_type_data is None:
        return None
    if origin_type_data is not None and origin_type_data[0] == 'typing.Optional':
        annotations = (annotation for annotation in annotations if annotation is None or annotation.type_name != 'None')
    origin_type, imports = origin_type_data
    joined = _join_argument_annotations(annotations)
    if joined is None or not joined[0]:
        return None
    arg_types, new_imports = joined
    imports.update(new_imports)
    return _AnnotationData(f'{origin_type}[{", ".join(arg_types)}]', imports)

def _join_argument_annotations(
    annotations: TypingIterable[Optional[_AnnotationData]]
) -> Optional[Tuple[List[str], Set[str]]]:
    imports: Set[str] = set()
    arg_types: List[str] = []
    for annotation in annotations:
        if annotation is None:
            return None
        arg_types.append(annotation.type_name)
        imports.update(annotation.imports)
    return (arg_types, imports)

def _parameter_to_annotation(parameter: Any) -> Optional[_AnnotationData]:
    if isinstance(parameter, str):
        return None
    if isinstance(parameter, ForwardRef):
        forwarded_value = parameter.__forward_value__
        if forwarded_value is None:
            return None
        return _parameter_to_annotation(forwarded_value)
    if isinstance(parameter, list):
        joined = _join_argument_annotations((_parameter_to_annotation(param) for param in parameter))
        if joined is None:
            return None
        arg_type_names, new_imports = joined
        return _AnnotationData(f'[{", ".join(arg_type_names)}]', new_imports)
    if isinstance(parameter, type):
        if parameter.__module__ == 'builtins':
            return _AnnotationData('None' if parameter.__name__ == 'NoneType' else parameter.__name__, set())
        type_name = _get_qualname(parameter, include_module=True)
    elif hasattr(parameter, '__module__') and hasattr(parameter, '__name__'):
        type_name = _get_qualname(parameter, include_module=True)
    else:
        type_name = str(parameter)
    if type_name.startswith('hypothesis.strategies.'):
        return _AnnotationData(type_name.replace('hypothesis.strategies', 'st'), set())
    origin_type = get_origin(parameter)
    if origin_type is None or origin_type == parameter:
        return _AnnotationData(type_name, set(type_name.rsplit('.', maxsplit=1)[:-1]))
    arg_types = get_args(parameter)
    if {type(a) for a in arg_types} == {TypeVar}:
        arg_types = ()
    if type_name.startswith('typing.'):
        try:
            new_type_name = type_name[:type_name.index('[')]
        except ValueError:
            new_type_name = type_name
        origin_annotation = _AnnotationData(new_type_name, {'typing'})
    else:
        origin_annotation = _parameter_to_annotation(origin_type)
    if arg_types:
        return _join_generics(origin_annotation, (_parameter_to_annotation(arg_type) for arg_type in arg_types))
    return origin_annotation

def _are_annotations_used(*functions: Callable[..., Any]) -> bool:
    for function in functions:
        try:
            params = get_signature(function).parameters.values()
        except Exception:
            pass
        else:
            if any((param.annotation != inspect.Parameter.empty for param in params)):
                return True
    return False

def _make_test(imports: Set[Union[str, Tuple[str, str]]], body: str) -> str:
    body = body.replace('builtins.', '').replace('__main__.', '')
    imports |= {('hypothesis', 'given'), ('hypothesis', 'strategies as st')}
    if '        reject()\n' in body:
        imports.add(('hypothesis', 'reject'))
    do_not_import = {'builtins', '__main__', 'hypothesis.strategies'}
    direct = {f'import {i}' for i in imports if isinstance(i, str) and i not in do_not_import}
    from_imports: DefaultDict[str, Set[str]] = defaultdict(set)
    for module, name in {i for i in imports if isinstance(i, tuple)}:
        if not (module.startswith('hypothesis.strategies') and name in st.__all__):
            from_imports[module].add(name)
    from_ = {f'from {module} import {", ".join(sorted(names))}' for module, names in from_imports.items() if module not in do_not_import}
    header = IMPORT_SECTION.format(imports='\n'.join(sorted(direct) + sorted(from_)))
    nothings = body.count('st.nothing()')
    if nothings == 1:
        header += '# TODO: replace st.nothing() with an appropriate strategy\n\n'
    elif nothings >= 1:
        header += '# TODO: replace st.nothing() with appropriate strategies\n\n'
    return black.format_str(header + body, mode=black.FileMode())

def _is_probably_ufunc(obj: Any) -> bool:
    has_attributes = 'nin nout nargs ntypes types identity signature'.split()
    return callable(obj) and all((hasattr(obj, name) for name in has_attributes))

ROUNDTRIP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('write(.+)', 'read{}'),
    ('save(.+)', 'load{}'),
    ('dump(.+)', 'load{}'),
    ('to(.+)', 'from{}'),
    ('(.*)en(.+)', '{}de{}'),
    ('(.+)', 'de{}'),
    ('(?!safe)(.+)', 'un{}'),
    ('(.+)2(.+?)(_.+)?', '{1}2{0}{2}'),
    ('(.+)_to_(.+)', '{1}_to_{0}'),
    ('(inet|if)_(.+)to(.+)', '{0}_{2}to{1}'),
    ('(\\w)to(\\w)(.+)', '{1}to{0}{2}'),
    ('send(.+)', 'recv{}'),
    ('send(.+)', 'receive{}')
)

def _get_testable_functions(thing: Union[Callable[..., Any], types.ModuleType]) -> Dict[str, Callable[..., Any]]:
    by_name: Dict[str, Callable[..., Any]] = {}
    if callable(thing):
        funcs: List[Callable[..., Any]] = [thing]
    elif isinstance(thing, types.ModuleType):
        if hasattr(thing, '__all__'):
            funcs = [getattr(thing, name, None) for name in thing.__all__]  # type: ignore
        elif hasattr(thing, '__package__'):
            pkg = thing.__package__
            funcs = [v for k, v in vars(thing).items() if callable(v) and (not is_mock(v)) and (not pkg or getattr(v, '__module__', pkg).startswith(pkg)) and (not k.startswith('_'))]
            if pkg and any((getattr(f, '__module__', pkg) == pkg for f in funcs)):
                funcs = [f for f in funcs if getattr(f, '__module__', pkg) == pkg]
    else:
        raise InvalidArgument(f"Can't test non-module non-callable {thing!r}")
    for f in list(funcs):
        if inspect.isclass(f):
            funcs += [v.__get__(f) for k, v in vars(f).items() if hasattr(v, '__func__') and (not is_mock(v)) and (not k.startswith('_'))]
    for f in funcs:
        try:
            if not is_mock(f) and callable(f) and _get_params(f) and (not isinstance(f, enum.EnumMeta)):
                if getattr(thing, '__name__', None):
                    if inspect.isclass(thing):
                        KNOWN_FUNCTION_LOCATIONS[f] = _get_module_helper(thing)
                    elif isinstance(thing, types.ModuleType):
                        KNOWN_FUNCTION_LOCATIONS[f] = thing.__name__
                try:
                    _get_params(f)
                    by_name[_get_qualname(f, include_module=True)] = f
                except Exception:
                    pass
        except (TypeError, ValueError):
            pass
    return by_name

def magic(
    *modules_or_functions: Union[Callable[..., Any], types.ModuleType],
    except_: Union[type[Exception], Tuple[type[Exception], ...]] = (),
    style: str = 'pytest',
    annotate: Optional[bool] = None
) -> str:
    except_ = _check_except(except_)
    _check_style(style)
    if not modules_or_functions:
        raise InvalidArgument('Must pass at least one function or module to test.')
    parts: List[str] = []
    by_name: Dict[str, Callable[..., Any]] = {}
    imports: Set[Union[str, Tuple[str, str]]] = set()
    for thing in modules_or_functions:
        by_name.update(_get_testable_functions(thing))
        if not _get_testable_functions(thing) and isinstance(thing, types.ModuleType):
            msg = f'# Found no testable functions in {thing.__name__} (from {thing.__file__!r})'
            mods: List[str] = []
            for k in sorted(sys.modules, key=lambda n: len(n)):
                if k.startswith(f'{thing.__name__}.') and '._' not in k.removeprefix(thing.__name__) and (not k.startswith(tuple(f'{m}.' for m in mods))) and _get_testable_functions(sys.modules[k]):
                    mods.append(k)
            if mods:
                msg += f'\n# Try writing tests for submodules, e.g. by using:\n#     hypothesis write {" ".join(sorted(mods))}'
            parts.append(msg)
    if not by_name:
        return '\n\n'.join(parts)
    if annotate is None:
        annotate = _are_annotations_used(*list(by_name.values()))
    def make_(how: Callable[..., Tuple[ImportSet, str]], *args: Any, **kwargs: Any) -> None:
        imp, body = how(*args, **kwargs, except_=except_, style=style)
        imports.update(imp)
        parts.append(body)
    for writename, readname in ROUNDTRIP_PAIRS:
        for name in sorted(by_name):
            match = re.fullmatch(writename, name.split('.')[-1])
            if match:
                inverse_name = readname.format(*match.groups())
                for other in sorted((n for n in by_name if n.split('.')[-1] == inverse_name)):
                    make_(_make_roundtrip_body, (by_name.pop(name), by_name.pop(other)), annotate=annotate)
                    break
                else:
                    try:
                        other_func = getattr(sys.modules[_get_module(by_name[name])], inverse_name)
                        _get_params(other_func)
                    except Exception:
                        pass
                    else:
                        make_(_make_roundtrip_body, (by_name.pop(name), other_func), annotate=annotate)
    names: DefaultDict[str, List[Callable[..., Any]]] = defaultdict(list)
    for _, f in sorted(by_name.items()):
        names[_get_qualname(f)].append(f)
    for group in names.values():
        if len(group) >= 2 and len({frozenset(_get_params(f)) for f in group}) == 1:
            sentinel = object()
            returns = {get_type_hints(f).get('return', sentinel) for f in group}
            if len(returns - {sentinel}) <= 1:
                make_(_make_equiv_body, group, annotate=annotate)
                for f in group:
                    by_name.pop(_get_qualname(f, include_module=True))
    for name, func in sorted(by_name.items()):
        hints = get_type_hints(func)
        hints.pop('return', None)
        params = _get_params(func)
        if len(hints) == len(params) == 2 or (
            _get_module(func) == 'operator' and 'item' not in func.__name__ and (tuple(params) in [('a', 'b'), ('x', 'y')])
        ):
            a, b = list(hints.values()) or [Any, Any]
            arg1, arg2 = list(params)
            if a == b and len(arg1.name) == len(arg2.name) <= 3:
                known = {'mul': 'add', 'matmul': 'add', 'or_': 'and_', 'and_': 'or_'}.get(func.__name__, '')
                distributes_over = getattr(sys.modules[_get_module(func)], known, None)
                make_(_make_binop_body, func, commutative=func.__name__ != 'matmul', distributes_over=distributes_over, annotate=annotate)
                del by_name[name]
    if 'numpy' in sys.modules:
        for name, func in sorted(by_name.items()):
            if _is_probably_ufunc(func):
                make_(_make_ufunc_body, func, annotate=annotate)
                del by_name[name]
    for _, f in sorted(by_name.items()):
        make_(_make_test_body, f, test_body=_write_call(f, except_=except_), ghost='fuzz', annotate=annotate)
    return _make_test(imports, '\n'.join(parts))

def fuzz(
    func: Callable[..., Any],
    *,
    except_: Union[type[Exception], Tuple[type[Exception], ...]] = (),
    style: str = 'pytest',
    annotate: Optional[bool] = None
) -> str:
    if not callable(func):
        raise InvalidArgument(f'Got non-callable func={func!r}')
    except_ = _check_except(except_)
    _check_style(style)
    if annotate is None:
        annotate = _are_annotations_used(func)
    imports, body = _make_test_body(func, test_body=_write_call(func, except_=except_), except_=except_, ghost='fuzz', style=style, annotate=annotate)
    return _make_test(imports, body)

def idempotent(
    func: Callable[..., Any],
    *,
    except_: Union[type[Exception], Tuple[type[Exception], ...]] = (),
    style: str = 'pytest',
    annotate: Optional[bool] = None
) -> str:
    if not callable(func):
        raise InvalidArgument(f'Got non-callable func={func!r}')
    except_ = _check_except(except_)
    _check_style(style)
    if annotate is None:
        annotate = _are_annotations_used(func)
    test_call = f'{_write_call(func, except_=except_)}'
    body_expr = f'result = {test_call}\nrepeat = {_write_call(func, "result", except_=except_)}'
    assertions = _assert_eq(style, 'result', 'repeat')
    imports, body = _make_test_body(func, test_body=body_expr, except_=except_, assertions=assertions, ghost='idempotent', style=style, annotate=annotate)
    return _make_test(imports, body)

def _make_roundtrip_body(
    funcs: Tuple[Callable[..., Any], ...],
    except_: Union[type[Exception], Tuple[type[Exception], ...]],
    style: str,
    annotate: bool
) -> Tuple[ImportSet, str]:
    first_param = next(iter(_get_params(funcs[0])))
    test_lines = [
        _write_call(funcs[0], assign='value0', except_=except_)
    ] + [
        _write_call(f, f'value{i}', assign=f'value{i + 1}', except_=except_)
        for i, f in enumerate(funcs[1:])
    ]
    return _make_test_body(
        *funcs,
        test_body='\n'.join(test_lines),
        except_=except_,
        assertions=_assert_eq(style, first_param.name, f'value{len(funcs) - 1}'),
        ghost='roundtrip',
        style=style,
        annotate=annotate
    )

def roundtrip(
    *funcs: Callable[..., Any],
    except_: Union[type[Exception], Tuple[type[Exception], ...]] = (),
    style: str = 'pytest',
    annotate: Optional[bool] = None
) -> str:
    if not funcs:
        raise InvalidArgument('Round-trip of zero functions is meaningless.')
    for i, f in enumerate(funcs):
        if not callable(f):
            raise InvalidArgument(f'Got non-callable funcs[{i}]={f!r}')
    except_ = _check_except(except_)
    _check_style(style)
    if annotate is None:
        annotate = _are_annotations_used(*funcs)
    imp, src = _make_roundtrip_body(funcs, except_, style, annotate)  # type: ignore
    return _make_test(imp, src)

def _get_varnames(funcs: Sequence[Callable[..., Any]]) -> List[str]:
    var_names = [f'result_{f.__name__}' for f in funcs]
    if len(set(var_names)) < len(var_names):
        var_names = [f'result_{f.__name__}_{_get_module(f)}' for f in funcs]
    if len(set(var_names)) < len(var_names):
        var_names = [f'result_{i}_{f.__name__}' for i, f in enumerate(funcs)]
    return var_names

def _make_equiv_body(
    funcs: Sequence[Callable[..., Any]],
    except_: Union[type[Exception], Tuple[type[Exception], ...]],
    style: str,
    annotate: bool
) -> Tuple[ImportSet, str]:
    var_names = _get_varnames(funcs)
    test_lines = [_write_call(f, assign=vname, except_=except_) for vname, f in zip(var_names, funcs)]
    assertions = '\n'.join((_assert_eq(style, var_names[0], vname) for vname in var_names[1:]))
    return _make_test_body(*funcs, test_body='\n'.join(test_lines), except_=except_, assertions=assertions, ghost='equivalent', style=style, annotate=annotate)

EQUIV_FIRST_BLOCK: str = (
    "\ntry:\n{}\n    exc_type = None\n    target(1, label=\"input was valid\")\n{}except Exception as exc:\n    exc_type = type(exc)\n"
).strip()
EQUIV_CHECK_BLOCK: str = (
    "\nif exc_type:\n    with {ctx}(exc_type):\n{check_raises}\nelse:\n{call}\n{compare}\n"
).rstrip()

def _make_equiv_errors_body(
    funcs: Sequence[Callable[..., Any]],
    except_: Union[type[Exception], Tuple[type[Exception], ...]],
    style: str,
    annotate: bool
) -> Tuple[ImportSet, str]:
    var_names = _get_varnames(funcs)
    first, *rest = funcs
    first_call = _write_call(first, assign=var_names[0], except_=except_)
    extra_imports, suppress = _exception_string(except_)
    extra_imports.add(('hypothesis', 'target'))
    catch = f'except {suppress}:\n    reject()\n' if suppress else ''
    test_lines = [EQUIV_FIRST_BLOCK.format(indent(first_call, prefix='    '), catch)]
    for vname, f in zip(var_names[1:], rest):
        if style == 'pytest':
            ctx = 'pytest.raises'
            extra_imports.add('pytest')
        else:
            assert style == 'unittest'
            ctx = 'self.assertRaises'
        block = EQUIV_CHECK_BLOCK.format(
            ctx=ctx,
            check_raises=indent(_write_call(f, except_=()), '        '),
            call=indent(_write_call(f, assign=vname, except_=()), '    '),
            compare=indent(_assert_eq(style, var_names[0], vname), '    ')
        )
        test_lines.append(block)
    imports, source_code = _make_test_body(*funcs, test_body='\n'.join(test_lines), except_=(), ghost='equivalent', style=style, annotate=annotate)
    return (imports | extra_imports, source_code)

def equivalent(
    *funcs: Callable[..., Any],
    allow_same_errors: bool = False,
    except_: Union[type[Exception], Tuple[type[Exception], ...]] = (),
    style: str = 'pytest',
    annotate: Optional[bool] = None
) -> str:
    if len(funcs) < 2:
        raise InvalidArgument('Need at least two functions to compare.')
    for i, f in enumerate(funcs):
        if not callable(f):
            raise InvalidArgument(f'Got non-callable funcs[{i}]={f!r}')
    check_type(bool, allow_same_errors, 'allow_same_errors')
    except_ = _check_except(except_)
    _check_style(style)
    if annotate is None:
        annotate = _are_annotations_used(*funcs)
    if allow_same_errors and (not any((issubclass(Exception, ex) for ex in except_))):
        imports, source_code = _make_equiv_errors_body(funcs, except_, style, annotate)
    else:
        imports, source_code = _make_equiv_body(funcs, except_, style, annotate)
    return _make_test(imports, source_code)

X = TypeVar('X')
Y = TypeVar('Y')

def binary_operation(
    func: Callable[..., Any],
    *,
    associative: bool = True,
    commutative: bool = True,
    identity: Any = ...,
    distributes_over: Optional[Callable[..., Any]] = None,
    except_: Union[type[Exception], Tuple[type[Exception], ...]] = (),
    style: str = 'pytest',
    annotate: Optional[bool] = None
) -> str:
    if not callable(func):
        raise InvalidArgument(f'Got non-callable func={func!r}')
    except_ = _check_except(except_)
    _check_style(style)
    check_type(bool, associative, 'associative')
    check_type(bool, commutative, 'commutative')
    if distributes_over is not None and (not callable(distributes_over)):
        raise InvalidArgument(f'distributes_over={distributes_over!r} must be an operation which distributes over {func.__name__}')
    if not any([associative, commutative, identity, distributes_over]):
        raise InvalidArgument('You must select at least one property of the binary operation to test.')
    if annotate is None:
        annotate = _are_annotations_used(func)
    imports, body = _make_binop_body(
        func,
        associative=associative,
        commutative=commutative,
        identity=identity,
        distributes_over=distributes_over,
        except_=except_,
        style=style,
        annotate=annotate
    )
    return _make_test(imports, body)

def _make_binop_body(
    func: Callable[..., Any],
    *,
    associative: bool,
    commutative: bool,
    identity: Any,
    distributes_over: Optional[Callable[..., Any]],
    except_: Union[type[Exception], Tuple[type[Exception], ...]],
    style: str,
    annotate: bool
) -> Tuple[ImportSet, str]:
    strategies = _get_strategies(func)
    params = list(_get_params(func).values())
    if not params:
        raise InvalidArgument("Function has no parameters.")
    operands = strategies.pop(params[0].name)
    b = None
    if len(params) > 1:
        b = strategies.pop(params[1].name)
    if b is not None and repr(operands) != repr(b):
        operands |= b
    operands_name = func.__name__ + '_operands'
    all_imports: Set[Union[str, Tuple[str, str]]] = set()
    parts: List[str] = []
    def maker(sub_property: str, args: str, body: str, right: Optional[str] = None) -> None:
        if right is None:
            assertions_local = ''
        else:
            body = f'{body}\n{right}'
            assertions_local = _assert_eq(style, 'left', 'right')
        imp, body_str = _make_test_body(
            func,
            test_body=body,
            ghost=sub_property + '_binary_operation',
            except_=except_,
            assertions=assertions_local,
            style=style,
            given_strategies={**strategies, **{n: st.nothing() for n in args}},
            annotate=annotate
        )
        all_imports.update(imp)
        if style == 'unittest':
            body_str = body_str[body_str.index('(unittest.TestCase):\n') + len('(unittest.TestCase):\n') + 1:]
        parts.append(body_str)
    if associative:
        left_expr = _write_call(func, 'a', _write_call(func, 'b', 'c'), assign='left')
        right_expr = _write_call(func, _write_call(func, 'a', 'b'), 'c', assign='right')
        maker('associative', 'abc', left_expr, right_expr)
    if commutative:
        left_expr = _write_call(func, 'a', 'b', assign='left')
        right_expr = _write_call(func, 'b', 'a', assign='right')
        maker('commutative', 'ab', left_expr, right_expr)
    if identity is not None:
        if identity is ...:
            try:
                identity = find(operands, lambda x: True, settings=_quietly_settings)
            except Exception:
                identity = 'identity element here'
        try:
            compile(repr(identity), '<string>', 'exec')
        except SyntaxError:
            identity = repr(identity)
        identity_parts = [
            f'identity = {identity!r}',
            _assert_eq(style, 'a', _write_call(func, 'a', 'identity')),
            _assert_eq(style, 'a', _write_call(func, 'identity', 'a'))
        ]
        maker('identity', 'a', '\n'.join(identity_parts))
    if distributes_over:
        do = distributes_over
        dist_parts = [
            _write_call(func, 'a', _write_call(do, 'b', 'c'), assign='left'),
            _write_call(do, _write_call(func, 'a', 'b'), _write_call(func, 'a', 'c'), assign='ldist'),
            _assert_eq(style, 'ldist', 'left'),
            '\n',
            _write_call(func, _write_call(do, 'a', 'b'), 'c', assign='right'),
            _write_call(do, _write_call(func, 'a', 'c'), _write_call(func, 'b', 'c'), assign='rdist'),
            _assert_eq(style, 'rdist', 'right')
        ]
        maker(do.__name__ + '_distributes_over', 'abc', '\n'.join(dist_parts))
    _, operands_repr = _valid_syntax_repr(operands)
    operands_repr = _st_strategy_names(operands_repr)
    classdef = ''
    if style == 'unittest':
        classdef = f'class TestBinaryOperation{func.__name__}(unittest.TestCase):\n    '
    return (all_imports, classdef + f'{operands_name} = {operands_repr}\n' + '\n'.join(parts))

def ufunc(
    func: Callable[..., Any],
    *,
    except_: Union[type[Exception], Tuple[type[Exception], ...]] = (),
    style: str = 'pytest',
    annotate: Optional[bool] = None
) -> str:
    if not _is_probably_ufunc(func):
        raise InvalidArgument(f'func={func!r} does not seem to be a ufunc')
    except_ = _check_except(except_)
    _check_style(style)
    if annotate is None:
        annotate = _are_annotations_used(func)
    return _make_test(*_make_ufunc_body(func, except_=except_, style=style, annotate=annotate))

def _make_ufunc_body(
    func: Callable[..., Any],
    *,
    except_: Union[type[Exception], Tuple[type[Exception], ...]],
    style: str,
    annotate: bool
) -> Tuple[ImportSet, str]:
    import hypothesis.extra.numpy as npst  # type: ignore
    if func.signature is None:
        shapes = npst.mutually_broadcastable_shapes(num_shapes=func.nin)
    else:
        shapes = npst.mutually_broadcastable_shapes(signature=func.signature)
    shapes.function.__module__ = npst.__name__
    body = dedent(f"""\
        input_shapes, expected_shape = shapes
        input_dtypes, expected_dtype = types.split("->")
        array_strats = [
            arrays(dtype=dtp, shape=shp, elements={{"allow_nan": True}})
            for dtp, shp in zip(input_dtypes, input_shapes)
        ]

        {', '.join(ascii_lowercase[:func.nin])} = data.draw(st.tuples(*array_strats))
        result = {_write_call(func, *ascii_lowercase[:func.nin], except_=except_)}
    """).strip()
    assertions = f'\n{_assert_eq(style, "result.shape", "expected_shape")}\n{_assert_eq(style, "result.dtype.char", "expected_dtype")}'
    qname = _get_qualname(func, include_module=True)
    obj_sigs = ['O' in sig for sig in func.types]
    if all(obj_sigs) or not any(obj_sigs):
        types_str = f'sampled_from({qname}.types)'
    else:
        types_str = f"sampled_from([sig for sig in {qname}.types if 'O' not in sig])"
    return _make_test_body(
        func,
        test_body=f"{body}\n{assertions}",
        except_=except_,
        ghost='ufunc' if func.signature is None else 'gufunc',
        style=style,
        given_strategies={'data': st.data(), 'shapes': shapes, 'types': types_str},
        imports={('hypothesis.extra.numpy', 'arrays')},
        annotate=annotate
    ) 
# End of file.
