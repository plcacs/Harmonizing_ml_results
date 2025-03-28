from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union
from collections import OrderedDict, defaultdict
from collections.abc import Iterable as IterableABC, Mapping
from itertools import permutations, zip_longest
from keyword import iskeyword as _iskeyword
from string import ascii_lowercase
from textwrap import dedent, indent
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

IMPORT_SECTION = """
# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

{imports}
"""

TEMPLATE = """
@given({given_args})
def test_{test_kind}_{func_name}({arg_names}){return_annotation}:
{test_body}
"""

SUPPRESS_BLOCK = """
try:
{test_body}
except {exceptions}:
    reject()
""".strip()

Except = Union[Type[Exception], Tuple[Type[Exception], ...]
ImportSet = Set[Union[str, Tuple[str, str]]]
_quietly_settings = settings(
    database=None,
    deadline=None,
    derandomize=True,
    verbosity=Verbosity.quiet,
)

def _dedupe_exceptions(exc: Tuple[Type[Exception], ...]) -> Tuple[Type[Exception], ...]:
    uniques = list(exc)
    for a, b in permutations(exc, 2):
        if a in uniques and issubclass(a, b):
            uniques.remove(a)
    return tuple(sorted(uniques, key=lambda e: e.__name__))

def _check_except(except_: Except) -> Tuple[Type[Exception], ...]:
    if isinstance(except_, tuple):
        for i, e in enumerate(except_):
            if not isinstance(e, type) or not issubclass(e, Exception):
                raise InvalidArgument(
                    f"Expected an Exception but got except_[{i}]={e!r}"
                    f" (type={_get_qualname(type(e))})"
                )
        return except_
    if not isinstance(except_, type) or not issubclass(except_, Exception):
        raise InvalidArgument(
            "Expected an Exception or tuple of exceptions, but got except_="
            f"{except_!r} (type={_get_qualname(type(except_))})"
        )
    return (except_,)

def _exception_string(except_: Tuple[Type[Exception], ...]) -> Tuple[ImportSet, str]:
    if not except_:
        return set(), ""
    exceptions = []
    imports: ImportSet = set()
    for ex in _dedupe_exceptions(except_):
        if ex.__qualname__ in dir(builtins):
            exceptions.append(ex.__qualname__)
        else:
            imports.add(ex.__module__)
            exceptions.append(_get_qualname(ex, include_module=True))
    return imports, (
        "(" + ", ".join(exceptions) + ")" if len(exceptions) > 1 else exceptions[0]
    )

def _check_style(style: str) -> None:
    if style not in ("pytest", "unittest"):
        raise InvalidArgument(f"Valid styles are 'pytest' or 'unittest', got {style!r}")

def _exceptions_from_docstring(doc: str) -> Tuple[Type[Exception], ...]:
    raises = []
    for excname in re.compile(r"\:raises\s+(\w+)\:", re.MULTILINE).findall(doc):
        exc_type = getattr(builtins, excname, None)
        if isinstance(exc_type, type) and issubclass(exc_type, Exception):
            raises.append(exc_type)
    return tuple(_dedupe_exceptions(tuple(raises)))

def _type_from_doc_fragment(token: str) -> Optional[type]:
    if token == "integer":
        return int
    if "numpy" in sys.modules:
        if re.fullmatch(r"[Aa]rray[-_ ]?like", token):
            return sys.modules["numpy"].ndarray
        elif token == "dtype":
            return sys.modules["numpy"].dtype
    coll_match = re.fullmatch(r"(\w+) of (\w+)", token)
    if coll_match is not None:
        coll_token, elem_token = coll_match.groups()
        elems = _type_from_doc_fragment(elem_token)
        if elems is None and elem_token.endswith("s"):
            elems = _type_from_doc_fragment(elem_token[:-1])
        if elems is not None and coll_token in ("list", "sequence", "collection"):
            return list[elems]  # type: ignore
        return _type_from_doc_fragment(coll_token)
    if "." not in token:
        return getattr(builtins, token, None)
    mod, name = token.rsplit(".", maxsplit=1)
    return getattr(sys.modules.get(mod, None), name, None)

def _strip_typevars(type_):
    with contextlib.suppress(Exception):
        if {type(a) for a in get_args(type_)} == {TypeVar}:
            return get_origin(type_)
    return type_

def _strategy_for(param: inspect.Parameter, docstring: str) -> st.SearchStrategy:
    for pattern in (
        rf"^\s*\:type\s+{param.name}\:\s+(.+)",  # RST-style
        rf"^\s*{param.name} \((.+)\):",  # Google-style
        rf"^\s*{param.name} \: (.+)",  # Numpy-style
    ):
        match = re.search(pattern, docstring, flags=re.MULTILINE)
        if match is None:
            continue
        doc_type = match.group(1)
        if doc_type.endswith(", optional"):
            doc_type = doc_type[: -len(", optional")]
        doc_type = doc_type.strip("}{")
        elements = []
        types = []
        for token in re.split(r",? +or +| *, *", doc_type):
            for prefix in ("default ", "python "):
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
        if (
            param.default is not inspect.Parameter.empty
            and param.default not in elements
            and not isinstance(
                param.default, tuple(t for t in types if isinstance(t, type))
            )
        ):
            with contextlib.suppress(SyntaxError):
                compile(repr(st.just(param.default)), "<string>", "eval")
                elements.insert(0, param.default)
        if elements or types:
            return (st.sampled_from(elements) if elements else st.nothing()) | (
                st.one_of(*map(st.from_type, types)) if types else st.nothing()
            )

    if isinstance(param.default, bool):
        return st.booleans()
    if isinstance(param.default, enum.Enum):
        return st.sampled_from(type(param.default))
    if param.default is not inspect.Parameter.empty:
        return st.just(param.default)
    return _guess_strategy_by_argname(name=param.name.lower())

BOOL_NAMES = (
    "keepdims", "verbose", "debug", "force", "train", "training", "trainable", "bias",
    "shuffle", "show", "load", "pretrained", "save", "overwrite", "normalize",
    "reverse", "success", "enabled", "strict", "copy", "quiet", "required", "inplace",
    "recursive", "enable", "active", "create", "validate", "refresh", "use_bias",
)
POSITIVE_INTEGER_NAMES = (
    "width", "size", "length", "limit", "idx", "stride", "epoch", "epochs", "depth",
    "pid", "steps", "iteration", "iterations", "vocab_size", "ttl", "count",
)
FLOAT_NAMES = (
    "real", "imag", "alpha", "theta", "beta", "sigma", "gamma", "angle", "reward",
    "tau", "temperature",
)
STRING_NAMES = (
    "text", "txt", "password", "label", "prefix", "suffix", "desc", "description",
    "str", "pattern", "subject", "reason", "comment", "prompt", "sentence", "sep",
)

def _guess_strategy_by_argname(name: str) -> st.SearchStrategy:
    if name in ("function", "func", "f"):
        return st.functions()
    if name in ("pred", "predicate"):
        return st.functions(returns=st.booleans(), pure=True)
    if name in ("iterable",):
        return st.iterables(st.integers()) | st.iterables(st.text())
    if name in ("list", "lst", "ls"):
        return st.lists(st.nothing())
    if name in ("object",):
        return st.builds(object)
    if "uuid" in name:
        return st.uuids().map(str)

    if name.startswith("is_") or name in BOOL_NAMES:
        return st.booleans()

    if name in ("amount", "threshold", "number", "num"):
        return st.integers() | st.floats()

    if name in ("port",):
        return st.integers(0, 2**16 - 1)
    if (
        name.endswith("_size")
        or (name.endswith("size") and "_" not in name)
        or re.fullmatch(r"n(um)?_[a-z_]*s", name)
        or name in POSITIVE_INTEGER_NAMES
    ):
        return st.integers(min_value=0)
    if name in ("offset", "seed", "dim", "total", "priority"):
        return st.integers()

    if name in ("learning_rate", "dropout", "dropout_rate", "epsilon", "eps", "prob"):
        return st.floats(0, 1)
    if name in ("lat", "latitude"):
        return st.floats(-90, 90)
    if name in ("lon", "longitude"):
        return st.floats(-180, 180)
    if name in ("radius", "tol", "tolerance", "rate"):
        return st.floats(min_value=0)
    if name in FLOAT_NAMES:
        return st.floats()

    if name in ("host", "hostname"):
        return domains()
    if name in ("email",):
        return st.emails()
    if name in ("word", "slug", "api_key"):
        return st.from_regex(r"\w+", fullmatch=True)
    if name in ("char", "character"):
        return st.characters()

    if (
        "file" in name
        or "path" in name
        or name.endswith("_dir")
        or name in ("fname", "dir", "dirname", "directory", "folder")
    ):
        return st.nothing()

    if (
        name.endswith(("_name", "label"))
        or (name.endswith("name") and "_" not in name)
        or ("string" in name and "as" not in name)
        or name in STRING_NAMES
    ):
        return st.text()

    if re.fullmatch(r"\w*[^s]s", name):
        elems = _guess_strategy_by_argname(name[:-1])
        if not elems.is_empty:
            return st.lists(elems)

    return st.nothing()

def _get_params_builtin_fn(func: Callable) -> List[inspect.Parameter]:
    if (
        isinstance(func, (types.BuiltinFunctionType, types.BuiltinMethodType))
        and hasattr(func, "__doc__")
        and isinstance(func.__doc__, str)
    ):
        match = re.match(rf"^{func.__name__}\((.+?)\)", func.__doc__)
        if match is None:
            return []
        args = match.group(1).replace("[", "").replace("]", "")
        params = []
        kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_ONLY
        for arg in args.split(", "):
            arg, *_ = arg.partition("=")
            arg = arg.strip()
            if arg == "/":
                kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                continue
            if arg.startswith("*") or arg == "...":
                kind = inspect.Parameter.KEYWORD_ONLY
                continue  # we omit *varargs, if there are any
            if _iskeyword(arg.lstrip("*")) or not arg.lstrip("*").isidentifier():
                break  # skip all subsequent params if this name is invalid
            params.append(inspect.Parameter(name=arg, kind=kind))
        return params
    return []

def _get_params_ufunc(func: Callable) -> List[inspect.Parameter]:
    if _is_probably_ufunc(func):
        return [
            inspect.Parameter(name=name, kind=inspect.Parameter.POSITIONAL_ONLY)
            for name in ascii_lowercase[: func.nin]  # type: ignore
        ]
    return []

def _get_params(func: Callable) -> Dict[str, inspect.Parameter]:
    try:
        params = list(get_signature(func).parameters.values())
    except Exception:
        if params := _get_params_ufunc(func):
            pass
        elif params := _get_params_builtin_fn(func):
            pass
        else:
            raise
    else:
        P = inspect.Parameter
        placeholder = [("args", P.VAR_POSITIONAL), ("kwargs", P.VAR_KEYWORD)]
        if [(p.name, p.kind) for p in params] == placeholder:
            params = _get_params_ufunc(func) or _get_params_builtin_fn(func) or params
    return _params_to_dict(params)

def _params_to_dict(
    params: Iterable[inspect.Parameter],
) -> Dict[str, inspect.Parameter]:
    var_param_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return OrderedDict((p.name, p) for p in params if p.kind not in var_param_kinds)

@contextlib.contextmanager
def _with_any_registered():
    if Any in _global_type_lookup:
        yield
    else:
        try:
            _global_type_lookup[Any] = st.builds(object)
            yield
        finally:
            del _global_type_lookup[Any]
            st.from_type.__clear_cache()

def _get_strategies(
    *funcs: Callable, pass_result_to_next_func: bool = False
) -> Dict[str, st.SearchStrategy]:
    assert funcs, "Must pass at least one function"
    given_strategies: Dict[str, st.SearchStrategy] = {}
    for i, f in enumerate(funcs):
        params = _get_params(f)
        if pass_result_to_next_func and i >= 1:
            del params[next(iter(params))]
        hints = get_type_hints(f)
        docstring = getattr(f, "__doc__", None) or ""
        builder_args = {
            k: ... if k in hints else _strategy_for(v, docstring)
            for k, v in params.items()
        }
        with _with_any_registered():
            strat = st.builds(f, **builder_args).wrapped_strategy  # type: ignore

        if strat.args:
            raise NotImplementedError("Expected to pass everything as kwargs")

        for k, v in strat.kwargs.items():
            if _valid_syntax_repr(v)[1] == "nothing()" and k in hints:
                v = LazyStrategy(st.from_type, (hints[k],), {})
            if k in given_strategies:
                given_strategies[k] |= v
            else:
                given_strategies[k] = v

    if len(funcs) == 1:
        return {name: given_strategies[name] for name in _get_params(f)}
    return dict(sorted(given_strategies.items()))

def _assert_eq(style: str, a: str, b: str) -> str:
    if style == "unittest":
        return f"self.assertEqual({a}, {b})"
    assert style == "pytest"
    if a.isidentifier() and b.isidentifier():
        return f"assert {a} == {b}, ({a}, {b})"
    return f"assert {a} == {b}"

def _imports_for_object(obj):
    if type(obj) is getattr(types, "UnionType", object()):
        return {mod for mod, _ in set().union(*map(_imports_for_object, obj.__args__))}
    if isinstance(obj, (re.Pattern, re.Match)):
        return {"re"}
    if isinstance(obj, st.SearchStrategy):
        return _imports_for_strategy(obj)
    if isinstance(obj, getattr(sys.modules.get("numpy"), "dtype", ())):
        return {("numpy", "dtype")}
    try:
        if is_generic_type(obj):
            if isinstance(obj, TypeVar):
                return {(obj.__module__, obj.__name__)}
            with contextlib.suppress(Exception):
                return set().union(*map(_imports_for_object, obj.__args__))
        if (not callable(obj)) or obj.__name__ == "<lambda>":
            return set()
        name = _get_qualname(obj).split(".")[0]
        return {(_get_module(obj), name)}
    except Exception:
        return set()

def _imports_for_strategy(strategy):
    if isinstance(strategy, LazyStrategy):
        imports = {
            imp
            for arg in set(strategy._LazyStrategy__args)
            | set(strategy._LazyStrategy__kwargs.values())
            for imp in _imports_for_object(_strip_typevars(arg))
        }
        if re.match(r"from_(type|regex)\(", repr(strategy)):
            return imports
        elif _get_module(strategy.function).startswith("hypothesis.extra."):
            module = _get_module(strategy.function).replace("._array_helpers", ".numpy")
            return {(module, strategy.function.__name__)} | imports

    imports = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallSearchSpaceWarning)
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

def _valid_syntax_repr(strategy):
    if isinstance(strategy, str):
        return set(), strategy
    try:
        if isinstance(strategy, DeferredStrategy):
            strategy = strategy.wrapped_strategy
        if isinstance(strategy, OneOfStrategy):
            seen = set()
            elems = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SmallSearchSpaceWarning)
                strategy.element_strategies  # might warn on first access
            for s in strategy.element_strategies:
                if isinstance(s, SampledFromStrategy) and s.elements == (os.environ,):
                    continue
                if repr(s) not in seen:
                    elems.append(s)
                    seen.add(repr(s))
            strategy = st.one_of(elems or st.nothing())
        if strategy == st.text().wrapped_strategy:
            return set(), "text()"
        if (
            isinstance(strategy, LazyStrategy)
            and strategy.function.__name__ == st.from_type.__name__
            and strategy._LazyStrategy__representation is None
        ):
            strategy._LazyStrategy__args = tuple(
                _strip_typevars(a) for a in strategy._LazyStrategy__args
            )
        r = (
            repr(strategy)
            .replace(".filter(_can_hash)", "")
            .replace("hypothesis.strategies.", "")
        )
        r = re.sub(r"(lambda.*?: )(<unknown>)([,)])", r"\1...\3", r)
        compile(r, "<string>", "eval")
        imports = {i for i in _imports_for_strategy(strategy) if i[1] in r}
        return imports, r
    except (SyntaxError, RecursionError, InvalidArgument):
        return set(), "nothing()"

KNOWN_FUNCTION_LOCATIONS: Dict[object, str] = {}

def _get_module_helper(obj):
    module_name = obj.__module__

    if module_name == "collections.abc":
        return module_name

    dots = [i for i, c in enumerate(module_name) if c == "."] + [None]
    for idx in dots:
        for candidate in (module_name[:idx].lstrip("_"), module_name[:idx]):
            if getattr(sys.modules.get(candidate), obj.__name__, None) is obj:
                KNOWN_FUNCTION_LOCATIONS[obj] = candidate
                return candidate
    return module_name

def _get_module(obj):
    if obj in KNOWN_FUNCTION_LOCATIONS:
        return KNOWN_FUNCTION_LOCATIONS[obj]
    try:
        return _get_module_helper(obj)
    except AttributeError:
        if not _is_probably_ufunc(obj):
            raise
    for module_name in sorted(sys.modules, key=lambda n: tuple(n.split("."))):
        if obj is getattr(sys.modules[module_name], obj.__name__, None):
            KNOWN_FUNCTION_LOCATIONS[obj] = module_name
            return module_name
    raise RuntimeError(f"Could not find module for ufunc {obj.__name__} ({obj!r}")

def _get_qualname(obj: Any, *, include_module: bool = False) -> str:
    qname = getattr(obj, "__qualname__", obj.__name__)
    qname = qname.replace("<", "_").replace(">", "_").replace(" ", "")
    if include_module:
        return _get_module(obj) + "." + qname
    return qname

def _write_call(
    func: Callable, *pass_variables: str, except_: Except = Exception, assign: str = ""
) -> str:
    args = ", ".join(
        (
            (v or p.name)
            if p.kind is inspect.Parameter.POSITIONAL_ONLY
            else f"{p.name}={v or p.name}"
        )
        for v, p in zip_longest(pass_variables, _get_params(func).values())
    )
    call = f"{_get_qualname(func, include_module=True)}({args})"
    if assign:
        call = f"{assign} = {call}"
    raises = _exceptions_from_docstring(getattr(func, "__doc__", "") or "")
    exnames = [ex.__name__ for ex in raises if not issubclass(ex, except_)]
    if not exnames:
        return call
    return SUPPRESS_BLOCK.format(
        test_body=indent(call, prefix="    "),
        exceptions="(" + ", ".join(exnames) + ")" if len(exnames) > 1 else exnames[0],
    )

def _st_strategy_names(s: str) -> str:
    names = "|".join(sorted(st.__all__, key=len, reverse=True))
    return re.sub(pattern=rf"\b(?:{names})\b[^= ]", repl=r"st.\g<0>", string=s)

def _make_test_body(
    *funcs: Callable,
    ghost: str,
    test_body: str,
    except_: Tuple[Type[Exception], ...],
    assertions: str = "",
    style: str,
    given_strategies: Optional[Mapping[str, Union[str, st.SearchStrategy]]] = None,
    imports: Optional[ImportSet] = None,
    annotate: bool,
) -> Tuple[ImportSet, str]:
    imports = (imports or set()) | {_get_module(f) for f in funcs}

    with _with_any_registered():
        given_strategies = given_strategies or _get_strategies(
            *funcs, pass_result_to_next_func=ghost in ("idempotent", "roundtrip")
        )
        reprs = [((k, *_valid_syntax_repr(v))) for k, v in given_strategies.items()]
        imports = imports.union(*(imp for _, imp, _ in reprs))
        given_args = ", ".join(f"{k}={v}" for k, _, v in reprs)
    given_args = _st_strategy_names(given_args)

    if except_:
        imp, exc_string = _exception_string(except_)
        imports.update(imp)
        test_body = SUPPRESS_BLOCK.format(
            test_body=indent(test_body, prefix="    "),
            exceptions=exc_string,
        )

    if assertions:
        test_body = f"{test_body}\n{assertions}"

    argnames = ["self"] if style == "unittest" else []
    if annotate:
        argnames.extend(_annotate_args(given_strategies, funcs, imports))
    else:
        argnames.extend(given_strategies)

    body = TEMPLATE.format(
        given_args=given_args,
        test_kind=ghost,
        func_name="_".join(_get_qualname(f).replace(".", "_") for f in funcs),
        arg_names=", ".join(argnames),
        return_annotation=" -> None" if annotate else "",
        test_body=indent(test_body, prefix="    "),
    )

    if style == "unittest":
        imports.add("unittest")
        body = "class Test{}{}(unittest.TestCase):\n{}".format(
            ghost.title(),
            "".join(_get_qualname(f).replace(".", "").title() for f in funcs),
            indent(body, "    "),
        )

    return imports, body

def _annotate_args(
    argnames: Iterable[str], funcs: Iterable[Callable], imports: ImportSet
) -> Iterable[str]:
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
        annotation = _parameters_to_annotation_name(parameters, imports)
        if annotation is None:
            yield argname
        else:
            yield f"{argname}: {annotation}"

class _AnnotationData(NamedTuple):
    type_name: str
    imports: Set[str]

def _parameters_to_annotation_name(
    parameters: Optional[Iterable[Any]], imports: ImportSet
) -> Optional[str]:
    if parameters is None:
        return None
    annotations = tuple(
        annotation
        for annotation in map(_parameter_to_annotation, parameters)
        if annotation is not None
    )
    if not annotations:
        return None
    if len(annotations) == 1:
        type_name, new_imports = annotations[0]
        imports.update(new_imports)
        return type_name
    joined = _join_generics(("typing.Union", {"typing"}), annotations)
    if joined is None:
        return None
    imports.update(joined.imports)
    return joined.type_name

def _join_generics(
    origin_type_data: Optional[Tuple[str, Set[str]]],
    annotations: Iterable[Optional[_AnnotationData]],
) -> Optional[_AnnotationData]:
    if origin_type_data is None:
        return None

    if origin_type_data is not None and origin_type_data[0] == "typing.Optional":
        annotations = (
            annotation
            for annotation in annotations
            if annotation is None or annotation.type_name != "None"
        )

    origin_type, imports = origin_type_data
    joined = _join_argument_annotations(annotations)
    if joined is None or not joined[0]:
        return None

    arg_types, new_imports = joined
    imports.update(new_imports)
    return _AnnotationData("{}[{}]".format(origin_type, ", ".join(arg_types)), imports)

def _join_argument_annotations(
    annotations: Iterable[Optional[_AnnotationData]],
) -> Optional[Tuple[List[str], Set[str]]]:
    imports: Set[str] = set()
    arg_types: List[str] = []

    for annotation in annotations:
        if annotation is None:
            return None
        arg_types.append(annotation.type_name)
        imports.update(annotation.imports)

    return arg_types, imports

def _parameter_to_annotation(parameter: Any) -> Optional[_AnnotationData]:
    if isinstance(parameter, str):
        return None

    if isinstance(parameter, ForwardRef):
        forwarded_value = parameter.__forward_value__
        if forwarded_value is None:
            return None
        return _parameter_to_annotation(forwarded_value)

    if isinstance(parameter, list):
        joined = _join_argument_annotations(
            _parameter_to_annotation(param) for param in parameter
        )
        if joined is None:
            return None
        arg_type_names, new_imports = joined
        return _AnnotationData("[{}]".format(", ".join(arg_type_names)), new_imports)

    if isinstance(parameter, type):
        if parameter.__module__ == "builtins":
            return _AnnotationData(
                "None" if parameter.__name__ == "NoneType" else parameter.__name__,
                set(),
            )

        type_name = _get_qualname(parameter, include_module=True)

        if type_name == "types.UnionType":
            return _AnnotationData("typing.Union", {"typing"})
    else:
        if hasattr(parameter, "__module__") and hasattr(parameter, "__name__"):
            type_name = _get_qualname(parameter, include_module=True)
        else:
            type_name = str(parameter)

    if type_name.startswith("hypothesis.strategies."):
        return _AnnotationData(type_name.replace("hypothesis.strategies", "st"), set())

    origin_type = get_origin(parameter)

    if origin_type is None or origin_type == parameter:
        return _AnnotationData(type_name, set(type_name.rsplit(".", maxsplit=1)[:-1]))

    arg_types = get_args(parameter)
    if {type(a) for a in arg_types} == {TypeVar}:
        arg_types = ()

    if type_name.startswith("typing."):
        try:
            new_type_name = type_name[: type_name.index("[")]
        except ValueError:
            new_type_name = type_name
        origin_annotation = _AnnotationData(new_type_name, {"typing"})
    else:
        origin_annotation = _parameter_to_annotation(origin_type)

    if arg_types:
        return _join_generics(
            origin_annotation,
            (_parameter_to_annotation(arg_type) for arg_type in arg_types),
        )
    return origin_annotation

def _are_annotations_used(*functions: Callable) -> bool:
    for function in functions:
        try:
            params = get_signature(function).parameters.values()
        except Exception:
            pass
        else:
            if any(param.annotation != inspect.Parameter.empty for param in params):
                return True
    return False

def _make_test(imports: ImportSet, body: str) -> str:
    body = body.replace("builtins.", "").replace("__main__.", "")
    imports |= {("hypothesis", "given"), ("hypothesis", "strategies as st")}
    if "        reject()\n" in body:
        imports.add(("hypothesis", "reject"))

    do_not_import = {"builtins", "__main__", "hypothesis.strategies"}
    direct = {f"import {i}" for i in imports - do_not_import if isinstance(i, str)}
    from_imports = defaultdict(set)
    for module, name in {i for i in imports if isinstance(i, tuple)}:
        if not (module.startswith("hypothesis.strategies") and name in st.__all__):
            from_imports[module].add(name)
    from_ = {
        "from {} import {}".format(module, ", ".join(sorted(names)))
        for module, names in from_imports.items()
        if isinstance(module, str) and module not in do_not_import
    }
    header = IMPORT_SECTION.format(imports="\n".join(sorted(direct) + sorted(from_)))
    nothings = body.count("st.nothing()")
    if nothings == 1:
        header += "# TODO: replace st.nothing() with an appropriate strategy\n\n"
    elif nothings >= 1:
        header += "# TODO: replace st.nothing() with appropriate strategies\n\n"
    return black.format_str(header + body, mode=black.FileMode())

def _is_probably_ufunc(obj):
    has_attributes = "nin nout nargs ntypes types identity signature".split()
    return callable(obj) and all(hasattr(obj, name) for name in has_attributes)

ROUNDTRIP_PAIRS = (
    (r"write(.+)", "read{}"),
    (r"save(.+)", "load{}"),
    (r"dump(.+)", "load{}"),
    (r"to(.+)", "from{}"),
    (r"(.*)en(.+)", "{}de{}"),
    (r"(.+)", "de{}"),
    (r"(?!safe)(.+)