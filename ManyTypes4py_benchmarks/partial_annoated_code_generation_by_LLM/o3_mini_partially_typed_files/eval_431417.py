from __future__ import annotations
import tokenize
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Iterable
import warnings
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.common import is_extension_array_dtype, is_string_dtype
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import PARSERS, Expr
from pandas.core.computation.parsing import tokenize_string
from pandas.core.computation.scope import ensure_scope
from pandas.core.generic import NDFrame
from pandas.io.formats.printing import pprint_thing
if TYPE_CHECKING:
    from pandas.core.computation.ops import BinOp

def _check_engine(engine: Optional[str]) -> str:
    """
    Make sure a valid engine is passed.

    Parameters
    ----------
    engine : Optional[str]
        String to validate.

    Raises
    ------
    KeyError
      * If an invalid engine is passed.
    ImportError
      * If numexpr was requested but doesn't exist.

    Returns
    -------
    str
        Engine name.
    """
    from pandas.core.computation.check import NUMEXPR_INSTALLED
    from pandas.core.computation.expressions import USE_NUMEXPR
    if engine is None:
        engine = 'numexpr' if USE_NUMEXPR else 'python'
    if engine not in ENGINES:
        valid_engines = list(ENGINES.keys())
        raise KeyError(f"Invalid engine '{engine}' passed, valid engines are {valid_engines}")
    if engine == 'numexpr' and (not NUMEXPR_INSTALLED):
        raise ImportError("'numexpr' is not installed or an unsupported version. Cannot use engine='numexpr' for query/eval if 'numexpr' is not installed")
    return engine

def _check_parser(parser: str) -> None:
    """
    Make sure a valid parser is passed.

    Parameters
    ----------
    parser : str

    Raises
    ------
    KeyError
      * If an invalid parser is passed
    """
    if parser not in PARSERS:
        raise KeyError(f"Invalid parser '{parser}' passed, valid parsers are {PARSERS.keys()}")

def _check_resolvers(resolvers: Iterable[Any]) -> None:
    if resolvers is not None:
        for resolver in resolvers:
            if not hasattr(resolver, '__getitem__'):
                name = type(resolver).__name__
                raise TypeError(f"Resolver of type '{name}' does not implement the __getitem__ method")

def _check_expression(expr: Any) -> None:
    """
    Make sure an expression is not an empty string

    Parameters
    ----------
    expr : object
        An object that can be converted to a string

    Raises
    ------
    ValueError
      * If expr is an empty string
    """
    if not expr:
        raise ValueError('expr cannot be an empty string')

def _convert_expression(expr: Any) -> str:
    """
    Convert an object to an expression.

    This function converts an object to an expression (a unicode string) and
    checks to make sure it isn't empty after conversion. This is used to
    convert operators to their string representation for recursive calls to
    :func:`~pandas.eval`.

    Parameters
    ----------
    expr : object
        The object to be converted to a string.

    Returns
    -------
    str
        The string representation of an object.

    Raises
    ------
    ValueError
      * If the expression is empty.
    """
    s = pprint_thing(expr)
    _check_expression(s)
    return s

def _check_for_locals(expr: str, stack_level: int, parser: str) -> None:
    at_top_of_stack: bool = stack_level == 0
    not_pandas_parser: bool = parser != 'pandas'
    if not_pandas_parser:
        msg = "The '@' prefix is only supported by the pandas parser"
    elif at_top_of_stack:
        msg = "The '@' prefix is not allowed in top-level eval calls.\nplease refer to your variables by name without the '@' prefix."
    if at_top_of_stack or not_pandas_parser:
        for (toknum, tokval) in tokenize_string(expr):
            if toknum == tokenize.OP and tokval == '@':
                raise SyntaxError(msg)

def eval(expr: Union[str, BinOp],
         parser: str = 'pandas',
         engine: Optional[str] = None,
         local_dict: Optional[Dict[str, Any]] = None,
         global_dict: Optional[Dict[str, Any]] = None,
         resolvers: Iterable[Any] = (),
         level: int = 0,
         target: Optional[Any] = None,
         inplace: bool = False) -> Any:
    """
    Evaluate a Python expression as a string using various backends.

    .. warning::

        This function can run arbitrary code which can make you vulnerable to code
        injection if you pass user input to this function.

    Parameters
    ----------
    expr : Union[str, BinOp]
        The expression to evaluate. This string cannot contain any Python
        `statements <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,
        only Python `expressions <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.
    parser : {'pandas', 'python'}, default 'pandas'
        The parser to use to construct the syntax tree from the expression. The
        default of ``'pandas'`` parses code slightly different than standard
        Python. Alternatively, you can parse an expression using the
        ``'python'`` parser to retain strict Python semantics.  See the
        :ref:`enhancing performance <enhancingperf.eval>` documentation for
        more details.
    engine : {'python', 'numexpr'}, default 'numexpr'
        The engine used to evaluate the expression. Supported engines are

        - None : tries to use ``numexpr``, falls back to ``python``
        - ``'numexpr'`` : This default engine evaluates pandas objects using
          numexpr for large speed ups in complex expressions with large frames.
        - ``'python'`` : Performs operations as if you had ``eval``'d in top
          level python. This engine is generally not that useful.

        More backends may be available in the future.
    local_dict : Optional[Dict[str, Any]], optional
        A dictionary of local variables, taken from locals() by default.
    global_dict : Optional[Dict[str, Any]], optional
        A dictionary of global variables, taken from globals() by default.
    resolvers : Iterable[Any], optional
        A list of objects implementing the ``__getitem__`` special method that
        you can use to inject an additional collection of namespaces to use for
        variable lookup.
    level : int, optional
        The number of prior stack frames to traverse and add to the current
        scope. Most users will **not** need to change this parameter.
    target : Optional[Any], optional, default None
        This is the target object for assignment. It is used when there is
        variable assignment in the expression. If so, then `target` must
        support item assignment with string keys, and if a copy is being
        returned, it must also support `.copy()`.
    inplace : bool, default False
        If `target` is provided, and the expression mutates `target`, whether
        to modify `target` inplace. Otherwise, return a copy of `target` with
        the mutation.

    Returns
    -------
    Any
        The completion value of evaluating the given code or None if ``inplace=True``.

    Raises
    ------
    ValueError
        There are many instances where such an error can be raised:

        - `target=None`, but the expression is multiline.
        - The expression is multiline, but not all them have item assignment.
        - `inplace=True`, but the expression is missing item assignment.
        - Item assignment is provided, but the `target` does not support
          string item assignment.
        - Item assignment is provided and `inplace=False`, but the `target` does not support
          the `.copy()` method
    """
    inplace = validate_bool_kwarg(inplace, 'inplace')
    exprs: List[Union[str, BinOp]] = []
    if isinstance(expr, str):
        _check_expression(expr)
        exprs = [e.strip() for e in expr.splitlines() if e.strip() != '']
    else:
        exprs = [expr]
    multi_line: bool = len(exprs) > 1
    if multi_line and target is None:
        raise ValueError('multi-line expressions are only valid in the context of data, use DataFrame.eval')
    engine = _check_engine(engine)
    _check_parser(parser)
    _check_resolvers(resolvers)
    ret: Any = None
    first_expr: bool = True
    target_modified: bool = False
    for expr_item in exprs:
        expr_str: str = _convert_expression(expr_item)
        _check_for_locals(expr_str, level, parser)
        env = ensure_scope(level + 1, global_dict=global_dict, local_dict=local_dict, resolvers=resolvers, target=target)
        parsed_expr = Expr(expr_str, engine=engine, parser=parser, env=env)
        if engine == 'numexpr' and ((is_extension_array_dtype(parsed_expr.terms.return_type) and (not is_string_dtype(parsed_expr.terms.return_type))) or (getattr(parsed_expr.terms, 'operand_types', None) is not None and any((is_extension_array_dtype(elem) and (not is_string_dtype(elem)) for elem in parsed_expr.terms.operand_types)))):
            warnings.warn("Engine has switched to 'python' because numexpr does not support extension array dtypes. Please set your engine to python manually.", RuntimeWarning, stacklevel=find_stack_level())
            engine = 'python'
        eng = ENGINES[engine]
        eng_inst = eng(parsed_expr)
        ret = eng_inst.evaluate()
        if parsed_expr.assigner is None:
            if multi_line:
                raise ValueError('Multi-line expressions are only valid if all expressions contain an assignment')
            if inplace:
                raise ValueError('Cannot operate inplace if there is no assignment')
        assigner = parsed_expr.assigner
        if env.target is not None and assigner is not None:
            target_modified = True
            if not inplace and first_expr:
                try:
                    target = env.target
                    if isinstance(target, NDFrame):
                        target = target.copy(deep=False)
                    else:
                        target = target.copy()
                except AttributeError as err:
                    raise ValueError('Cannot return a copy of the target') from err
            else:
                target = env.target
            try:
                if inplace and isinstance(target, NDFrame):
                    target.loc[:, assigner] = ret
                else:
                    target[assigner] = ret
            except (TypeError, IndexError) as err:
                raise ValueError('Cannot assign expression output to target') from err
            if not resolvers:
                resolvers = ({assigner: ret},)
            else:
                found: bool = False
                for resolver in resolvers:
                    if assigner in resolver:
                        resolver[assigner] = ret
                        found = True
                        break
                if not found:
                    resolvers = (*resolvers, {assigner: ret})
            ret = None
            first_expr = False
    if not inplace:
        return target if target_modified else ret
