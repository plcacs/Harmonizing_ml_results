import asyncio
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from third_party import X, Y, Z
from library import some_connection, some_decorator
f'trigger 3.6 mode'

def func_no_args() -> None:
    a
    b
    c
    if True:
        raise RuntimeError
    if False:
        ...
    for i in range(10):
        print(i)
        continue
    exec('new-style exec', {}, {})
    return None

async def coroutine(arg: Any, exec: bool = False) -> None:
    """Single-line docstring. Multiline is harder to reformat."""
    async with some_connection() as conn:
        await conn.do_what_i_mean('SELECT bobby, tables FROM xkcd', timeout=2)
    await asyncio.sleep(1)

@asyncio.coroutine
@some_decorator(with_args=True, many_args=[1, 2, 3])
def function_signature_stress_test(
    number: int, 
    no_annotation: Optional[Any] = None, 
    text: str = 'default', 
    *, 
    debug: bool = False, 
    **kwargs: Any
) -> str:
    return text[number:-1]

def spaces(
    a: int = 1, 
    b: Tuple = (), 
    c: List = [], 
    d: Dict = {}, 
    e: bool = True, 
    f: int = -1, 
    g: int = 1 if False else 2, 
    h: str = '', 
    i: str = ''
) -> None:
    offset = attr.ib(default=attr.Factory(lambda: _r.uniform(10000, 200000)))
    assert task._cancel_stack[:len(old_stack)] == old_stack

def spaces_types(
    a: int = 1, 
    b: Tuple = (), 
    c: List = [], 
    d: Dict = {}, 
    e: bool = True, 
    f: int = -1, 
    g: int = 1 if False else 2, 
    h: str = '', 
    i: str = ''
) -> None:
    ...

def spaces2(result: Any = _core.Value(None)) -> None:
    assert fut is self._read_fut, (fut, self._read_fut)

def example(session: Any) -> List[Any]:
    result = session.query(models.Customer.id).filter(models.Customer.account_id == account_id, models.Customer.email == email_address).order_by(models.Customer.id.asc()).all()
    return result

def long_lines() -> None:
    if True:
        typedargslist.extend(gen_annotated_params(ast_args.kwonlyargs, ast_args.kw_defaults, parameters, implicit_default=True))
        typedargslist.extend(gen_annotated_params(ast_args.kwonlyargs, ast_args.kw_defaults, parameters, implicit_default=True))
    _type_comment_re = re.compile('\n        ^\n        [\\t ]*\n        \\#[ ]type:[ ]*\n        (?P<type>\n            [^#\\t\\n]+?\n        )\n        (?<!ignore)     # note: this will force the non-greedy + in <type> to match\n                        # a trailing space which is why we need the silliness below\n        (?<!ignore[ ]{1})(?<!ignore[ ]{2})(?<!ignore[ ]{3})(?<!ignore[ ]{4})\n        (?<!ignore[ ]{5})(?<!ignore[ ]{6})(?<!ignore[ ]{7})(?<!ignore[ ]{8})\n        (?<!ignore[ ]{9})(?<!ignore[ ]{10})\n        [\\t ]*\n        (?P<nl>\n            (?:\\#[^\\n]*)?\n            \\n?\n        )\n        $\n        ', re.MULTILINE | re.VERBOSE)

def trailing_comma() -> Dict[Any, float]:
    mapping = {A: 0.25 * (10.0 / 12), B: 0.1 * (10.0 / 12), C: 0.1 * (10.0 / 12), D: 0.1 * (10.0 / 12)}
    return mapping

def f(a: Any, **kwargs: Any) -> Any:
    return (yield from A(very_long_argument_name1=very_long_value_for_the_argument, very_long_argument_name2=very_long_value_for_the_argument, **kwargs))

def __await__() -> Any:
    return (yield)
