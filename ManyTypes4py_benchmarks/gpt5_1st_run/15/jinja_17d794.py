import re
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, NoReturn, Optional, Tuple
import jinja2
import jinja2.ext
import jinja2.nativetypes
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox
from dbt.contracts.graph.nodes import GenericTestNode
from dbt.exceptions import DbtInternalError, MaterializtionMacroNotUsedError, NoSupportedLanguagesFoundError
from dbt.node_types import ModelLanguage
from dbt_common.clients.jinja import CallableMacroGenerator, MacroProtocol, get_template, render_template
from dbt_common.utils import deep_map_render
from re import Pattern

SUPPORTED_LANG_ARG: jinja2.nodes.Name = jinja2.nodes.Name('supported_languages', 'param')


class MacroStack(threading.local):

    def __init__(self) -> None:
        super().__init__()
        self.call_stack: List[str] = []

    @property
    def depth(self) -> int:
        return len(self.call_stack)

    def push(self, name: str) -> None:
        self.call_stack.append(name)

    def pop(self, name: str) -> None:
        got = self.call_stack.pop()
        if got != name:
            raise DbtInternalError(f'popped {got}, expected {name}')


class MacroGenerator(CallableMacroGenerator):

    def __init__(
        self,
        macro: MacroProtocol,
        context: Optional[Dict[str, Any]] = None,
        node: Optional[Any] = None,
        stack: Optional[MacroStack] = None,
    ) -> None:
        super().__init__(macro, context)
        self.node: Optional[Any] = node
        self.stack: Optional[MacroStack] = stack

    @contextmanager
    def track_call(self) -> Iterator[None]:
        if self.stack is None:
            yield
        else:
            unique_id = self.macro.unique_id
            depth = self.stack.depth
            if depth == 0 and self.node:
                self.node.depends_on.add_macro(unique_id)
            self.stack.push(unique_id)
            try:
                yield
            finally:
                self.stack.pop(unique_id)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.track_call():
            return self.call_macro(*args, **kwargs)


class UnitTestMacroGenerator(MacroGenerator):

    def __init__(self, macro_generator: MacroGenerator, call_return_value: Any) -> None:
        super().__init__(macro_generator.macro, macro_generator.context, macro_generator.node, macro_generator.stack)
        self.call_return_value: Any = call_return_value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.track_call():
            return self.call_return_value


_HAS_RENDER_CHARS_PAT: Pattern[str] = re.compile('({[{%#]|[#}%]})')
_render_cache: Dict[str, Any] = dict()


def get_rendered(
    string: Any,
    ctx: Dict[str, Any],
    node: Optional[Any] = None,
    capture_macros: bool = False,
    native: bool = False,
) -> Any:
    has_render_chars = not isinstance(string, str) or _HAS_RENDER_CHARS_PAT.search(string)
    if not has_render_chars:
        if not native:
            return string
        elif string in _render_cache:
            return _render_cache[string]
    template = get_template(string, ctx, node, capture_macros=capture_macros, native=native)
    rendered = render_template(template, ctx, node)
    if not has_render_chars and native and isinstance(string, str):
        _render_cache[string] = rendered
    return rendered


def undefined_error(msg: str) -> NoReturn:
    raise jinja2.exceptions.UndefinedError(msg)


GENERIC_TEST_KWARGS_NAME: str = '_dbt_generic_test_kwargs'


def add_rendered_test_kwargs(context: Dict[str, Any], node: GenericTestNode, capture_macros: bool = False) -> None:
    """Render each of the test kwargs in the given context using the native
    renderer, then insert that value into the given context as the special test
    keyword arguments member.
    """
    looks_like_func: str = '^\\s*(env_var|ref|var|source|doc)\\s*\\(.+\\)\\s*$'

    def _convert_function(value: Any, keypath: Tuple[str, ...]) -> Any:
        if isinstance(value, str):
            if keypath == ('column_name',):
                return value
            if re.match(looks_like_func, value) is not None:
                value = f'{{{{ {value} }}}}'
            value = get_rendered(value, context, node, capture_macros=capture_macros, native=True)
        return value

    kwargs: Dict[str, Any] = deep_map_render(_convert_function, node.test_metadata.kwargs)
    context[GENERIC_TEST_KWARGS_NAME] = kwargs


def get_supported_languages(node: jinja2.nodes.Macro) -> List[ModelLanguage]:
    if 'materialization' not in node.name:
        raise MaterializtionMacroNotUsedError(node=node)
    no_kwargs = not node.defaults
    no_langs_found = SUPPORTED_LANG_ARG not in node.args
    if no_kwargs or no_langs_found:
        raise NoSupportedLanguagesFoundError(node=node)
    lang_idx = node.args.index(SUPPORTED_LANG_ARG)
    return [ModelLanguage[item.value] for item in node.defaults[-(len(node.args) - lang_idx)].items]