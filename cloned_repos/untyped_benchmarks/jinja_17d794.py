import re
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union
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
SUPPORTED_LANG_ARG = jinja2.nodes.Name('supported_languages', 'param')

class MacroStack(threading.local):

    def __init__(self):
        super().__init__()
        self.call_stack = []

    @property
    def depth(self):
        return len(self.call_stack)

    def push(self, name):
        self.call_stack.append(name)

    def pop(self, name):
        got = self.call_stack.pop()
        if got != name:
            raise DbtInternalError(f'popped {got}, expected {name}')

class MacroGenerator(CallableMacroGenerator):

    def __init__(self, macro, context=None, node=None, stack=None):
        super().__init__(macro, context)
        self.node = node
        self.stack = stack

    @contextmanager
    def track_call(self):
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

    def __call__(self, *args, **kwargs):
        with self.track_call():
            return self.call_macro(*args, **kwargs)

class UnitTestMacroGenerator(MacroGenerator):

    def __init__(self, macro_generator, call_return_value):
        super().__init__(macro_generator.macro, macro_generator.context, macro_generator.node, macro_generator.stack)
        self.call_return_value = call_return_value

    def __call__(self, *args, **kwargs):
        with self.track_call():
            return self.call_return_value
_HAS_RENDER_CHARS_PAT = re.compile('({[{%#]|[#}%]})')
_render_cache = dict()

def get_rendered(string, ctx, node=None, capture_macros=False, native=False):
    has_render_chars = not isinstance(string, str) or _HAS_RENDER_CHARS_PAT.search(string)
    if not has_render_chars:
        if not native:
            return string
        elif string in _render_cache:
            return _render_cache[string]
    template = get_template(string, ctx, node, capture_macros=capture_macros, native=native)
    rendered = render_template(template, ctx, node)
    if not has_render_chars and native:
        _render_cache[string] = rendered
    return rendered

def undefined_error(msg):
    raise jinja2.exceptions.UndefinedError(msg)
GENERIC_TEST_KWARGS_NAME = '_dbt_generic_test_kwargs'

def add_rendered_test_kwargs(context, node, capture_macros=False):
    """Render each of the test kwargs in the given context using the native
    renderer, then insert that value into the given context as the special test
    keyword arguments member.
    """
    looks_like_func = '^\\s*(env_var|ref|var|source|doc)\\s*\\(.+\\)\\s*$'

    def _convert_function(value, keypath):
        if isinstance(value, str):
            if keypath == ('column_name',):
                return value
            if re.match(looks_like_func, value) is not None:
                value = f'{{{{ {value} }}}}'
            value = get_rendered(value, context, node, capture_macros=capture_macros, native=True)
        return value
    kwargs = deep_map_render(_convert_function, node.test_metadata.kwargs)
    context[GENERIC_TEST_KWARGS_NAME] = kwargs

def get_supported_languages(node):
    if 'materialization' not in node.name:
        raise MaterializtionMacroNotUsedError(node=node)
    no_kwargs = not node.defaults
    no_langs_found = SUPPORTED_LANG_ARG not in node.args
    if no_kwargs or no_langs_found:
        raise NoSupportedLanguagesFoundError(node=node)
    lang_idx = node.args.index(SUPPORTED_LANG_ARG)
    return [ModelLanguage[item.value] for item in node.defaults[-(len(node.args) - lang_idx)].items]