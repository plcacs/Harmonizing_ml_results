from inspect import Parameter
from typing import Iterator, List, Optional, Tuple, Any, Union
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils

class _SignatureMixin:

    def to_string(self) -> str:
        def param_strings() -> Iterator[str]:
            is_positional = False
            is_kw_only = False
            for n in self.get_param_names(resolve_stars=True):
                kind = n.get_kind()
                is_positional |= kind == Parameter.POSITIONAL_ONLY
                if is_positional and kind != Parameter.POSITIONAL_ONLY:
                    yield '/'
                    is_positional = False
                if kind == Parameter.VAR_POSITIONAL:
                    is_kw_only = True
                elif kind == Parameter.KEYWORD_ONLY and (not is_kw_only):
                    yield '*'
                    is_kw_only = True
                yield n.to_string()
            if is_positional:
                yield '/'
        s = self.name.string_name + '(' + ', '.join(param_strings()) + ')'
        annotation = self.annotation_string
        if annotation:
            s += ' -> ' + annotation
        return s

class AbstractSignature(_SignatureMixin):

    def __init__(self, value: Any, is_bound: bool = False) -> None:
        self.value = value
        self.is_bound = is_bound

    @property
    def name(self) -> Any:
        return self.value.name

    @property
    def annotation_string(self) -> str:
        return ''

    def get_param_names(self, resolve_stars: bool = False) -> List[Any]:
        param_names = self._function_value.get_param_names()
        if self.is_bound:
            return param_names[1:]
        return param_names

    def bind(self, value: Any) -> Any:
        raise NotImplementedError

    def matches_signature(self, arguments: Any) -> bool:
        return True

    def __repr__(self) -> str:
        if self.value is self._function_value:
            return '<%s: %s>' % (self.__class__.__name__, self.value)
        return '<%s: %s, %s>' % (self.__class__.__name__, self.value, self._function_value)

class TreeSignature(AbstractSignature):

    def __init__(self, value: Any, function_value: Optional[Any] = None, is_bound: bool = False) -> None:
        super().__init__(value, is_bound)
        self._function_value = function_value or value

    def bind(self, value: Any) -> 'TreeSignature':
        return TreeSignature(value, self._function_value, is_bound=True)

    @property
    def _annotation(self) -> Optional[Any]:
        if self.value.is_class():
            return None
        return self._function_value.tree_node.annotation

    @property
    def annotation_string(self) -> str:
        a = self._annotation
        if a is None:
            return ''
        return a.get_code(include_prefix=False)

    @memoize_method
    def get_param_names(self, resolve_stars: bool = False) -> List[Any]:
        params = self._function_value.get_param_names()
        if resolve_stars:
            from jedi.inference.star_args import process_params
            params = process_params(params)
        if self.is_bound:
            return params[1:]
        return params

    def matches_signature(self, arguments: Any) -> bool:
        from jedi.inference.param import get_executed_param_names_and_issues
        executed_param_names, issues = get_executed_param_names_and_issues(self._function_value, arguments)
        if issues:
            return False
        matches = all((executed_param_name.matches_signature() for executed_param_name in executed_param_names))
        if debug.enable_notice:
            tree_node = self._function_value.tree_node
            signature = parser_utils.get_signature(tree_node)
            if matches:
                debug.dbg('Overloading match: %s@%s (%s)', signature, tree_node.start_pos[0], arguments, color='BLUE')
            else:
                debug.dbg('Overloading no match: %s@%s (%s)', signature, tree_node.start_pos[0], arguments, color='BLUE')
        return matches

class BuiltinSignature(AbstractSignature):

    def __init__(self, value: Any, return_string: str, function_value: Optional[Any] = None, is_bound: bool = False) -> None:
        super().__init__(value, is_bound)
        self._return_string = return_string
        self.__function_value = function_value

    @property
    def annotation_string(self) -> str:
        return self._return_string

    @property
    def _function_value(self) -> Any:
        if self.__function_value is None:
            return self.value
        return self.__function_value

    def bind(self, value: Any) -> 'BuiltinSignature':
        return BuiltinSignature(value, self._return_string, function_value=self.value, is_bound=True)

class SignatureWrapper(_SignatureMixin):

    def __init__(self, wrapped_signature: Any) -> None:
        self._wrapped_signature = wrapped_signature

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_signature, name)
