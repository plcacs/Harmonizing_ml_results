from typing import List, Optional, Tuple, Dict, Set, Any, Union, Callable
from pathlib import Path
from parso.tree import search_ancestor
from jedi import settings, debug
from jedi.inference.utils import unite
from jedi.cache import memoize_method
from jedi.inference.compiled.mixed import MixedName
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.base_value import ValueSet, HasNoContext
from jedi.api.keywords import KeywordName
from jedi.api import completion_cache
from jedi.api.helpers import filter_follow_imports

def _sort_names_by_start_pos(names: List[Any]) -> List[Any]:
    return sorted(names, key=lambda s: s.start_pos or (0, 0))

def defined_names(inference_state: Any, value: Any) -> List['Name']:
    try:
        context = value.as_context()
    except HasNoContext:
        return []
    filter = next(context.get_filters())
    names = [name for name in filter.values()]
    return [Name(inference_state, n) for n in _sort_names_by_start_pos(names)]

def _values_to_definitions(values: List[Any]) -> List['Name']:
    return [Name(c.inference_state, c.name) for c in values]

class BaseName:
    _mapping: Dict[str, str] = {
        'posixpath': 'os.path',
        'riscospath': 'os.path',
        'ntpath': 'os.path',
        'os2emxpath': 'os.path',
        'macpath': 'os.path',
        'genericpath': 'os.path',
        'posix': 'os',
        '_io': 'io',
        '_functools': 'functools',
        '_collections': 'collections',
        '_socket': 'socket',
        '_sqlite3': 'sqlite3',
    }

    _tuple_mapping: Dict[Tuple[str, ...], str] = dict((tuple(k.split('.')), v) for (k, v) in {
        'argparse._ActionsContainer': 'argparse.ArgumentParser',
    }.items())

    def __init__(self, inference_state: Any, name: Any) -> None:
        self._inference_state = inference_state
        self._name = name
        self.is_keyword = isinstance(self._name, KeywordName)

    @memoize_method
    def _get_module_context(self) -> Any:
        return self._name.get_root_context()

    @property
    def module_path(self) -> Optional[Path]:
        module = self._get_module_context()
        if module.is_stub() or not module.is_compiled():
            path: Optional[Path] = self._get_module_context().py__file__()
            if path is not None:
                return path
        return None

    @property
    def name(self) -> Optional[str]:
        return self._name.get_public_name()

    @property
    def type(self) -> str:
        tree_name = self._name.tree_name
        resolve = False
        if tree_name is not None:
            definition = tree_name.get_definition()
            if definition is not None and definition.type == 'import_from' and \
                    tree_name.is_definition():
                resolve = True

        if isinstance(self._name, SubModuleName) or resolve:
            for value in self._name.infer():
                return value.api_type
        return self._name.api_type

    @property
    def module_name(self) -> str:
        return self._get_module_context().py__name__()

    def in_builtin_module(self) -> bool:
        value = self._get_module_context().get_value()
        if isinstance(value, StubModuleValue):
            return any(v.is_compiled() for v in value.non_stub_value_set)
        return value.is_compiled()

    @property
    def line(self) -> Optional[int]:
        start_pos = self._name.start_pos
        if start_pos is None:
            return None
        return start_pos[0]

    @property
    def column(self) -> Optional[int]:
        start_pos = self._name.start_pos
        if start_pos is None:
            return None
        return start_pos[1]

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        if self._name.tree_name is None:
            return None
        definition = self._name.tree_name.get_definition()
        if definition is None:
            return self._name.start_pos
        return definition.start_pos

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        if self._name.tree_name is None:
            return None
        definition = self._name.tree_name.get_definition()
        if definition is None:
            return self._name.tree_name.end_pos
        if self.type in ("function", "class"):
            last_leaf = definition.get_last_leaf()
            if last_leaf.type == "newline":
                return last_leaf.get_previous_leaf().end_pos
            return last_leaf.end_pos
        return definition.end_pos

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        if isinstance(self._name, ImportName) and fast:
            return ''
        doc = self._get_docstring()
        if raw:
            return doc

        signature_text = self._get_docstring_signature()
        if signature_text and doc:
            return signature_text + '\n\n' + doc
        else:
            return signature_text + doc

    def _get_docstring(self) -> str:
        return self._name.py__doc__()

    def _get_docstring_signature(self) -> str:
        return '\n'.join(
            signature.to_string()
            for signature in self._get_signatures(for_docstring=True)
        )

    @property
    def description(self) -> str:
        typ = self.type
        tree_name = self._name.tree_name
        if typ == 'param':
            return typ + ' ' + self._name.to_string()
        if typ in ('function', 'class', 'module', 'instance') or tree_name is None:
            if typ == 'function':
                typ = 'def'
            return typ + ' ' + self._name.get_public_name()

        definition = tree_name.get_definition(include_setitem=True) or tree_name
        txt = definition.get_code(include_prefix=False)
        txt = re.sub(r'#[^\n]+\n', ' ', txt)
        txt = re.sub(r'\s+', ' ', txt).strip()
        return txt

    @property
    def full_name(self) -> Optional[str]:
        if not self._name.is_value_name:
            return None

        names = self._name.get_qualified_names(include_module_names=True)
        if names is None:
            return None

        names = list(names)
        try:
            names[0] = self._mapping[names[0]]
        except KeyError:
            pass

        return '.'.join(names)

    def is_stub(self) -> bool:
        if not self._name.is_value_name:
            return False
        return self._name.get_root_context().is_stub()

    def is_side_effect(self) -> bool:
        tree_name = self._name.tree_name
        if tree_name is None:
            return False
        return tree_name.is_definition() and tree_name.parent.type == 'trailer'

    @debug.increase_indent_cm('goto on name')
    def goto(self, *, follow_imports: bool = False, follow_builtin_imports: bool = False,
             only_stubs: bool = False, prefer_stubs: bool = False) -> List['Name']:
        if not self._name.is_value_name:
            return []

        names = self._name.goto()
        if follow_imports:
            names = filter_follow_imports(names, follow_builtin_imports)
        names = convert_names(
            names,
            only_stubs=only_stubs,
            prefer_stubs=prefer_stubs,
        )
        return [self if n == self._name else Name(self._inference_state, n)
                for n in names]

    @debug.increase_indent_cm('infer on name')
    def infer(self, *, only_stubs: bool = False, prefer_stubs: bool = False) -> List['Name']:
        assert not (only_stubs and prefer_stubs)

        if not self._name.is_value_name:
            return []

        names = convert_names([self._name], prefer_stubs=True)
        values = convert_values(
            ValueSet.from_sets(n.infer() for n in names),
            only_stubs=only_stubs,
            prefer_stubs=prefer_stubs,
        )
        resulting_names = [c.name for c in values]
        return [self if n == self._name else Name(self._inference_state, n)
                for n in resulting_names]

    def parent(self) -> Optional['Name']:
        if not self._name.is_value_name:
            return None

        if self.type in ('function', 'class', 'param') and self._name.tree_name is not None:
            cls_or_func_node = self._name.tree_name.get_definition()
            parent = search_ancestor(cls_or_func_node, 'funcdef', 'classdef', 'file_input')
            context = self._get_module_context().create_value(parent).as_context()
        else:
            context = self._name.parent_context

        if context is None:
            return None
        while context.name is None:
            context = context.parent_context

        return Name(self._inference_state, context.name)

    def __repr__(self) -> str:
        return "<%s %sname=%r, description=%r>" % (
            self.__class__.__name__,
            'full_' if self.full_name else '',
            self.full_name or self.name,
            self.description,
        )

    def get_line_code(self, before: int = 0, after: int = 0) -> str:
        if not self._name.is_value_name:
            return ''

        lines = self._name.get_root_context().code_lines
        if lines is None:
            return ''

        index = self._name.start_pos[0] - 1
        start_index = max(index - before, 0)
        return ''.join(lines[start_index:index + after + 1])

    def _get_signatures(self, for_docstring: bool = False) -> List[Any]:
        if self._name.api_type == 'property':
            return []
        if for_docstring and self._name.api_type == 'statement' and not self.is_stub():
            return []

        if isinstance(self._name, MixedName):
            return self._name.infer_compiled_value().get_signatures()

        names = convert_names([self._name], prefer_stubs=True)
        return [sig for name in names for sig in name.infer().get_signatures()]

    def get_signatures(self) -> List['BaseSignature']:
        return [
            BaseSignature(self._inference_state, s)
            for s in self._get_signatures()
        ]

    def execute(self) -> List['Name']:
        return _values_to_definitions(self._name.infer().execute_with_values())

    def get_type_hint(self) -> str:
        return self._name.infer().get_type_hint()

class Completion(BaseName):
    def __init__(self, inference_state: Any, name: Any, stack: Any, like_name_length: int,
                 is_fuzzy: bool, cached_name: Optional[Any] = None) -> None:
        super().__init__(inference_state, name)
        self._like_name_length = like_name_length
        self._stack = stack
        self._is_fuzzy = is_fuzzy
        self._cached_name = cached_name
        self._same_name_completions: List[Any] = []

    def _complete(self, like_name: bool) -> str:
        append = ''
        if settings.add_bracket_after_function \
                and self.type == 'function':
            append = '('

        name = self._name.get_public_name()
        if like_name:
            name = name[self._like_name_length:]
        return name + append

    @property
    def complete(self) -> Optional[str]:
        if self._is_fuzzy:
            return None
        return self._complete(True)

    @property
    def name_with_symbols(self) -> str:
        return self._complete(False)

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        if self._like_name_length >= 3:
            fast = False
        return super().docstring(raw=raw, fast=fast)

    def _get_docstring(self) -> str:
        if self._cached_name is not None:
            return completion_cache.get_docstring(
                self._cached_name,
                self._name.get_public_name(),
                lambda: self._get_cache()
            )
        return super()._get_docstring()

    def _get_docstring_signature(self) -> str:
        if self._cached_name is not None:
            return completion_cache.get_docstring_signature(
                self._cached_name,
                self._name.get_public_name(),
                lambda: self._get_cache()
            )
        return super()._get_docstring_signature()

    def _get_cache(self) -> Tuple[str, str, str]:
        return (
            super().type,
            super()._get_docstring_signature(),
            super()._get_docstring(),
        )

    @property
    def type(self) -> str:
        if self._cached_name is not None:
            return completion_cache.get_type(
                self._cached_name,
                self._name.get_public_name(),
                lambda: self._get_cache()
            )
        return super().type

    def get_completion_prefix_length(self) -> int:
        return self._like_name_length

    def __repr__(self) -> str:
        return '<%s: %s>' % (type(self).__name__, self._name.get_public_name())

class Name(BaseName):
    def __init__(self, inference_state: Any, definition: Any) -> None:
        super().__init__(inference_state, definition)

    @memoize_method
    def defined_names(self) -> List['Name']:
        defs = self._name.infer()
        return sorted(
            unite(defined_names(self._inference_state, d) for d in defs),
            key=lambda s: s._name.start_pos or (0, 0)
        )

    def is_definition(self) -> bool:
        if self._name.tree_name is None:
            return True
        else:
            return self._name.tree_name.is_definition()

    def __eq__(self, other: Any) -> bool:
        return self._name.start_pos == other._name.start_pos \
            and self.module_path == other.module_path \
            and self.name == other.name \
            and self._inference_state == other._inference_state

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self._name.start_pos, self.module_path, self.name, self._inference_state))

class BaseSignature(Name):
    def __init__(self, inference_state: Any, signature: Any) -> None:
        super().__init__(inference_state, signature.name)
        self._signature = signature

    @property
    def params(self) -> List['ParamName']:
        return [ParamName(self._inference_state, n)
                for n in self._signature.get_param_names(resolve_stars=True)]

    def to_string(self) -> str:
        return self._signature.to_string()

class Signature(BaseSignature):
    def __init__(self, inference_state: Any, signature: Any, call_details: Any) -> None:
        super().__init__(inference_state, signature)
        self._call_details = call_details
        self._signature = signature

    @property
    def index(self) -> Optional[int]:
        return self._call_details.calculate_index(
            self._signature.get_param_names(resolve_stars=True)
        )

    @property
    def bracket_start(self) -> Tuple[int, int]:
        return self._call_details.bracket_leaf.start_pos

    def __repr__(self) -> str:
        return '<%s: index=%r %s>' % (
            type(self).__name__,
            self.index,
            self._signature.to_string(),
        )

class ParamName(Name):
    def infer_default(self) -> List['Name']:
        return _values_to_definitions(self._name.infer_default())

    def infer_annotation(self, **kwargs: Any) -> List['Name']:
        return _values_to_definitions(self._name.infer_annotation(ignore_stars=True, **kwargs))

    def to_string(self) -> str:
        return self._name.to_string()

    @property
    def kind(self) -> Any:
        return self._name.get_kind()
