"""
There are a couple of classes documented in here:

- :class:`.BaseName` as an abstact base class for almost everything.
- :class:`.Name` used in a lot of places
- :class:`.Completion` for completions
- :class:`.BaseSignature` as a base class for signatures
- :class:`.Signature` for :meth:`.Script.get_signatures` only
- :class:`.ParamName` used for parameters of signatures
- :class:`.Refactoring` for refactorings
- :class:`.SyntaxError` for :meth:`.Script.get_syntax_errors` only

These classes are the much biggest part of the API, because they contain
the interesting information about all operations.
"""
import re
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from parso.python.tree import Name as ParsoName
from parso.tree import search_ancestor
from jedi import settings
from jedi import debug
from jedi.inference.utils import unite
from jedi.cache import memoize_method
from jedi.inference.compiled.mixed import MixedName
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.conversion import (
    convert_names,
    convert_values,
)
from jedi.inference.base_value import ValueSet, HasNoContext
from jedi.api.keywords import KeywordName
from jedi.api import completion_cache
from jedi.api.helpers import filter_follow_imports


def _sort_names_by_start_pos(names: List[Any]) -> List[Any]:
    return sorted(names, key=lambda s: s.start_pos or (0, 0))


def defined_names(inference_state: Any, value: Any) -> List["Name"]:
    """
    List sub-definitions (e.g., methods in class).

    :type scope: Scope
    :rtype: list of Name
    """
    try:
        context = value.as_context()
    except HasNoContext:
        return []
    filter_ = next(context.get_filters())
    names = [name for name in filter_.values()]
    return [Name(inference_state, n) for n in _sort_names_by_start_pos(names)]


def _values_to_definitions(values: List[Any]) -> List["Name"]:
    return [Name(c.inference_state, c.name) for c in values]


class BaseName:
    """
    The base class for all definitions, completions and signatures.
    """

    _mapping: Dict[str, str] = {
        "posixpath": "os.path",
        "riscospath": "os.path",
        "ntpath": "os.path",
        "os2emxpath": "os.path",
        "macpath": "os.path",
        "genericpath": "os.path",
        "posix": "os",
        "_io": "io",
        "_functools": "functools",
        "_collections": "collections",
        "_socket": "socket",
        "_sqlite3": "sqlite3",
    }
    _tuple_mapping: Dict[Tuple[str, ...], str] = dict(
        (
            tuple(k.split(".")),
            v,
        )
        for k, v in {"argparse._ActionsContainer": "argparse.ArgumentParser"}.items()
    )

    def __init__(self, inference_state: Any, name: Any) -> None:
        self._inference_state = inference_state
        self._name = name
        "\n        An instance of :class:`parso.python.tree.Name` subclass.\n        "
        self.is_keyword: bool = isinstance(self._name, KeywordName)

    @memoize_method
    def _get_module_context(self) -> Any:
        return self._name.get_root_context()

    @property
    def module_path(self) -> Optional[str]:
        """
        Shows the file path of a module. e.g. ``/usr/lib/python3.9/os.py``
        """
        module = self._get_module_context()
        if module.is_stub() or not module.is_compiled():
            path = self._get_module_context().py__file__()
            if path is not None:
                return path
        return None

    @property
    def name(self) -> Optional[str]:
        """
        Name of variable/function/class/module.

        For example, for ``x = None`` it returns ``'x'``.

        :rtype: str or None
        """
        return self._name.get_public_name()

    @property
    def type(self) -> Optional[str]:
        """
        The type of the definition.

        [Docstring omitted for brevity]
        """
        tree_name = getattr(self._name, "tree_name", None)
        resolve = False
        if tree_name is not None:
            definition = tree_name.get_definition()
            if definition is not None and definition.type == "import_from" and tree_name.is_definition():
                resolve = True
        if isinstance(self._name, SubModuleName) or resolve:
            for value in self._name.infer():
                return value.api_type
        return self._name.api_type

    @property
    def module_name(self) -> Optional[str]:
        """
        The module name, a bit similar to what ``__name__`` is in a random
        Python module.

        [Docstring omitted for brevity]
        """
        return self._get_module_context().py__name__()

    def in_builtin_module(self) -> bool:
        """
        Returns True, if this is a builtin module.
        """
        value = self._get_module_context().get_value()
        if isinstance(value, StubModuleValue):
            return any(v.is_compiled() for v in value.non_stub_value_set)
        return value.is_compiled()

    @property
    def line(self) -> Optional[int]:
        """The line where the definition occurs (starting with 1)."""
        start_pos = getattr(self._name, "start_pos", None)
        if start_pos is None:
            return None
        return start_pos[0]

    @property
    def column(self) -> Optional[int]:
        """The column where the definition occurs (starting with 0)."""
        start_pos = getattr(self._name, "start_pos", None)
        if start_pos is None:
            return None
        return start_pos[1]

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        """
        The (row, column) of the start of the definition range. Rows start with
        1, columns start with 0.

        :rtype: Optional[Tuple[int, int]]
        """
        if getattr(self._name, "tree_name", None) is None:
            return None
        definition = self._name.tree_name.get_definition()
        if definition is None:
            return self._name.start_pos
        return definition.start_pos

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        """
        The (row, column) of the end of the definition range. Rows start with
        1, columns start with 0.

        :rtype: Optional[Tuple[int, int]]
        """
        if getattr(self._name, "tree_name", None) is None:
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

    def docstring(
        self, raw: bool = False, fast: bool = True
    ) -> str:
        """
        Return a document string for this completion object.

        [Docstring omitted for brevity]
        """
        if isinstance(self._name, ImportName) and fast:
            return ""
        doc = self._get_docstring()
        if raw:
            return doc
        signature_text = self._get_docstring_signature()
        if signature_text and doc:
            return signature_text + "\n\n" + doc
        else:
            return signature_text + doc

    def _get_docstring(self) -> str:
        return self._name.py__doc__()

    def _get_docstring_signature(self) -> str:
        return "\n".join(
            signature.to_string()
            for signature in self._get_signatures(for_docstring=True)
        )

    @property
    def description(self) -> str:
        """
        A description of the :class:`.Name` object, which is heavily used
        in testing. e.g. for ``isinstance`` it returns ``def isinstance``.

        [Docstring omitted for brevity]
        """
        typ = self.type
        tree_name = getattr(self._name, "tree_name", None)
        if typ == "param":
            return typ + " " + self._name.to_string()
        if typ in ("function", "class", "module", "instance") or tree_name is None:
            if typ == "function":
                typ = "def"
            return typ + " " + self._name.get_public_name()
        definition = self._name.tree_name.get_definition(include_setitem=True) or self._name.tree_name
        txt = definition.get_code(include_prefix=False)
        txt = re.sub(r"#[^\n]+\n", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    @property
    def full_name(self) -> Optional[str]:
        """
        Dot-separated path of this object.

        [Docstring omitted for brevity]
        """
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
        return ".".join(names)

    def is_stub(self) -> bool:
        """
        Returns True if the current name is defined in a stub file.
        """
        if not self._name.is_value_name:
            return False
        return self._name.get_root_context().is_stub()

    def is_side_effect(self) -> bool:
        """
        Checks if a name is defined as ``self.foo = 3``. In case of self, this
        function would return False, for foo it would return True.
        """
        tree_name = getattr(self._name, "tree_name", None)
        if tree_name is None:
            return False
        return tree_name.is_definition() and tree_name.parent.type == "trailer"

    @debug.increase_indent_cm("goto on name")
    def goto(
        self,
        *,
        follow_imports: bool = False,
        follow_builtin_imports: bool = False,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List["Name"]:
        """
        Like :meth:`.Script.goto` (also supports the same params), but does it
        for the current name. This is typically useful if you are using
        something like :meth:`.Script.get_names()`.

        [Docstring omitted for brevity]
        """
        if not self._name.is_value_name:
            return []
        names = self._name.goto()
        if follow_imports:
            names = filter_follow_imports(names, follow_builtin_imports)
        names = convert_names(names, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        return [
            self
            if n == self._name
            else Name(self._inference_state, n)
            for n in names
        ]

    @debug.increase_indent_cm("infer on name")
    def infer(
        self,
        *,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List["Name"]:
        """
        Like :meth:`.Script.infer`, it can be useful to understand which type
        the current name has.

        [Docstring omitted for brevity]
        """
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
        return [
            self
            if n == self._name
            else Name(self._inference_state, n)
            for n in resulting_names
        ]

    def parent(self) -> Optional["Name"]:
        """
        Returns the parent scope of this identifier.

        :rtype: Name
        """
        if not self._name.is_value_name:
            return None
        if self.type in ("function", "class", "param") and hasattr(self._name, "tree_name") and self._name.tree_name is not None:
            cls_or_func_node = self._name.tree_name.get_definition()
            parent = search_ancestor(cls_or_func_node, "funcdef", "classdef", "file_input")
            context = self._get_module_context().create_value(parent).as_context()
        else:
            context = self._name.parent_context
        if context is None:
            return None
        while context.name is None:
            context = context.parent_context
            if context is None:
                return None
        return Name(self._inference_state, context.name)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {'full_' if self.full_name else ''}"
            f"name={repr(self.full_name or self.name)}, description={repr(self.description)}>"
        )

    def get_line_code(
        self, before: int = 0, after: int = 0
    ) -> str:
        """
        Returns the line of code where this object was defined.

        [Docstring omitted for brevity]
        """
        if not self._name.is_value_name:
            return ""
        lines = self._name.get_root_context().code_lines
        if lines is None:
            return ""
        index = self._name.start_pos[0] - 1
        start_index = max(index - before, 0)
        return "".join(lines[start_index : index + after + 1])

    def _get_signatures(
        self, for_docstring: bool = False
    ) -> List[Any]:
        if self._name.api_type == "property":
            return []
        if for_docstring and self._name.api_type == "statement" and not self.is_stub():
            return []
        if isinstance(self._name, MixedName):
            return self._name.infer_compiled_value().get_signatures()
        names = convert_names([self._name], prefer_stubs=True)
        return [
            sig
            for name in names
            for sig in name.infer().get_signatures()
        ]

    def get_signatures(self) -> List["BaseSignature"]:
        """
        Returns all potential signatures for a function or a class. Multiple
        signatures are typical if you use Python stubs with ``@overload``.

        :rtype: list of :class:`BaseSignature`
        """
        return [BaseSignature(self._inference_state, s) for s in self._get_signatures()]

    def execute(self) -> List["Name"]:
        """
        Uses type inference to "execute" this identifier and returns the
        executed objects.

        :rtype: list of :class:`Name`
        """
        return _values_to_definitions(self._name.infer().execute_with_values())

    def get_type_hint(self) -> str:
        """
        Returns type hints like ``Iterable[int]`` or ``Union[int, str]``.

        [Docstring omitted for brevity]
        """
        return self._name.infer().get_type_hint()


class Completion(BaseName):
    """
    ``Completion`` objects are returned from :meth:`.Script.complete`. They
    provide additional information about a completion.
    """

    def __init__(
        self,
        inference_state: Any,
        name: Any,
        stack: Any,
        like_name_length: int,
        is_fuzzy: bool,
        cached_name: Optional[Any] = None,
    ) -> None:
        super().__init__(inference_state, name)
        self._like_name_length: int = like_name_length
        self._stack: Any = stack
        self._is_fuzzy: bool = is_fuzzy
        self._cached_name: Optional[Any] = cached_name
        self._same_name_completions: List[Any] = []

    def _complete(self, like_name: bool) -> str:
        append = ""
        if settings.add_bracket_after_function and self.type == "function":
            append = "("
        name = self._name.get_public_name()
        if like_name:
            name = name[self._like_name_length :]
        return name + append

    @property
    def complete(self) -> Optional[str]:
        """
        Only works with non-fuzzy completions. Returns None if fuzzy
        completions are used.

        [Docstring omitted for brevity]
        """
        if self._is_fuzzy:
            return None
        return self._complete(True)

    @property
    def name_with_symbols(self) -> str:
        """
        Similar to :attr:`.name`, but like :attr:`.name` returns also the
        symbols, for example assuming the following function definition::

            def foo(param=0):
                pass

        completing ``foo(`` would give a ``Completion`` which
        ``name_with_symbols`` would be "param=".

        """
        return self._complete(False)

    def docstring(
        self, raw: bool = False, fast: bool = True
    ) -> str:
        """
        Documented under :meth:`BaseName.docstring`.
        """
        if self._like_name_length >= 3:
            fast = False
        return super().docstring(raw=raw, fast=fast)

    def _get_docstring(self) -> str:
        if self._cached_name is not None:
            return completion_cache.get_docstring(
                self._cached_name,
                self._name.get_public_name(),
                lambda: self._get_cache(),
            )
        return super()._get_docstring()

    def _get_docstring_signature(self) -> str:
        if self._cached_name is not None:
            return completion_cache.get_docstring_signature(
                self._cached_name,
                self._name.get_public_name(),
                lambda: self._get_cache(),
            )
        return super()._get_docstring_signature()

    def _get_cache(self) -> Tuple[Optional[str], str, str]:
        return (
            super().type,
            super()._get_docstring_signature(),
            super()._get_docstring(),
        )

    @property
    def type(self) -> Optional[str]:
        """
        Documented under :meth:`BaseName.type`.
        """
        if self._cached_name is not None:
            return completion_cache.get_type(
                self._cached_name,
                self._name.get_public_name(),
                lambda: self._get_cache(),
            )
        return super().type

    def get_completion_prefix_length(self) -> int:
        """
        Returns the length of the prefix being completed.
        For example, completing ``isinstance``::

            isinstan# <-- Cursor is here

        would return 8, because len('isinstan') == 8.

        Assuming the following function definition::

            def foo(param=0):
                pass

        completing ``foo(par`` would return 3.
        """
        return self._like_name_length

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self._name.get_public_name()}>"

        
class Name(BaseName):
    """
    *Name* objects are returned from many different APIs including
    :meth:`.Script.goto` or :meth:`.Script.infer`.
    """

    def __init__(self, inference_state: Any, definition: Any) -> None:
        super().__init__(inference_state, definition)

    @memoize_method
    def defined_names(self) -> List["Name"]:
        """
        List sub-definitions (e.g., methods in class).

        :rtype: list of :class:`Name`
        """
        defs = self._name.infer()
        return sorted(
            unite(
                (defined_names(self._inference_state, d) for d in defs)
            ),
            key=lambda s: s._name.start_pos or (0, 0),
        )

    def is_definition(self) -> bool:
        """
        Returns True, if defined as a name in a statement, function or class.
        Returns False, if it's a reference to such a definition.
        """
        tree_name = getattr(self._name, "tree_name", None)
        if tree_name is None:
            return True
        else:
            return self._name.tree_name.is_definition()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Name):
            return NotImplemented
        return (
            self._name.start_pos == other._name.start_pos
            and self.module_path == other.module_path
            and self.name == other.name
            and self._inference_state == other._inference_state
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(
            (
                self._name.start_pos,
                self.module_path,
                self.name,
                self._inference_state,
            )
        )


class BaseSignature(Name):
    """
    These signatures are returned by :meth:`BaseName.get_signatures`
    calls.
    """

    def __init__(self, inference_state: Any, signature: Any) -> None:
        super().__init__(inference_state, signature.name)
        self._signature = signature

    @property
    def params(self) -> List["ParamName"]:
        """
        Returns definitions for all parameters that a signature defines.
        This includes stuff like ``*args`` and ``**kwargs``.

        :rtype: list of :class:`.ParamName`
        """
        return [
            ParamName(self._inference_state, n)
            for n in self._signature.get_param_names(resolve_stars=True)
        ]

    def to_string(self) -> str:
        """
        Returns a text representation of the signature. This could for example
        look like ``foo(bar, baz: int, **kwargs)``.

        :rtype: str
        """
        return self._signature.to_string()


class Signature(BaseSignature):
    """
    A full signature object is the return value of
    :meth:`.Script.get_signatures`.
    """

    def __init__(
        self, 
        inference_state: Any, 
        signature: Any, 
        call_details: Any
    ) -> None:
        super().__init__(inference_state, signature)
        self._call_details = call_details
        self._signature = signature

    @property
    def index(self) -> Optional[int]:
        """
        Returns the param index of the current cursor position.
        Returns None if the index cannot be found in the curent call.

        :rtype: int
        """
        return self._call_details.calculate_index(
            self._signature.get_param_names(resolve_stars=True)
        )

    @property
    def bracket_start(self) -> Tuple[int, int]:
        """
        Returns a line/column tuple of the bracket that is responsible for the
        last function call. The first line is 1 and the first column 0.

        :rtype: int, int
        """
        return self._call_details.bracket_leaf.start_pos

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: index={repr(self.index)} {self._signature.to_string()}>"
    

class ParamName(Name):
    def infer_default(self) -> List["Name"]:
        """
        Returns default values like the ``1`` of ``def foo(x=1):``.

        :rtype: list of :class:`.Name`
        """
        return _values_to_definitions(self._name.infer_default())

    def infer_annotation(self, **kwargs: Any) -> List["Name"]:
        """
        :param execute_annotation: Default True; If False, values are not
            executed and classes are returned instead of instances.
        :rtype: list of :class:`.Name`
        """
        return _values_to_definitions(
            self._name.infer_annotation(ignore_stars=True, **kwargs)
        )

    def to_string(self) -> str:
        """
        Returns a simple representation of a param, like
        ``f: Callable[..., Any]``.

        :rtype: str
        """
        return self._name.to_string()

    @property
    def kind(self) -> Any:
        """
        Returns an enum instance of :mod:`inspect`'s ``Parameter`` enum.

        :rtype: :py:attr:`inspect.Parameter.kind`
        """
        return self._name.get_kind()
