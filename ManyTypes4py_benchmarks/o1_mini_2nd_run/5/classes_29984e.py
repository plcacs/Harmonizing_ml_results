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
from typing import Optional, List, Tuple, Union, Callable, Any
from parso.tree import search_ancestor
from jedi import settings
from jedi import debug
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
from jedi.inference.context import Context  # Assuming this exists
from jedi.inference.value import Value  # Assuming this exists

def _sort_names_by_start_pos(names: List[Any]) -> List[Any]:
    return sorted(names, key=lambda s: s.start_pos or (0, 0))

def defined_names(inference_state: Any, value: Any) -> List['Name']:
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

def _values_to_definitions(values: List[Value]) -> List['Name']:
    return [Name(c.inference_state, c.name) for c in values]

class BaseName:
    """
    The base class for all definitions, completions and signatures.
    """
    _mapping: dict = {'posixpath': 'os.path', 'riscospath': 'os.path', 'ntpath': 'os.path', 'os2emxpath': 'os.path', 'macpath': 'os.path', 'genericpath': 'os.path', 'posix': 'os', '_io': 'io', '_functools': 'functools', '_collections': 'collections', '_socket': 'socket', '_sqlite3': 'sqlite3'}
    _tuple_mapping: dict = dict(((tuple(k.split('.')), v) for k, v in {'argparse._ActionsContainer': 'argparse.ArgumentParser'}.items()))

    def __init__(self, inference_state: Any, name: Any) -> None:
        self._inference_state = inference_state
        self._name = name
        '\n        An instance of :class:`parso.python.tree.Name` subclass.\n        '
        self.is_keyword: bool = isinstance(self._name, KeywordName)

    @memoize_method
    def _get_module_context(self) -> Context:
        return self._name.get_root_context()

    @property
    def module_path(self) -> Optional[str]:
        """
        Shows the file path of a module. e.g. ``/usr/lib/python3.9/os.py``
        """
        module = self._get_module_context()
        if module.is_stub() or not module.is_compiled():
            path = module.py__file__()  # type: Optional[str]
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
    def type(self) -> str:
        """
        The type of the definition.

        Here is an example of the value of this attribute.  Let's consider
        the following source.  As what is in ``variable`` is unambiguous
        to Jedi, :meth:`jedi.Script.infer` should return a list of
        definition for ``sys``, ``f``, ``C`` and ``x``.

        >>> from jedi import Script
        >>> source = '''
        ... import keyword
        ...
        ... class C:
        ...     pass
        ...
        ... class D:
        ...     pass
        ...
        ... x = D()
        ...
        ... def f():
        ...     pass
        ...
        ... for variable in [keyword, f, C, x]:
        ...     variable'''

        >>> script = Script(source)
        >>> defs = script.infer()

        Before showing what is in ``defs``, let's sort it by :attr:`line`
        so that it is easy to relate the result to the source code.

        >>> defs = sorted(defs, key=lambda d: d.line)
        >>> print(defs)  # doctest: +NORMALIZE_WHITESPACE
        [<Name full_name='keyword', description='module keyword'>,
         <Name full_name='__main__.C', description='class C'>,
         <Name full_name='__main__.D', description='instance D'>,
         <Name full_name='__main__.f', description='def f'>]

        Finally, here is what you can get from :attr:`type`:

        >>> defs = [d.type for d in defs]
        >>> defs[0]
        'module'
        >>> defs[1]
        'class'
        >>> defs[2]
        'instance'
        >>> defs[3]
        'function'

        Valid values for type are ``module``, ``class``, ``instance``, ``function``,
        ``param``, ``path``, ``keyword``, ``property`` and ``statement``.

        """
        tree_name = self._name.tree_name
        resolve = False
        if tree_name is not None:
            definition = tree_name.get_definition()
            if definition is not None and definition.type == 'import_from' and tree_name.is_definition():
                resolve = True
        if isinstance(self._name, SubModuleName) or resolve:
            for value in self._name.infer():
                return value.api_type
        return self._name.api_type

    @property
    def module_name(self) -> str:
        """
        The module name, a bit similar to what ``__name__`` is in a random
        Python module.

        >>> from jedi import Script
        >>> source = 'import json'
        >>> script = Script(source, path='example.py')
        >>> d = script.infer()[0]
        >>> print(d.module_name)  # doctest: +ELLIPSIS
        json
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
        start_pos = self._name.start_pos
        if start_pos is None:
            return None
        return start_pos[0]

    @property
    def column(self) -> Optional[int]:
        """The column where the definition occurs (starting with 0)."""
        start_pos = self._name.start_pos
        if start_pos is None:
            return None
        return start_pos[1]

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        """
        The (row, column) of the start of the definition range. Rows start with
        1, columns start with 0.

        :rtype: Optional[Tuple[int, int]]
        """
        if self._name.tree_name is None:
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
        if self._name.tree_name is None:
            return None
        definition = self._name.tree_name.get_definition()
        if definition is None:
            return self._name.tree_name.end_pos
        if self.type in ('function', 'class'):
            last_leaf = definition.get_last_leaf()
            if last_leaf.type == 'newline':
                return last_leaf.get_previous_leaf().end_pos
            return last_leaf.end_pos
        return definition.end_pos

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        """
        Return a document string for this completion object.

        Example:

        >>> from jedi import Script
        >>> source = '''\\
        ... def f(a, b=1):
        ...     "Document for function f."
        ... '''
        >>> script = Script(source, path='example.py')
        >>> doc = script.infer(1, len('def f'))[0].docstring()
        >>> print(doc)
        f(a, b=1)
        <BLANKLINE>
        Document for function f.

        Notice that useful extra information is added to the actual
        docstring, e.g. function signatures are prepended to their docstrings.
        If you need the actual docstring, use ``raw=True`` instead.

        >>> print(script.infer(1, len('def f'))[0].docstring(raw=True))
        Document for function f.

        :param fast: Don't follow imports that are only one level deep like
            ``import foo``, but follow ``from foo import bar``. This makes
            sense for speed reasons. Completing `import a` is slow if you use
            the ``foo.docstring(fast=False)`` on every object, because it
            parses all libraries starting with ``a``.
        """
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
        return '\n'.join(signature.to_string() for signature in self._get_signatures(for_docstring=True))

    @property
    def description(self) -> str:
        """
        A description of the :class:`.Name` object, which is heavily used
        in testing. e.g. for ``isinstance`` it returns ``def isinstance``.

        Example:

        >>> from jedi import Script
        >>> source = '''
        ... def f():
        ...     pass
        ...
        ... class C:
        ...     pass
        ...
        ... variable = f if random.choice([0,1]) else C'''
        >>> script = Script(source)  # line is maximum by default
        >>> defs = script.infer(column=3)
        >>> defs = sorted(defs, key=lambda d: d.line)
        >>> print(defs)  # doctest: +NORMALIZE_WHITESPACE
        [<Name full_name='__main__.f', description='def f'>,
         <Name full_name='__main__.C', description='class C'>]
        >>> str(defs[0].description)
        'def f'
        >>> str(defs[1].description)
        'class C'

        """
        typ: str = self.type
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
        """
        Dot-separated path of this object.

        It is in the form of ``<module>[.<submodule>[...]][.<object>]``.
        It is useful when you want to look up Python manual of the
        object at hand.

        Example:

        >>> from jedi import Script
        >>> source = '''
        ... import os
        ... os.path.join'''
        >>> script = Script(source, path='example.py')
        >>> print(script.infer(3, len('os.path.join'))[0].full_name)
        os.path.join

        Notice that it returns ``'os.path.join'`` instead of (for example)
        ``'posixpath.join'``. This is not correct, since the modules name would
        be ``<module 'posixpath' ...>