"""
The API basically only provides one class. You can create a :class:`Script` and
use its methods.

Additionally you can add a debug function with :func:`set_debug_function`.
Alternatively, if you don't need a custom function and are happy with printing
debug messages to stdout, simply call :func:`set_debug_function` without
arguments.
"""
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union, Callable, Iterator, Any, Iterable, TypeVar, Sequence, cast, overload

import parso
from parso.python import tree
from jedi.parser_utils import get_executable_nodes
from jedi import debug
from jedi import settings
from jedi import cache
from jedi.file_io import KnownContentFileIO
from jedi.api import classes
from jedi.api import interpreter
from jedi.api import helpers
from jedi.api.helpers import validate_line_column
from jedi.api.completion import Completion, search_in_module
from jedi.api.keywords import KeywordName
from jedi.api.environment import InterpreterEnvironment
from jedi.api.project import get_default_project, Project
from jedi.api.errors import parso_to_jedi_errors
from jedi.api import refactoring
from jedi.api.refactoring.extract import extract_function, extract_variable
from jedi.inference import InferenceState
from jedi.inference import imports
from jedi.inference.references import find_references
from jedi.inference.arguments import try_iter_content
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.sys_path import transform_path_to_dotted
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value import ModuleValue
from jedi.inference.base_value import ValueSet
from jedi.inference.value.iterable import unpack_tuple_to_dict
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.gradual.utils import load_proper_stub_module
from jedi.inference.utils import to_list
sys.setrecursionlimit(3000)

class Script:
    """
    A Script is the base for completions, goto or whatever you want to do with
    Jedi. The counter part of this class is :class:`Interpreter`, which works
    with actual dictionaries and can work with a REPL. This class
    should be used when a user edits code in an editor.

    You can either use the ``code`` parameter or ``path`` to read a file.
    Usually you're going to want to use both of them (in an editor).

    The Script's ``sys.path`` is very customizable:

    - If `project` is provided with a ``sys_path``, that is going to be used.
    - If `environment` is provided, its ``sys.path`` will be used
      (see :func:`Environment.get_sys_path <jedi.api.environment.Environment.get_sys_path>`);
    - Otherwise ``sys.path`` will match that of the default environment of
      Jedi, which typically matches the sys path that was used at the time
      when Jedi was imported.

    Most methods have a ``line`` and a ``column`` parameter. Lines in Jedi are
    always 1-based and columns are always zero based. To avoid repetition they
    are not always documented. You can omit both line and column. Jedi will
    then just do whatever action you are calling at the end of the file. If you
    provide only the line, just will complete at the end of that line.

    .. warning:: By default :attr:`jedi.settings.fast_parser` is enabled, which means
        that parso reuses modules (i.e. they are not immutable). With this setting
        Jedi is **not thread safe** and it is also not safe to use multiple
        :class:`.Script` instances and its definitions at the same time.

        If you are a normal plugin developer this should not be an issue. It is
        an issue for people that do more complex stuff with Jedi.

        This is purely a performance optimization and works pretty well for all
        typical usages, however consider to turn the setting off if it causes
        you problems. See also
        `this discussion <https://github.com/davidhalter/jedi/issues/1240>`_.

    :param code: The source code of the current file, separated by newlines.
    :type code: str
    :param path: The path of the file in the file system, or ``''`` if
        it hasn't been saved yet.
    :type path: str or pathlib.Path or None
    :param Environment environment: Provide a predefined :ref:`Environment <environments>`
        to work with a specific Python version or virtualenv.
    :param Project project: Provide a :class:`.Project` to make sure finding
        references works well, because the right folder is searched. There are
        also ways to modify the sys path and other things.
    """

    def __init__(self, code: Optional[str] = None, *, path: Optional[Union[str, Path]] = None, 
                 environment: Optional[Any] = None, project: Optional[Project] = None) -> None:
        self._orig_path: Optional[Union[str, Path]] = path
        if isinstance(path, str):
            path = Path(path)
        self.path: Optional[Path] = path.absolute() if path else None
        if code is None:
            if path is None:
                raise ValueError('Must provide at least one of code or path')
            with open(path, 'rb') as f:
                code = f.read()
        if project is None:
            project = get_default_project(None if self.path is None else self.path.parent)
        self._inference_state: InferenceState = InferenceState(project, environment=environment, script_path=self.path)
        debug.speed('init')
        self._module_node, code = self._inference_state.parse_and_get_code(
            code=code, path=self.path, use_latest_grammar=path and path.suffix == '.pyi', 
            cache=False, diff_cache=settings.fast_parser, cache_path=settings.cache_directory)
        debug.speed('parsed')
        self._code_lines: List[str] = parso.split_lines(code, keepends=True)
        self._code: str = code
        cache.clear_time_caches()
        debug.reset_time()

    @cache.memoize_method
    def _get_module(self) -> ModuleValue:
        names: Optional[Tuple[str, ...]] = None
        is_package: bool = False
        if self.path is not None:
            import_names, is_p = transform_path_to_dotted(self._inference_state.get_sys_path(add_parent_paths=False), self.path)
            if import_names is not None:
                names = import_names
                is_package = is_p
        if self.path is None:
            file_io = None
        else:
            file_io = KnownContentFileIO(self.path, self._code)
        if self.path is not None and self.path.suffix == '.pyi':
            stub_module = load_proper_stub_module(self._inference_state, self._inference_state.latest_grammar, file_io, names, self._module_node)
            if stub_module is not None:
                return stub_module
        if names is None:
            names = ('__main__',)
        module = ModuleValue(self._inference_state, self._module_node, file_io=file_io, string_names=names, code_lines=self._code_lines, is_package=is_package)
        if names[0] not in ('builtins', 'typing'):
            self._inference_state.module_cache.add(names, ValueSet([module]))
        return module

    def _get_module_context(self) -> Any:
        return self._get_module().as_context()

    def __repr__(self) -> str:
        return '<%s: %s %r>' % (self.__class__.__name__, repr(self._orig_path), self._inference_state.environment)

    @validate_line_column
    def complete(self, line: Optional[int] = None, column: Optional[int] = None, *, fuzzy: bool = False) -> List[Completion]:
        """
        Completes objects under the cursor.

        Those objects contain information about the completions, more than just
        names.

        :param fuzzy: Default False. Will return fuzzy completions, which means
            that e.g. ``ooa`` will match ``foobar``.
        :return: Completion objects, sorted by name. Normal names appear
            before "private" names that start with ``_`` and those appear
            before magic methods and name mangled names that start with ``__``.
        :rtype: list of :class:`.Completion`
        """
        with debug.increase_indent_cm('complete'):
            completion = Completion(self._inference_state, self._get_module_context(), self._code_lines, (line, column), self.get_signatures, fuzzy=fuzzy)
            return completion.complete()

    @validate_line_column
    def infer(self, line: Optional[int] = None, column: Optional[int] = None, *, 
              only_stubs: bool = False, prefer_stubs: bool = False) -> List[classes.Name]:
        """
        Return the definitions of under the cursor. It is basically a wrapper
        around Jedi's type inference.

        This method follows complicated paths and returns the end, not the
        first definition. The big difference between :meth:`goto` and
        :meth:`infer` is that :meth:`goto` doesn't
        follow imports and statements. Multiple objects may be returned,
        because depending on an option you can have two different versions of a
        function.

        :param only_stubs: Only return stubs for this method.
        :param prefer_stubs: Prefer stubs to Python objects for this method.
        :rtype: list of :class:`.Name`
        """
        pos = (line, column)
        leaf = self._module_node.get_name_of_position(pos)
        if leaf is None:
            leaf = self._module_node.get_leaf_for_position(pos)
            if leaf is None or leaf.type == 'string':
                return []
            if leaf.end_pos == (line, column) and leaf.type == 'operator':
                next_ = leaf.get_next_leaf()
                if next_.start_pos == leaf.end_pos and next_.type in ('number', 'string', 'keyword'):
                    leaf = next_
        context = self._get_module_context().create_context(leaf)
        values = helpers.infer(self._inference_state, context, leaf)
        values = convert_values(values, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        defs = [classes.Name(self._inference_state, c.name) for c in values]
        return helpers.sorted_definitions(set(defs))

    @validate_line_column
    def goto(self, line: Optional[int] = None, column: Optional[int] = None, *, 
             follow_imports: bool = False, follow_builtin_imports: bool = False, 
             only_stubs: bool = False, prefer_stubs: bool = False) -> List[classes.Name]:
        """
        Goes to the name that defined the object under the cursor. Optionally
        you can follow imports.
        Multiple objects may be returned, depending on an if you can have two
        different versions of a function.

        :param follow_imports: The method will follow imports.
        :param follow_builtin_imports: If ``follow_imports`` is True will try
            to look up names in builtins (i.e. compiled or extension modules).
        :param only_stubs: Only return stubs for this method.
        :param prefer_stubs: Prefer stubs to Python objects for this method.
        :rtype: list of :class:`.Name`
        """
        tree_name = self._module_node.get_name_of_position((line, column))
        if tree_name is None:
            return self.infer(line, column, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        name = self._get_module_context().create_name(tree_name)
        names: List[Any] = []
        if name.tree_name.is_definition() and name.parent_context.is_class():
            class_node = name.parent_context.tree_node
            class_value = self._get_module_context().create_value(class_node)
            mro = class_value.py__mro__()
            next(mro)
            for cls in mro:
                names = cls.goto(tree_name.value)
                if names:
                    break
        if not names:
            names = list(name.goto())
        if follow_imports:
            names = helpers.filter_follow_imports(names, follow_builtin_imports)
        names = convert_names(names, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        defs = [classes.Name(self._inference_state, d) for d in set(names)]
        return list(set(helpers.sorted_definitions(defs)))

    def search(self, string: str, *, all_scopes: bool = False) -> List[classes.Name]:
        """
        Searches a name in the current file. For a description of how the
        search string should look like, please have a look at
        :meth:`.Project.search`.

        :param bool all_scopes: Default False; searches not only for
            definitions on the top level of a module level, but also in
            functions and classes.
        :yields: :class:`.Name`
        """
        return self._search_func(string, all_scopes=all_scopes)

    @to_list
    def _search_func(self, string: str, all_scopes: bool = False, 
                    complete: bool = False, fuzzy: bool = False) -> Iterator[classes.Name]:
        names = self._names(all_scopes=all_scopes)
        wanted_type, wanted_names = helpers.split_search_string(string)
        return search_in_module(self._inference_state, self._get_module_context(), names=names, wanted_type=wanted_type, wanted_names=wanted_names, complete=complete, fuzzy=fuzzy)

    def complete_search(self, string: str, **kwargs: Any) -> List[classes.Name]:
        """
        Like :meth:`.Script.search`, but completes that string. If you want to
        have all possible definitions in a file you can also provide an empty
        string.

        :param bool all_scopes: Default False; searches not only for
            definitions on the top level of a module level, but also in
            functions and classes.
        :param fuzzy: Default False. Will return fuzzy completions, which means
            that e.g. ``ooa`` will match ``foobar``.
        :yields: :class:`.Completion`
        """
        return self._search_func(string, complete=True, **kwargs)

    @validate_line_column
    def help(self, line: Optional[int] = None, column: Optional[int] = None) -> List[classes.Name]:
        """
        Used to display a help window to users.  Uses :meth:`.Script.goto` and
        returns additional definitions for keywords and operators.

        Typically you will want to display :meth:`.BaseName.docstring` to the
        user for all the returned definitions.

        The additional definitions are ``Name(...).type == 'keyword'``.
        These definitions do not have a lot of value apart from their docstring
        attribute, which contains the output of Python's :func:`help` function.

        :rtype: list of :class:`.Name`
        """
        definitions = self.goto(line, column, follow_imports=True)
        if definitions:
            return definitions
        leaf = self._module_node.get_leaf_for_position((line, column))
        if leaf is not None and leaf.type in ('keyword', 'operator', 'error_leaf'):

            def need_pydoc() -> bool:
                if leaf.value in ('(', ')', '[', ']'):
                    if leaf.parent.type == 'trailer':
                        return False
                    if leaf.parent.type == 'atom':
                        return False
                grammar = self._inference_state.grammar
                reserved = grammar._pgen_grammar.reserved_syntax_strings.keys()
                return leaf.value in reserved
            if need_pydoc():
                name = KeywordName(self._inference_state, leaf.value)
                return [classes.Name(self._inference_state, name)]
        return []

    @validate_line_column
    def get_references(self, line: Optional[int] = None, column: Optional[int] = None, **kwargs: Any) -> List[classes.Name]:
        """
        Lists all references of a variable in a project. Since this can be
        quite hard to do for Jedi, if it is too complicated, Jedi will stop
        searching.

        :param include_builtins: Default ``True``. If ``False``, checks if a definition
            is a builtin (e.g. ``sys``) and in that case does not return it.
        :param scope: Default ``'project'``. If ``'file'``, include references in
            the current module only.
        :rtype: list of :class:`.Name`
        """

        def _references(include_builtins: bool = True, scope: str = 'project') -> List[classes.Name]:
            if scope not in ('project', 'file'):
                raise ValueError('Only the scopes "file" and "project" are allowed')
            tree_name = self._module_node.get_name_of_position((line, column))
            if tree_name is None:
                return []
            names = find_references(self._get_module_context(), tree_name, scope == 'file')
            definitions = [classes.Name(self._inference_state, n) for n in names]
            if not include_builtins or scope == 'file':
                definitions = [d for d in definitions if not d.in_builtin_module()]
            return helpers.sorted_definitions(definitions)
        return _references(**kwargs)

    @validate_line_column
    def get_signatures(self, line: Optional[int] = None, column: Optional[int] = None) -> List[classes.Signature]:
        """
        Return the function object of the call under the cursor.

        E.g. if the cursor is here::

            abs(# <-- cursor is here

        This would return the ``abs`` function. On the other hand::

            abs()# <-- cursor is here

        This would return an empty list..

        :rtype: list of :class:`.Signature`
        """
        pos = (line, column)
        call_details = helpers.get_signature_details(self._module_node, pos)
        if call_details is None:
            return []
        context = self._get_module_context().create_context(call_details.bracket_leaf)
        definitions = helpers.cache_signatures(self._inference_state, context, call_details.bracket_leaf, self._code_lines, pos)
        debug.speed('func_call followed')
        return [classes.Signature(self._inference_state, signature, call_details) for signature in definitions.get_signatures()]

    @validate_line_column
    def get_context(self, line: Optional[int] = None, column: Optional[int] = None) -> classes.Name:
        """
        Returns the scope context under the cursor. This basically means the
        function, class or module where the cursor is at.

        :rtype: :class:`.Name`
        """
        pos = (line, column)
        leaf = self._module_node.get_leaf_for_position(pos, include_prefixes=True)
        if leaf.start_pos > pos or leaf.type == 'endmarker':
            previous_leaf = leaf.get_previous_leaf()
            if previous_leaf is not None:
                leaf = previous_leaf
        module_context = self._get_module_context()
        n = tree.search_ancestor(leaf, 'funcdef', 'classdef')
        if n is not None and n.start_pos < pos <= n.children[-1].start_pos:
            context = module_context.create_value(n).as_context()
        else:
            context = module_context.create_context(leaf)
        while context.name is None:
            context = context.parent_context
        definition = classes.Name(self._inference_state, context.name)
        while definition.type != 'module':
            name = definition._name
            tree_name = name.tree_name
            if tree_name is not None:
                scope = tree_name.get_definition()
                if scope.start_pos[1] < column:
                    break
            definition = definition.parent()
        return definition

    def _analysis(self) -> List[Any]:
        self._inference_state.is_analysis = True
        self._inference_state.analysis_modules = [self._module_node]
        module = self._get_module_context()
        try:
            for node in get_executable_nodes(self._module_node):
                context = module.create_context(node)
                if node.type in ('funcdef', 'classdef'):
                    tree_name_to_values(self._inference_state, context, node.children[1])
                elif isinstance(node, tree.Import):
                    import_names = set(node.get_defined_names())
                    if node.is_nested():
                        import_names |= set((path[-1] for path in node.get_paths()))
                    for n in import_names:
                        imports.infer_import(context, n)
                elif node.type == 'expr_stmt':
                    types = context.infer_node(node)
                    for testlist in node.children[:-1:2]:
                        unpack_tuple_to_dict(context, types, testlist)
                else:
                    if node.type == 'name':
                        defs = self._inference_state.infer(context, node)
                    else:
                        defs = infer_call_of_leaf(context, node)
                    try_iter_content(defs)
                self._inference_state.reset_recursion_limitations()
            ana = [a for a in self._inference_state.analysis if self.path == a.path]
            return sorted(set(ana), key=lambda x: x.line)
        finally:
            self._inference_state.is_analysis = False

    def get_names(self, **kwargs: Any) -> List[classes.Name]:
        """
        Returns names defined in the current file.

        :param all_scopes: If True lists the names of all scopes instead of
            only the module namespace.
        :param definitions: If True lists the names that have been defined by a
            class, function or a statement (``a = b`` returns ``a``).
        :param references: If True lists all the names that are not listed by
            ``definitions=True``. E.g. ``a = b`` returns ``b``.
        :rtype: list of :class:`.Name`
        """
        names = self._names(**kwargs)
        return [classes.Name(self._inference_state, n) for n in names]

    def get_syntax_errors(self) -> List[Any]:
        """
        Lists all syntax errors in the current file.

        :rtype: list of :class:`.SyntaxError`
        """
        return parso_to_jedi_errors(self._inference_state.grammar, self._module_node)

    def _names(self, all_scopes: bool = False, definitions: bool = True, references: bool = False) -> List[Any]:
        module_context = self._get_module_context()
        defs = [module_context.create_name(name) for name in helpers.get_module_names(self._module_node, all_scopes=all_scopes, definitions=definitions, references=references)]
        return sorted(defs, key=lambda x: x.start_pos)

    def rename(self, line: Optional[int] = None, column: Optional[int] = None, *, new_name: str) -> Any:
        """
        Renames all references of the variable under the cursor.

        :param new_name: The variable under the cursor will be renamed to this
            string.
        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        definitions = self.get_references(line, column, include_builtins=False)
        return refactoring.rename(self._inference_state, definitions, new_name)

    @validate_line_column
    def extract_variable(self, line: int, column: int, *, new_name: str, 
                         until_line: Optional[int] = None, until_column: Optional[int] = None) -> Any:
        """
        Moves an expression to a new statement.

        For example if you have the cursor on ``foo`` and provide a
        ``new_name`` called ``bar``::

            foo = 3.1
            x = int(foo + 1)

        the code above will become::

            foo = 3.1
            bar = foo + 1
            x = int(bar)

        :param new_name: The expression under the cursor will be renamed to
            this string.
        :param int until_line: The the selection range ends at this line, when
            omitted, Jedi will be clever and try to define the range itself.
        :param int until_column: The the selection range ends at this column, when
            omitted, Jedi will be clever and try to define the range itself.
        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        if until_line is None and until_column is None:
            until_pos = None
        else:
            if until_line is None:
                until_line = line
            if until_column is None:
                until_column = len(self._code_lines[until_line - 1])
            until_pos = (until_line, until_column)
        return extract_variable(self._inference_state, self.path, self._module_node, new_name, (line, column), until_pos)

    @validate_line_column
    def extract_function(self, line: int, column: int, *, new_name: str, 
                         until_line: Optional[int] = None, until_column: Optional[int] = None) -> Any:
        """
        Moves an expression to a new function.

        For example if you have the cursor on ``foo`` and provide a
        ``new_name`` called ``bar``::

            global_var = 3

            def x():
                foo = 3.1
                x = int(foo + 1 + global_var)

        the code above will become::

            global_var = 3

            def bar(foo):
                return int(foo + 1 + global_var)

            def x():
                foo = 3.1
                x = bar(foo)

        :param new_name: The expression under the cursor will be replaced with
            a function with this name.
        :param int until_line: The the selection range ends at this line, when
            omitted, Jedi will be clever and try to define the range itself.
        :param int until_column: The the selection range ends at this column, when
            omitted, Jedi will be clever and try to define the range itself.
        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        if until_line is None and until_column is None:
            until_pos = None
        else:
            if until_line is None:
                until_line = line
            if until_column is None:
                until_column = len(self._code_lines[until_line - 1])
            until_pos = (until_line, until_column)
        return extract_function(self._inference_state, self.path, self._get_module_context(), new_name, (line, column), until_pos)

    def inline(self, line: Optional[int] = None, column: Optional[int] = None) -> Any:
        """
        Inlines a variable under the cursor. This is basically the opposite of
        extracting a variable. For example with the cursor on bar::

            foo = 3.1
            bar = foo + 1
            x = int(bar)

        the code above will become::

            foo = 3.1
            x = int(foo + 1)

        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        names = [d._name for d in self.get_references(line, column, include_builtins=True)]
        return refactoring.inline(self._inference_state, names)

class Interpreter(Script):
    """
    Jedi's API for Python REPLs.

    Implements all of the methods that are present in :class:`.Script` as well.

    In addition to completions that normal REPL completion does like
    ``str.upper``, Jedi also supports code completion based on static code
    analysis. For example Jedi will complete ``str().upper``.

    >>> from os.path import join
    >>> namespace = locals()
    >>> script = Interpreter('join("").up', [namespace])
    >>> print(script.complete()[0].name)
    upper

    All keyword arguments are same as the arguments for :class:`.Script`.

    :param str code: Code to parse.
    :type namespaces: typing.List[dict]
    :param namespaces: A list of namespace dictionaries such as the one
        returned by :func:`globals` and :func:`locals`.
    """
    _allow_descriptor_getattr_default: bool = True

    def __init__(self, code: str, namespaces: List[Dict[str, Any]], *, project: Optional[Project] = None, **kwds: Any) -> None:
        try:
            namespaces = [dict(n) for n in namespaces]
        except Exception:
            raise TypeError('namespaces must be a non-empty list of dicts.')
        environment = kwds.get('environment', None)
        if environment is None:
            environment = InterpreterEnvironment()
        elif not isinstance(environment, InterpreterEnvironment):
            raise TypeError('The environment needs to be an InterpreterEnvironment subclass.')
        if project is None:
            project = Project(Path.cwd())
        super().__init__(code, environment=environment, project=project, **kwds)
        self.namespaces: List[Dict[str, Any]] = namespaces
        self._inference_state.allow_descriptor_getattr = self._allow_descriptor_getattr_default

    @cache.memoize_method
    def _get_module_context(self) -> Any:
        if self.path is None:
            file_io = None
        else:
            file_io = KnownContentFileIO(self.path, self._code)
        tree_module_value = ModuleValue(self._inference_state, self._module_node, file_io=file_io, string_names=('__main__',), code_lines=self._code_lines)
        return interpreter.MixedModuleContext(tree_module_value, self.namespaces)

def preload_module(*modules: str) -> None:
    """
    Preloading modules tells Jedi to load a module now, instead of lazy parsing
    of modules. This can be useful for IDEs, to control which modules to load
    on startup.

    :param modules: different module names, list of string.
    """
    for m in modules:
        s = 'import %s as x; x.' % m
        Script(s).complete(1, len(s))

def set_debug_function(func_cb: Optional[Callable[..., None]] = debug.print_to_stdout, 
                      warnings: bool = True, notices: bool = True, speed: bool = True) -> None:
    """
    Define a callback debug function to get all the debug messages.

    If you don't specify any arguments, debug messages will be printed to stdout.

    :param func_cb: The callback function for debug messages.
    """
    debug.debug_function = func_cb
    debug.enable_warning = warnings
    debug.enable_notice = notices
    debug.enable_speed = speed
