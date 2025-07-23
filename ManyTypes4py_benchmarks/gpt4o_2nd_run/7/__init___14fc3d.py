import sys
from pathlib import Path
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
from typing import Optional, List, Callable, Union, Generator
sys.setrecursionlimit(3000)

class Script:
    def __init__(self, code: Optional[str] = None, *, path: Optional[Union[str, Path]] = None, environment: Optional[InterpreterEnvironment] = None, project: Optional[Project] = None):
        self._orig_path = path
        if isinstance(path, str):
            path = Path(path)
        self.path = path.absolute() if path else None
        if code is None:
            if path is None:
                raise ValueError('Must provide at least one of code or path')
            with open(path, 'rb') as f:
                code = f.read()
        if project is None:
            project = get_default_project(None if self.path is None else self.path.parent)
        self._inference_state = InferenceState(project, environment=environment, script_path=self.path)
        debug.speed('init')
        self._module_node, code = self._inference_state.parse_and_get_code(code=code, path=self.path, use_latest_grammar=path and path.suffix == '.pyi', cache=False, diff_cache=settings.fast_parser, cache_path=settings.cache_directory)
        debug.speed('parsed')
        self._code_lines = parso.split_lines(code, keepends=True)
        self._code = code
        cache.clear_time_caches()
        debug.reset_time()

    @cache.memoize_method
    def _get_module(self) -> ModuleValue:
        names = None
        is_package = False
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

    def _get_module_context(self):
        return self._get_module().as_context()

    def __repr__(self) -> str:
        return '<%s: %s %r>' % (self.__class__.__name__, repr(self._orig_path), self._inference_state.environment)

    @validate_line_column
    def complete(self, line: Optional[int] = None, column: Optional[int] = None, *, fuzzy: bool = False) -> List[Completion]:
        with debug.increase_indent_cm('complete'):
            completion = Completion(self._inference_state, self._get_module_context(), self._code_lines, (line, column), self.get_signatures, fuzzy=fuzzy)
            return completion.complete()

    @validate_line_column
    def infer(self, line: Optional[int] = None, column: Optional[int] = None, *, only_stubs: bool = False, prefer_stubs: bool = False) -> List[classes.Name]:
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
    def goto(self, line: Optional[int] = None, column: Optional[int] = None, *, follow_imports: bool = False, follow_builtin_imports: bool = False, only_stubs: bool = False, prefer_stubs: bool = False) -> List[classes.Name]:
        tree_name = self._module_node.get_name_of_position((line, column))
        if tree_name is None:
            return self.infer(line, column, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        name = self._get_module_context().create_name(tree_name)
        names = []
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

    def search(self, string: str, *, all_scopes: bool = False) -> Generator[classes.Name, None, None]:
        return self._search_func(string, all_scopes=all_scopes)

    @to_list
    def _search_func(self, string: str, all_scopes: bool = False, complete: bool = False, fuzzy: bool = False) -> List[classes.Name]:
        names = self._names(all_scopes=all_scopes)
        wanted_type, wanted_names = helpers.split_search_string(string)
        return search_in_module(self._inference_state, self._get_module_context(), names=names, wanted_type=wanted_type, wanted_names=wanted_names, complete=complete, fuzzy=fuzzy)

    def complete_search(self, string: str, **kwargs) -> Generator[Completion, None, None]:
        return self._search_func(string, complete=True, **kwargs)

    @validate_line_column
    def help(self, line: Optional[int] = None, column: Optional[int] = None) -> List[classes.Name]:
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
    def get_references(self, line: Optional[int] = None, column: Optional[int] = None, **kwargs) -> List[classes.Name]:
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

    def _analysis(self) -> List:
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

    def get_names(self, **kwargs) -> List[classes.Name]:
        names = self._names(**kwargs)
        return [classes.Name(self._inference_state, n) for n in names]

    def get_syntax_errors(self) -> List:
        return parso_to_jedi_errors(self._inference_state.grammar, self._module_node)

    def _names(self, all_scopes: bool = False, definitions: bool = True, references: bool = False) -> List:
        module_context = self._get_module_context()
        defs = [module_context.create_name(name) for name in helpers.get_module_names(self._module_node, all_scopes=all_scopes, definitions=definitions, references=references)]
        return sorted(defs, key=lambda x: x.start_pos)

    def rename(self, line: Optional[int] = None, column: Optional[int] = None, *, new_name: str):
        definitions = self.get_references(line, column, include_builtins=False)
        return refactoring.rename(self._inference_state, definitions, new_name)

    @validate_line_column
    def extract_variable(self, line: int, column: int, *, new_name: str, until_line: Optional[int] = None, until_column: Optional[int] = None):
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
    def extract_function(self, line: int, column: int, *, new_name: str, until_line: Optional[int] = None, until_column: Optional[int] = None):
        if until_line is None and until_column is None:
            until_pos = None
        else:
            if until_line is None:
                until_line = line
            if until_column is None:
                until_column = len(self._code_lines[until_line - 1])
            until_pos = (until_line, until_column)
        return extract_function(self._inference_state, self.path, self._get_module_context(), new_name, (line, column), until_pos)

    def inline(self, line: Optional[int] = None, column: Optional[int] = None):
        names = [d._name for d in self.get_references(line, column, include_builtins=True)]
        return refactoring.inline(self._inference_state, names)

class Interpreter(Script):
    _allow_descriptor_getattr_default = True

    def __init__(self, code: str, namespaces: List[dict], *, project: Optional[Project] = None, **kwds):
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
        self.namespaces = namespaces
        self._inference_state.allow_descriptor_getattr = self._allow_descriptor_getattr_default

    @cache.memoize_method
    def _get_module_context(self):
        if self.path is None:
            file_io = None
        else:
            file_io = KnownContentFileIO(self.path, self._code)
        tree_module_value = ModuleValue(self._inference_state, self._module_node, file_io=file_io, string_names=('__main__',), code_lines=self._code_lines)
        return interpreter.MixedModuleContext(tree_module_value, self.namespaces)

def preload_module(*modules: str) -> None:
    for m in modules:
        s = 'import %s as x; x.' % m
        Script(s).complete(1, len(s))

def set_debug_function(func_cb: Callable = debug.print_to_stdout, warnings: bool = True, notices: bool = True, speed: bool = True) -> None:
    debug.debug_function = func_cb
    debug.enable_warning = warnings
    debug.enable_notice = notices
    debug.enable_speed = speed
