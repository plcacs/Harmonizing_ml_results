"""
:mod:`jedi.inference.imports` is here to resolve import statements and return
the modules/classes/functions/whatever, which they stand for. However there's
not any actual importing done. This module is about finding modules in the
filesystem. This can be quite tricky sometimes, because Python imports are not
always that simple.

This module also supports import autocompletion, which means to complete
statements like ``from datetim`` (cursor at the end would return ``datetime``).
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Any, Iterator, Union
from parso.python import tree
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.file_io import FolderIO, FileIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference import sys_path
from jedi.inference import helpers
from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.utils import unite
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.typeshed import import_module_decorator, create_stub_module, parse_stub_module
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.plugins import plugin_manager
from jedi.inference.state import InferenceState
from jedi.inference.context import Context
from jedi.inference.value.namespace import ImplicitNamespaceValue
from jedi.inference.value import ModuleValue
from jedi.inference.compiled import load_module as compiled_load_module


class ModuleCache:
    def __init__(self) -> None:
        self._name_cache: Dict[Optional[Tuple[str, ...]], ValueSet] = {}

    def add(self, string_names: Optional[Tuple[str, ...]], value_set: ValueSet) -> None:
        if string_names is not None:
            self._name_cache[string_names] = value_set

    def get(self, string_names: Tuple[str, ...]) -> Optional[ValueSet]:
        return self._name_cache.get(string_names)


@inference_state_method_cache(default=NO_VALUES)
def infer_import(context: Context, tree_name: tree.Name) -> ValueSet:
    module_context: Context = context.get_root_context()
    from_import_name: Optional[str]
    import_path: Tuple[str, ...]
    level: int
    values: ValueSet
    from_import_name, import_path, level, values = _prepare_infer_import(module_context, tree_name)
    if values:
        if from_import_name is not None:
            values = values.py__getattribute__(from_import_name, name_context=context, analysis_errors=False)
            if not values:
                path = import_path + (from_import_name,)
                importer = Importer(context.inference_state, path, module_context, level)
                values = importer.follow()
    debug.dbg('after import: %s', values)
    return values


@inference_state_method_cache(default=[])
def goto_import(context: Context, tree_name: tree.Name) -> Set[str]:
    module_context: Context = context.get_root_context()
    from_import_name: Optional[str]
    import_path: Tuple[str, ...]
    level: int
    values: ValueSet
    from_import_name, import_path, level, values = _prepare_infer_import(module_context, tree_name)
    if not values:
        return set()
    if from_import_name is not None:
        names: Set[Any] = unite([
            c.goto(from_import_name, name_context=context, analysis_errors=False)
            for c in values
        ])
        if names and not any((n.tree_name is tree_name for n in names)):
            return names
        path = import_path + (from_import_name,)
        importer = Importer(context.inference_state, path, module_context, level)
        values = importer.follow()
    return set(s.name for s in values)


def _prepare_infer_import(
    module_context: Context, 
    tree_name: tree.Name
) -> Tuple[Optional[str], Tuple[str, ...], int, ValueSet]:
    import_node: tree.BaseNode = search_ancestor(tree_name, 'import_name', 'import_from')
    import_path: Tuple[str, ...] = import_node.get_path_for_name(tree_name)
    from_import_name: Optional[str] = None
    try:
        from_names: List[str] = import_node.get_from_names()
    except AttributeError:
        pass
    else:
        if len(from_names) + 1 == len(import_path):
            from_import_name = import_path[-1]
            import_path = tuple(from_names)
    importer = Importer(
        module_context.inference_state, 
        tuple(import_path), 
        module_context, 
        import_node.level
    )
    return (from_import_name, tuple(import_path), import_node.level, importer.follow())


def _add_error(value: Optional[Any], name: Any, message: str) -> None:
    if hasattr(name, 'parent') and value is not None:
        analysis.add(value, 'import-error', name, message)
    else:
        debug.warning('ImportError without origin: ' + message)


def _level_to_base_import_path(
    project_path: str, 
    directory: str, 
    level: int
) -> Tuple[Optional[List[str]], str]:
    """
    In case the level is outside of the currently known package (something like
    import .....foo), we can still try our best to help the user for
    completions.
    """
    for _ in range(level - 1):
        old: str = directory
        directory = os.path.dirname(directory)
        if old == directory:
            return (None, None)
    d: str = directory
    level_import_paths: List[str] = []
    while True:
        if d == project_path:
            return (level_import_paths, d)
        dir_name: str = os.path.basename(d)
        if dir_name:
            level_import_paths.insert(0, dir_name)
            d = os.path.dirname(d)
        else:
            return (None, d)


class Importer:
    def __init__(
        self, 
        inference_state: InferenceState, 
        import_path: Tuple[str, ...], 
        module_context: Context, 
        level: int = 0
    ) -> None:
        """
        An implementation similar to ``__import__``. Use `follow`
        to actually follow the imports.

        *level* specifies whether to use absolute or relative imports. 0 (the
        default) means only perform absolute imports. Positive values for level
        indicate the number of parent directories to search relative to the
        directory of the module calling ``__import__()`` (see PEP 328 for the
        details).

        :param import_path: List of namespaces (strings or Names).
        """
        debug.speed('import %s %s' % (import_path, module_context))
        self._inference_state: InferenceState = inference_state
        self.level: int = level
        self._module_context: Context = module_context
        self._fixed_sys_path: Optional[List[str]] = None
        self._infer_possible: bool = True
        if level:
            base = module_context.get_value().py__package__()  # type: ignore
            if level <= len(base):
                base = tuple(base)
                if level > 1:
                    base = base[:-level + 1]
                import_path = base + tuple(import_path)
            else:
                path: Optional[str] = module_context.py__file__()  # type: ignore
                project_path: str = self._inference_state.project.path
                import_path = list(import_path)
                if path is None:
                    directory: str = project_path
                else:
                    directory = os.path.dirname(path)
                base_import_path: Optional[List[str]]
                base_directory: str
                base_import_path, base_directory = _level_to_base_import_path(project_path, directory, level)
                if base_directory is None:
                    self._infer_possible = False
                else:
                    self._fixed_sys_path = [base_directory]
                if base_import_path is None:
                    if import_path:
                        _add_error(
                            module_context, 
                            import_path[0], 
                            message='Attempted relative import beyond top-level package.'
                        )
                else:
                    import_path = base_import_path + import_path
        self.import_path: Tuple[str, ...] = tuple(import_path)

    @property
    def _str_import_path(self) -> Tuple[str, ...]:
        """Returns the import path as pure strings instead of `Name`."""
        return tuple(
            name.value if isinstance(name, tree.Name) else name 
            for name in self.import_path
        )

    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]:
        if self._fixed_sys_path is not None:
            return self._fixed_sys_path
        return self._inference_state.get_sys_path(add_init_paths=not is_completion) + [
            str(p) for p in sys_path.check_sys_path_modifications(self._module_context)
        ]

    def follow(self) -> ValueSet:
        if not self.import_path:
            if self._fixed_sys_path:
                import_path_str: str = os.path.basename(self._fixed_sys_path[0])
                ns: ImplicitNamespaceValue = ImplicitNamespaceValue(
                    self._inference_state, 
                    string_names=(import_path_str,), 
                    paths=self._fixed_sys_path
                )
                return ValueSet({ns})
            return NO_VALUES
        if not self._infer_possible:
            return NO_VALUES
        from_cache: Optional[ModuleValue] = self._inference_state.stub_module_cache.get(self._str_import_path)
        if from_cache is not None:
            return ValueSet({from_cache})
        from_cache = self._inference_state.module_cache.get(self._str_import_path)
        if from_cache is not None:
            return from_cache
        sys_path_list: List[str] = self._sys_path_with_modifications(is_completion=False)
        return import_module_by_names(
            self._inference_state, 
            self.import_path, 
            sys_path=sys_path_list, 
            module_context=self._module_context
        )

    def _get_module_names(
        self, 
        search_path: Optional[List[str]] = None, 
        in_module: Optional[Any] = None
    ) -> List[Union[ImportName, SubModuleName]]:
        """
        Get the names of all modules in the search_path. This means file names
        and not names defined in the files.
        """
        if search_path is None:
            sys_path_list = self._sys_path_with_modifications(is_completion=True)
        else:
            sys_path_list = search_path
        return list(
            iter_module_names(
                self._inference_state, 
                self._module_context, 
                sys_path_list, 
                module_cls=ImportName if in_module is None else SubModuleName, 
                add_builtin_modules=(search_path is None and in_module is None)
            )
        )

    def completion_names(
        self, 
        inference_state: InferenceState, 
        only_modules: bool = False
    ) -> List[Union[ImportName, SubModuleName]]:
        """
        :param only_modules: Indicates whether it's possible to import a
            definition that is not defined in a module.
        """
        if not self._infer_possible:
            return []
        names: List[Union[ImportName, SubModuleName]] = []
        if self.import_path:
            if self._str_import_path == ('flask', 'ext'):
                for mod in self._get_module_names():
                    modname: str = mod.string_name
                    if modname.startswith('flask_'):
                        extname: str = modname[len('flask_'):]
                        names.append(ImportName(self._module_context, extname))
                for dir in self._sys_path_with_modifications(is_completion=True):
                    flaskext: str = os.path.join(dir, 'flaskext')
                    if os.path.isdir(flaskext):
                        names += self._get_module_names([flaskext])
            values: ValueSet = self.follow()
            for value in values:
                if value.api_type not in ('module', 'namespace'):
                    continue
                if not value.is_compiled():
                    names += list(value.sub_modules_dict().values())
            if not only_modules:
                from jedi.inference.gradual.conversion import convert_values
                both_values: ValueSet = values | convert_values(values)
                for c in both_values:
                    for filter in c.get_filters():
                        names += list(filter.values())
        elif self.level:
            names += self._get_module_names(self._fixed_sys_path)
        else:
            names += self._get_module_names()
        return names


def import_module_by_names(
    inference_state: InferenceState, 
    import_names: Tuple[str, ...], 
    sys_path: Optional[List[str]] = None, 
    module_context: Optional[Context] = None, 
    prefer_stubs: bool = True
) -> ValueSet:
    if sys_path is None:
        sys_path = inference_state.get_sys_path()
    str_import_names: Tuple[str, ...] = tuple(
        i.value if isinstance(i, tree.Name) else i 
        for i in import_names
    )
    value_set: ValueSet = ValueSet({None})
    for i in range(len(import_names)):
        current_import: Tuple[str, ...] = str_import_names[:i + 1]
        parent_values: ValueSet = ValueSet.from_sets([
            import_module(
                inference_state, 
                current_import, 
                parent_module_value, 
                sys_path, 
                prefer_stubs=prefer_stubs
            ) 
            for parent_module_value in value_set
        ])
        value_set = parent_values
        if not value_set:
            message: str = 'No module named ' + '.'.join(str_import_names)
            if module_context is not None:
                _add_error(module_context, import_names[i], message)
            else:
                debug.warning(message)
            return NO_VALUES
    return value_set


@plugin_manager.decorate()
@import_module_decorator
def import_module(
    inference_state: InferenceState, 
    import_names: Tuple[str, ...], 
    parent_module_value: Optional[ModuleValue], 
    sys_path: List[str]
) -> ValueSet:
    """
    This method is very similar to importlib's `_gcd_import`.
    """
    if import_names[0] in settings.auto_import_modules:
        module: Optional[Any] = _load_builtin_module(inference_state, import_names, sys_path)
        if module is None:
            return NO_VALUES
        return ValueSet([module])
    module_name: str = '.'.join(import_names)
    if parent_module_value is None:
        file_io_or_ns: Optional[Union[FileIO, ImplicitNSInfo]] = inference_state.compiled_subprocess.get_module_info(
            string=import_names[-1], 
            full_name=module_name, 
            sys_path=sys_path, 
            is_global_search=True
        )[0]
        is_pkg: Optional[bool] = inference_state.compiled_subprocess.get_module_info(
            string=import_names[-1], 
            full_name=module_name, 
            sys_path=sys_path, 
            is_global_search=True
        )[1]
        if is_pkg is None:
            return NO_VALUES
    else:
        paths: Optional[List[str]] = parent_module_value.py__path__()  # type: ignore
        if paths is None:
            return NO_VALUES
        file_io_or_ns, is_pkg = inference_state.compiled_subprocess.get_module_info(
            string=import_names[-1], 
            path=paths, 
            full_name=module_name, 
            is_global_search=False
        )
        if is_pkg is None:
            return NO_VALUES
    if isinstance(file_io_or_ns, ImplicitNSInfo):
        module = ImplicitNamespaceValue(
            inference_state, 
            string_names=tuple(file_io_or_ns.name.split('.')), 
            paths=file_io_or_ns.paths
        )
    elif file_io_or_ns is None:
        module = _load_builtin_module(inference_state, import_names, sys_path)
        if module is None:
            return NO_VALUES
    else:
        module = _load_python_module(
            inference_state, 
            file_io_or_ns, 
            import_names=import_names, 
            is_package=is_pkg
        )
    if parent_module_value is None:
        debug.dbg('global search_module %s: %s', import_names[-1], module)
    else:
        debug.dbg('search_module %s in paths %s: %s', module_name, paths, module)
    return ValueSet([module])


def _load_python_module(
    inference_state: InferenceState, 
    file_io: FileIO, 
    import_names: Optional[Tuple[str, ...]] = None, 
    is_package: bool = False
) -> ModuleValue:
    module_node: tree.Module = inference_state.parse(
        file_io=file_io, 
        cache=True, 
        diff_cache=settings.fast_parser, 
        cache_path=settings.cache_directory
    )
    return ModuleValue(
        inference_state, 
        module_node, 
        file_io=file_io, 
        string_names=import_names, 
        code_lines=get_cached_code_lines(inference_state.grammar, file_io.path), 
        is_package=is_package
    )


def _load_builtin_module(
    inference_state: InferenceState, 
    import_names: Optional[Tuple[str, ...]] = None, 
    sys_path: Optional[List[str]] = None
) -> Optional[Any]:
    project = inference_state.project
    if sys_path is None:
        sys_path = inference_state.get_sys_path()
    if not project._load_unsafe_extensions:
        safe_paths = project._get_base_sys_path(inference_state)
        sys_path = [p for p in sys_path if p in safe_paths]
    dotted_name: str = '.'.join(import_names) if import_names else ''
    assert dotted_name is not None
    module = compiled_load_module(
        inference_state, 
        dotted_name=dotted_name, 
        sys_path=sys_path
    )
    if module is None:
        return None
    return module


def load_module_from_path(
    inference_state: InferenceState, 
    file_io: FileIO, 
    import_names: Optional[Tuple[str, ...]] = None, 
    is_package: Optional[bool] = None
) -> ModuleValue:
    """
    This should pretty much only be used for get_modules_containing_name. It's
    here to ensure that a random path is still properly loaded into the Jedi
    module structure.
    """
    path: Path = Path(file_io.path)
    if import_names is None:
        import_names, is_package = sys_path.transform_path_to_dotted(
            inference_state.get_sys_path(), 
            path
        )
    else:
        assert isinstance(is_package, bool)
    is_stub: bool = path.suffix == '.pyi'
    if is_stub:
        folder_io: FolderIO = file_io.get_parent_folder()
        if folder_io.path.endswith('-stubs'):
            folder_io = FolderIO(folder_io.path[:-6])
        if path.name == '__init__.pyi':
            python_file_io: FileIO = folder_io.get_file_io('__init__.py')
        else:
            python_file_io: FileIO = folder_io.get_file_io(import_names[-1] + '.py')
        try:
            v: ModuleValue = load_module_from_path(
                inference_state, 
                python_file_io, 
                import_names, 
                is_package=is_package
            )
            values: ValueSet = ValueSet([v])
        except FileNotFoundError:
            values: ValueSet = NO_VALUES
        return create_stub_module(
            inference_state, 
            inference_state.latest_grammar, 
            values, 
            parse_stub_module(inference_state, file_io), 
            file_io, 
            import_names
        )
    else:
        module: ModuleValue = _load_python_module(
            inference_state, 
            file_io, 
            import_names=import_names, 
            is_package=is_package
        )
        inference_state.module_cache.add(import_names, ValueSet([module]))
        return module


def load_namespace_from_path(
    inference_state: InferenceState, 
    folder_io: FolderIO
) -> ImplicitNamespaceValue:
    import_names, is_package = sys_path.transform_path_to_dotted(
        inference_state.get_sys_path(), 
        Path(folder_io.path)
    )
    return ImplicitNamespaceValue(
        inference_state, 
        import_names, 
        [folder_io.path]
    )


def follow_error_node_imports_if_possible(
    context: Context, 
    name: tree.Name
) -> Optional[ValueSet]:
    error_node: Optional[tree.ErrorNode] = tree.search_ancestor(name, 'error_node')
    if error_node is not None:
        start_index: int = 0
        for index, n in enumerate(error_node.children):
            if n.start_pos > name.start_pos:
                break
            if n == ';':
                start_index = index + 1
        nodes: List[tree.BaseNode] = error_node.children[start_index:]
        first_name: str = nodes[0].get_first_leaf().value
        if first_name in ('from', 'import'):
            is_import_from: bool = first_name == 'from'
            level: int
            names: Tuple[str, ...]
            level, names = helpers.parse_dotted_names(
                nodes, 
                is_import_from=is_import_from, 
                until_node=name
            )
            return Importer(
                context.inference_state, 
                names, 
                context.get_root_context(), 
                level
            ).follow()
    return None


def iter_module_names(
    inference_state: InferenceState, 
    module_context: Context, 
    search_path: List[str], 
    module_cls: Any = ImportName, 
    add_builtin_modules: bool = True
) -> Iterator[Any]:
    """
    Get the names of all modules in the search_path. This means file names
    and not names defined in the files.
    """
    if add_builtin_modules:
        for name in inference_state.compiled_subprocess.get_builtin_module_names():
            yield module_cls(module_context, name)
    for name in inference_state.compiled_subprocess.iter_module_names(search_path):
        yield module_cls(module_context, name)
