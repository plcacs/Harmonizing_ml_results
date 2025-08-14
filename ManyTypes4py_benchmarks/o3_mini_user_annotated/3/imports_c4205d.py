from typing import Any, Optional, Tuple, List, Sequence, Set, Iterator, Union
import os
from pathlib import Path

from parso.python import tree
from parso.tree import search_ancestor

from jedi import debug
from jedi import settings
from jedi.file_io import FolderIO
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

class ModuleCache:
    def __init__(self) -> None:
        self._name_cache: dict[Tuple[str, ...], ValueSet] = {}

    def add(self, string_names: Optional[Tuple[str, ...]], value_set: ValueSet) -> None:
        if string_names is not None:
            self._name_cache[string_names] = value_set

    def get(self, string_names: Tuple[str, ...]) -> Optional[ValueSet]:
        return self._name_cache.get(string_names)


@inference_state_method_cache(default=NO_VALUES)
def infer_import(context: Any, tree_name: Any) -> ValueSet:
    module_context = context.get_root_context()
    from_import_name, import_path, level, values = _prepare_infer_import(module_context, tree_name)
    if values:
        if from_import_name is not None:
            values = values.py__getattribute__(
                from_import_name,
                name_context=context,
                analysis_errors=False
            )
            if not values:
                path = import_path + (from_import_name,)
                importer = Importer(context.inference_state, path, module_context, level)
                values = importer.follow()
    debug.dbg('after import: %s', values)
    return values


@inference_state_method_cache(default=[])
def goto_import(context: Any, tree_name: Any) -> Set[str]:
    module_context = context.get_root_context()
    from_import_name, import_path, level, values = _prepare_infer_import(module_context, tree_name)
    if not values:
        return set()

    if from_import_name is not None:
        names = unite([
            c.goto(
                from_import_name,
                name_context=context,
                analysis_errors=False
            ) for c in values
        ])
        if names and not any(n.tree_name is tree_name for n in names):
            return names

        path = import_path + (from_import_name,)
        importer = Importer(context.inference_state, path, module_context, level)
        values = importer.follow()
    return set(s.name for s in values)


def _prepare_infer_import(module_context: Any, tree_name: Any) -> Tuple[Optional[str], Tuple[str, ...], int, ValueSet]:
    import_node = search_ancestor(tree_name, 'import_name', 'import_from')
    import_path = import_node.get_path_for_name(tree_name)
    from_import_name: Optional[str] = None
    try:
        from_names = import_node.get_from_names()
    except AttributeError:
        pass
    else:
        if len(from_names) + 1 == len(import_path):
            from_import_name = import_path[-1]
            import_path = from_names

    importer = Importer(module_context.inference_state, tuple(import_path), module_context, import_node.level)
    return from_import_name, tuple(import_path), import_node.level, importer.follow()


def _add_error(value: Any, name: Any, message: str) -> None:
    if hasattr(name, 'parent') and value is not None:
        analysis.add(value, 'import-error', name, message)
    else:
        debug.warning('ImportError without origin: ' + message)


def _level_to_base_import_path(project_path: str, directory: str, level: int) -> Tuple[Optional[List[str]], Optional[str]]:
    for i in range(level - 1):
        old = directory
        directory = os.path.dirname(directory)
        if old == directory:
            return None, None

    d = directory
    level_import_paths: List[str] = []
    while True:
        if d == project_path:
            return level_import_paths, d
        dir_name = os.path.basename(d)
        if dir_name:
            level_import_paths.insert(0, dir_name)
            d = os.path.dirname(d)
        else:
            return None, directory


class Importer:
    def __init__(self, inference_state: Any, import_path: Tuple[Union[tree.Name, str], ...],
                 module_context: Any, level: int = 0) -> None:
        debug.speed('import %s %s' % (import_path, module_context))
        self._inference_state = inference_state
        self.level = level
        self._module_context = module_context

        self._fixed_sys_path: Optional[List[str]] = None
        self._infer_possible: bool = True
        if level:
            base = module_context.get_value().py__package__()
            if level <= len(base):
                base = tuple(base)
                if level > 1:
                    base = base[:-level + 1]
                import_path = base + tuple(import_path)
            else:
                path = module_context.py__file__()
                project_path = self._inference_state.project.path
                import_path = list(import_path)
                if path is None:
                    directory = project_path
                else:
                    directory = os.path.dirname(path)

                base_import_path, base_directory = _level_to_base_import_path(project_path, directory, level)
                if base_directory is None:
                    self._infer_possible = False
                else:
                    self._fixed_sys_path = [base_directory]

                if base_import_path is None:
                    if import_path:
                        _add_error(module_context, import_path[0],
                                   message='Attempted relative import beyond top-level package.')
                else:
                    import_path = base_import_path + import_path
        self.import_path: Tuple[Union[tree.Name, str], ...] = tuple(import_path)

    @property
    def _str_import_path(self) -> Tuple[str, ...]:
        return tuple(
            name.value if isinstance(name, tree.Name) else name
            for name in self.import_path
        )

    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]:
        if self._fixed_sys_path is not None:
            return self._fixed_sys_path

        return (self._inference_state.get_sys_path(add_init_paths=not is_completion)
                + [str(p) for p in sys_path.check_sys_path_modifications(self._module_context)])

    def follow(self) -> ValueSet:
        if not self.import_path:
            if self._fixed_sys_path:
                from jedi.inference.value.namespace import ImplicitNamespaceValue
                import_path = (os.path.basename(self._fixed_sys_path[0]),)
                ns = ImplicitNamespaceValue(
                    self._inference_state,
                    string_names=import_path,
                    paths=self._fixed_sys_path,
                )
                return ValueSet({ns})
            return NO_VALUES
        if not self._infer_possible:
            return NO_VALUES

        from_cache = self._inference_state.stub_module_cache.get(self._str_import_path)
        if from_cache is not None:
            return ValueSet({from_cache})
        from_cache = self._inference_state.module_cache.get(self._str_import_path)
        if from_cache is not None:
            return from_cache

        sys_path_list = self._sys_path_with_modifications(is_completion=False)

        return import_module_by_names(
            self._inference_state, self.import_path, sys_path_list, self._module_context
        )

    def _get_module_names(self, search_path: Optional[List[str]] = None, in_module: Optional[Any] = None
                          ) -> List[Union[ImportName, SubModuleName]]:
        if search_path is None:
            sys_path_list = self._sys_path_with_modifications(is_completion=True)
        else:
            sys_path_list = search_path
        return list(iter_module_names(
            self._inference_state, self._module_context, sys_path_list,
            module_cls=ImportName if in_module is None else SubModuleName,
            add_builtin_modules=search_path is None and in_module is None,
        ))

    def completion_names(self, inference_state: Any, only_modules: bool = False) -> List[Any]:
        if not self._infer_possible:
            return []

        names: List[Any] = []
        if self.import_path:
            if self._str_import_path == ('flask', 'ext'):
                for mod in self._get_module_names():
                    modname = mod.string_name
                    if modname.startswith('flask_'):
                        extname = modname[len('flask_'):]
                        names.append(ImportName(self._module_context, extname))
                for dir in self._sys_path_with_modifications(is_completion=True):
                    flaskext = os.path.join(dir, 'flaskext')
                    if os.path.isdir(flaskext):
                        names += self._get_module_names([flaskext])
            values = self.follow()
            for value in values:
                if value.api_type not in ('module', 'namespace'):
                    continue
                if not value.is_compiled():
                    names += value.sub_modules_dict().values()
            if not only_modules:
                from jedi.inference.gradual.conversion import convert_values
                both_values = values | convert_values(values)
                for c in both_values:
                    for filter in c.get_filters():
                        names += filter.values()
        else:
            if self.level:
                names += self._get_module_names(self._fixed_sys_path)  # type: ignore
            else:
                names += self._get_module_names()
        return names


def import_module_by_names(inference_state: Any, import_names: Sequence[Union[tree.Name, str]],
                           sys_path: Optional[List[str]] = None, module_context: Optional[Any] = None,
                           prefer_stubs: bool = True) -> ValueSet:
    if sys_path is None:
        sys_path = inference_state.get_sys_path()

    str_import_names: Tuple[str, ...] = tuple(
        i.value if isinstance(i, tree.Name) else i
        for i in import_names
    )
    value_set: Any = [None]
    for i, name in enumerate(import_names):
        value_set = ValueSet.from_sets([
            import_module(
                inference_state,
                str_import_names[:i+1],
                parent_module_value,
                sys_path,
                prefer_stubs=prefer_stubs,
            ) for parent_module_value in value_set
        ])
        if not value_set:
            message = 'No module named ' + '.'.join(str_import_names)
            if module_context is not None:
                _add_error(module_context, name, message)
            else:
                debug.warning(message)
            return NO_VALUES
    return value_set


@plugin_manager.decorate()
@import_module_decorator
def import_module(inference_state: Any, import_names: Sequence[Union[tree.Name, str]],
                  parent_module_value: Optional[Any], sys_path: List[str]) -> ValueSet:
    if import_names[0] in settings.auto_import_modules:
        module = _load_builtin_module(inference_state, import_names, sys_path)
        if module is None:
            return NO_VALUES
        return ValueSet([module])

    module_name: str = '.'.join(import_names)  # type: ignore
    if parent_module_value is None:
        file_io_or_ns, is_pkg = inference_state.compiled_subprocess.get_module_info(
            string=import_names[-1] if isinstance(import_names[-1], str) else import_names[-1].value,
            full_name=module_name,
            sys_path=sys_path,
            is_global_search=True,
        )
        if is_pkg is None:
            return NO_VALUES
    else:
        paths = parent_module_value.py__path__()
        if paths is None:
            return NO_VALUES

        file_io_or_ns, is_pkg = inference_state.compiled_subprocess.get_module_info(
            string=import_names[-1] if isinstance(import_names[-1], str) else import_names[-1].value,
            path=paths,
            full_name=module_name,
            is_global_search=False,
        )
        if is_pkg is None:
            return NO_VALUES

    if isinstance(file_io_or_ns, ImplicitNSInfo):
        from jedi.inference.value.namespace import ImplicitNamespaceValue
        module = ImplicitNamespaceValue(
            inference_state,
            string_names=tuple(file_io_or_ns.name.split('.')),
            paths=file_io_or_ns.paths,
        )
    elif file_io_or_ns is None:
        module = _load_builtin_module(inference_state, import_names, sys_path)
        if module is None:
            return NO_VALUES
    else:
        module = _load_python_module(
            inference_state, file_io_or_ns,
            import_names=import_names,
            is_package=is_pkg,
        )

    if parent_module_value is None:
        debug.dbg('global search_module %s: %s', import_names[-1], module)
    else:
        debug.dbg('search_module %s in paths %s: %s', module_name, paths, module)
    return ValueSet([module])


def _load_python_module(inference_state: Any, file_io: Any,
                        import_names: Optional[Sequence[Union[tree.Name, str]]] = None,
                        is_package: bool = False) -> Any:
    module_node = inference_state.parse(
        file_io=file_io,
        cache=True,
        diff_cache=settings.fast_parser,
        cache_path=settings.cache_directory,
    )

    from jedi.inference.value import ModuleValue
    return ModuleValue(
        inference_state, module_node,
        file_io=file_io,
        string_names=import_names,
        code_lines=get_cached_code_lines(inference_state.grammar, file_io.path),
        is_package=is_package,
    )


def _load_builtin_module(inference_state: Any, import_names: Optional[Sequence[Union[tree.Name, str]]] = None,
                         sys_path: Optional[List[str]] = None) -> Optional[Any]:
    project = inference_state.project
    if sys_path is None:
        sys_path = inference_state.get_sys_path()
    if not project._load_unsafe_extensions:
        safe_paths = project._get_base_sys_path(inference_state)
        sys_path = [p for p in sys_path if p in safe_paths]

    dotted_name: str = '.'.join(import_names)  # type: ignore
    assert dotted_name is not None
    module = compiled.load_module(inference_state, dotted_name=dotted_name, sys_path=sys_path)
    if module is None:
        return None
    return module


def load_module_from_path(inference_state: Any, file_io: Any,
                          import_names: Optional[Sequence[Union[tree.Name, str]]] = None,
                          is_package: Optional[bool] = None) -> Any:
    path = Path(file_io.path)
    if import_names is None:
        e_sys_path = inference_state.get_sys_path()
        import_names, is_package = sys_path.transform_path_to_dotted(e_sys_path, path)
    else:
        assert isinstance(is_package, bool)

    is_stub = path.suffix == '.pyi'
    if is_stub:
        folder_io = file_io.get_parent_folder()
        if folder_io.path.endswith('-stubs'):
            folder_io = FolderIO(folder_io.path[:-6])
        if path.name == '__init__.pyi':
            python_file_io = folder_io.get_file_io('__init__.py')
        else:
            python_file_io = folder_io.get_file_io(import_names[-1] + '.py')  # type: ignore

        try:
            v = load_module_from_path(
                inference_state, python_file_io,
                import_names, is_package=is_package
            )
            values = ValueSet([v])
        except FileNotFoundError:
            values = NO_VALUES

        return create_stub_module(
            inference_state, inference_state.latest_grammar, values,
            parse_stub_module(inference_state, file_io), file_io, import_names  # type: ignore
        )
    else:
        module = _load_python_module(
            inference_state, file_io,
            import_names=import_names,
            is_package=is_package,  # type: ignore
        )
        inference_state.module_cache.add(import_names, ValueSet([module]))  # type: ignore
        return module


def load_namespace_from_path(inference_state: Any, folder_io: Any) -> Any:
    import_names, is_package = sys_path.transform_path_to_dotted(
        inference_state.get_sys_path(),
        Path(folder_io.path)
    )
    from jedi.inference.value.namespace import ImplicitNamespaceValue
    return ImplicitNamespaceValue(inference_state, import_names, [folder_io.path])


def follow_error_node_imports_if_possible(context: Any, name: Any) -> Optional[ValueSet]:
    error_node = tree.search_ancestor(name, 'error_node')
    if error_node is not None:
        start_index = 0
        for index, n in enumerate(error_node.children):
            if n.start_pos > name.start_pos:
                break
            if n == ';':
                start_index = index + 1
        nodes = error_node.children[start_index:]
        first_name = nodes[0].get_first_leaf().value

        if first_name in ('from', 'import'):
            is_import_from = first_name == 'from'
            level, names = helpers.parse_dotted_names(
                nodes,
                is_import_from=is_import_from,
                until_node=name,
            )
            return Importer(
                context.inference_state, names, context.get_root_context(), level
            ).follow()
    return None


def iter_module_names(inference_state: Any, module_context: Any, search_path: List[str],
                      module_cls: Any = ImportName, add_builtin_modules: bool = True
                      ) -> Iterator[Any]:
    if add_builtin_modules:
        for name in inference_state.compiled_subprocess.get_builtin_module_names():
            yield module_cls(module_context, name)

    for name in inference_state.compiled_subprocess.iter_module_names(search_path):
        yield module_cls(module_context, name)