import os
import re
from typing import Any, Iterator, Iterable, Dict, Set, Tuple, Optional
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names

_IGNORE_FOLDERS: Tuple[str, ...] = ('.tox', '.venv', '.mypy_cache', 'venv', '__pycache__')
_OPENED_FILE_LIMIT: int = 2000
"\nStats from a 2016 Lenovo Notebook running Linux:\nWith os.walk, it takes about 10s to scan 11'000 files (without filesystem\ncaching). Once cached it only takes 5s. So it is expected that reading all\nthose files might take a few seconds, but not a lot more.\n"
_PARSED_FILE_LIMIT: int = 30
'\nFor now we keep the amount of parsed files really low, since parsing might take\neasily 100ms for bigger files.\n'


def _resolve_names(definition_names: Iterable[Any], avoid_names: Iterable[Any] = ()) -> Iterator[Any]:
    for name in definition_names:
        if name in avoid_names:
            continue
        if not isinstance(name, SubModuleName):
            yield name
        if name.api_type == 'module':
            yield from _resolve_names(name.goto(), definition_names)


def _dictionarize(names: Iterable[Any]) -> Dict[Any, Any]:
    return dict(((n if n.tree_name is None else n.tree_name, n) for n in names))


def _find_defining_names(module_context: Any, tree_name: Any) -> Set[Any]:
    found_names: Set[Any] = _find_names(module_context, tree_name)
    for name in list(found_names):
        found_names |= set(convert_names([name], only_stubs=not name.get_root_context().is_stub(), prefer_stub_to_compiled=False))
    found_names |= set(_find_global_variables(found_names, tree_name.value))
    for name in list(found_names):
        if name.api_type == 'param' or name.tree_name is None or name.tree_name.parent.type == 'trailer':
            continue
        found_names |= set(_add_names_in_same_context(name.parent_context, name.string_name))
    return set(_resolve_names(found_names))


def _find_names(module_context: Any, tree_name: Any) -> Set[Any]:
    name = module_context.create_name(tree_name)
    found_names: Set[Any] = set(name.goto())
    found_names.add(name)
    return set(_resolve_names(found_names))


def _add_names_in_same_context(context: Any, string_name: str) -> Iterator[Any]:
    if context.tree_node is None:
        return
    until_position: Optional[Any] = None
    while True:
        filter_ = ParserTreeFilter(parent_context=context, until_position=until_position)
        names = set(filter_.get(string_name))
        if not names:
            break
        yield from names
        ordered = sorted(names, key=lambda x: x.start_pos)
        until_position = ordered[0].start_pos


def _find_global_variables(names: Iterable[Any], search_name: str) -> Iterator[Any]:
    for name in names:
        if name.tree_name is None:
            continue
        module_context = name.get_root_context()
        try:
            method = module_context.get_global_filter
        except AttributeError:
            continue
        else:
            for global_name in method().get(search_name):
                yield global_name
                c = module_context.create_context(global_name.tree_name)
                yield from _add_names_in_same_context(c, global_name.string_name)


def find_references(module_context: Any, tree_name: Any, only_in_module: bool = False) -> Iterable[Any]:
    inf = module_context.inference_state
    search_name: str = tree_name.value
    try:
        inf.flow_analysis_enabled = False
        found_names = _find_defining_names(module_context, tree_name)
    finally:
        inf.flow_analysis_enabled = True
    found_names_dct: Dict[Any, Any] = _dictionarize(found_names)
    module_contexts: Iterable[Any] = [module_context]
    if not only_in_module:
        for m in set((d.get_root_context() for d in found_names)):
            if m != module_context and m.tree_node is not None and (inf.project.path in m.py__file__().parents):
                module_contexts = list(module_contexts) + [m]
    if only_in_module or any((n.api_type == 'param' for n in found_names)):
        potential_modules: Iterable[Any] = module_contexts
    else:
        potential_modules = get_module_contexts_containing_name(inf, module_contexts, search_name)
    non_matching_reference_maps: Dict[Any, list] = {}
    for module_context in potential_modules:
        for name_leaf in module_context.tree_node.get_used_names().get(search_name, []):
            new = _dictionarize(_find_names(module_context, name_leaf))
            if any((tree_name in found_names_dct for tree_name in new)):
                found_names_dct.update(new)
                for tree_name in new:
                    for dct in non_matching_reference_maps.get(tree_name, []):
                        found_names_dct.update(dct)
                    try:
                        del non_matching_reference_maps[tree_name]
                    except KeyError:
                        pass
            else:
                for name in new:
                    non_matching_reference_maps.setdefault(name, []).append(new)
    result = found_names_dct.values()
    if only_in_module:
        return [n for n in result if n.get_root_context() == module_context]
    return result


def _check_fs(inference_state: Any, file_io: Any, regex: re.Pattern) -> Optional[Any]:
    try:
        code = file_io.read()
    except FileNotFoundError:
        return None
    code = python_bytes_to_unicode(code, errors='replace')
    if not regex.search(code):
        return None
    new_file_io = KnownContentFileIO(file_io.path, code)
    m = load_module_from_path(inference_state, new_file_io)
    if m.is_compiled():
        return None
    return m.as_context()


def gitignored_lines(folder_io: Any, file_io: Any) -> Tuple[Set[Any], Set[Any]]:
    ignored_paths: Set[Any] = set()
    ignored_names: Set[Any] = set()
    for l in file_io.read().splitlines():
        if not l or l.startswith(b'#'):
            continue
        p = l.decode('utf-8', 'ignore')
        if p.startswith('/'):
            name = p[1:]
            if name.endswith(os.path.sep):
                name = name[:-1]
            ignored_paths.add(os.path.join(folder_io.path, name))
        else:
            ignored_names.add(p)
    return (ignored_paths, ignored_names)


def recurse_find_python_folders_and_files(folder_io: Any, except_paths: Iterable[Any] = ()) -> Iterator[Tuple[Optional[Any], Optional[Any]]]:
    except_paths_set: Set[Any] = set(except_paths)
    for root_folder_io, folder_ios, file_ios in folder_io.walk():
        for file_io in file_ios:
            path = file_io.path
            if path.suffix in ('.py', '.pyi'):
                if path not in except_paths_set:
                    yield (None, file_io)
            if path.name == '.gitignore':
                ignored_paths, ignored_names = gitignored_lines(root_folder_io, file_io)
                except_paths_set |= ignored_paths
        folder_ios[:] = [folder_io for folder_io in folder_ios if folder_io.path not in except_paths_set and folder_io.get_base_name() not in _IGNORE_FOLDERS]
        for folder_io in folder_ios:
            yield (folder_io, None)


def recurse_find_python_files(folder_io: Any, except_paths: Iterable[Any] = ()) -> Iterator[Any]:
    for folder_io_entry, file_io in recurse_find_python_folders_and_files(folder_io, except_paths):
        if file_io is not None:
            yield file_io


def _find_python_files_in_sys_path(inference_state: Any, module_contexts: Iterable[Any]) -> Iterator[Any]:
    sys_path = inference_state.get_sys_path()
    except_paths: Set[Any] = set()
    yielded_paths = [m.py__file__() for m in module_contexts]
    for module_context in module_contexts:
        file_io = module_context.get_value().file_io
        if file_io is None:
            continue
        folder_io = file_io.get_parent_folder()
        while True:
            path = folder_io.path
            if not any((path.startswith(p) for p in sys_path)) or path in except_paths:
                break
            for file_io in recurse_find_python_files(folder_io, except_paths):
                if file_io.path not in yielded_paths:
                    yield file_io
            except_paths.add(path)
            folder_io = folder_io.get_parent_folder()


def _find_project_modules(inference_state: Any, module_contexts: Iterable[Any]) -> Iterator[Any]:
    except_ = [m.py__file__() for m in module_contexts]
    yield from recurse_find_python_files(FolderIO(inference_state.project.path), except_)


def get_module_contexts_containing_name(inference_state: Any, module_contexts: Iterable[Any], name: str, limit_reduction: int = 1) -> Iterator[Any]:
    """
    Search a name in the directories of modules.

    :param limit_reduction: Divides the limits on opening/parsing files by this
        factor.
    """
    for module_context in module_contexts:
        if module_context.is_compiled():
            continue
        yield module_context
    if len(name) <= 2:
        return
    file_io_iterator = _find_project_modules(inference_state, module_contexts)
    yield from search_in_file_ios(inference_state, file_io_iterator, name, limit_reduction=limit_reduction)


def search_in_file_ios(inference_state: Any, file_io_iterator: Iterable[Any], name: str, limit_reduction: int = 1, complete: bool = False) -> Iterator[Any]:
    parse_limit = _PARSED_FILE_LIMIT / limit_reduction
    open_limit = _OPENED_FILE_LIMIT / limit_reduction
    file_io_count = 0
    parsed_file_count = 0
    regex: re.Pattern = re.compile(r'\b' + re.escape(name) + ('' if complete else r'\b'))
    for file_io in file_io_iterator:
        file_io_count += 1
        m = _check_fs(inference_state, file_io, regex)
        if m is not None:
            parsed_file_count += 1
            yield m
            if parsed_file_count >= parse_limit:
                dbg('Hit limit of parsed files: %s', parse_limit)
                break
        if file_io_count >= open_limit:
            dbg('Hit limit of opened files: %s', open_limit)
            break
