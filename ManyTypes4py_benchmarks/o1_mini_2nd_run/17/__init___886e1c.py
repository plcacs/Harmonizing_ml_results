import difflib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from parso import split_lines
from parso.python.tree import Node
from jedi.api.exceptions import RefactoringError

EXPRESSION_PARTS = 'or_test and_test not_test comparison expr xor_expr and_expr shift_expr arith_expr term factor power atom_expr'.split()


class ChangedFile:
    def __init__(
        self,
        inference_state: Any,
        from_path: Optional[Path],
        to_path: Optional[Path],
        module_node: Node,
        node_to_str_map: Dict[Node, str],
    ) -> None:
        self._inference_state = inference_state
        self._from_path = from_path
        self._to_path = to_path
        self._module_node = module_node
        self._node_to_str_map = node_to_str_map

    def get_diff(self) -> str:
        old_lines: List[str] = split_lines(self._module_node.get_code(), keepends=True)
        new_lines: List[str] = split_lines(self.get_new_code(), keepends=True)
        if old_lines[-1] != '':
            old_lines[-1] += '\n'
        if new_lines[-1] != '':
            new_lines[-1] += '\n'
        project_path: Path = self._inference_state.project.path
        if self._from_path is None:
            from_p: str = ''
        else:
            from_p = str(self._from_path.relative_to(project_path))
        if self._to_path is None:
            to_p: str = ''
        else:
            to_p = str(self._to_path.relative_to(project_path))
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=from_p,
            tofile=to_p
        )
        return ''.join(diff).rstrip(' ')

    def get_new_code(self) -> str:
        return self._inference_state.grammar.refactor(self._module_node, self._node_to_str_map)

    def apply(self) -> None:
        if self._from_path is None:
            raise RefactoringError('Cannot apply a refactoring on a Script with path=None')
        with open(self._from_path, 'w', newline='') as f:
            f.write(self.get_new_code())

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self._from_path}>'


class Refactoring:
    def __init__(
        self,
        inference_state: Any,
        file_to_node_changes: Dict[Path, Dict[Node, str]],
        renames: Iterable[Tuple[Path, Path]] = (),
    ) -> None:
        self._inference_state = inference_state
        self._renames = renames
        self._file_to_node_changes = file_to_node_changes

    def get_changed_files(self) -> Dict[Path, ChangedFile]:
        def calculate_to_path(p: Optional[Path]) -> Optional[Path]:
            if p is None:
                return p
            p_str = str(p)
            for from_, to in renames:
                from_str = str(from_)
                if p_str.startswith(from_str):
                    return Path(str(to) + p_str[len(from_str):])
            return Path(p_str)

        renames: List[Tuple[Path, Path]] = self.get_renames()
        changed_files: Dict[Path, ChangedFile] = {}
        for path, map_ in sorted(self._file_to_node_changes.items()):
            to_path = calculate_to_path(path)
            module_node = next(iter(map_)).get_root_node()
            changed_file = ChangedFile(
                self._inference_state,
                from_path=path,
                to_path=to_path,
                module_node=module_node,
                node_to_str_map=map_
            )
            changed_files[path] = changed_file
        return changed_files

    def get_renames(self) -> List[Tuple[Path, Path]]:
        """
        Files can be renamed in a refactoring.
        """
        return sorted(self._renames)

    def get_diff(self) -> str:
        text: str = ''
        project_path: Path = self._inference_state.project.path
        for from_, to in self.get_renames():
            text += f'rename from {from_.relative_to(project_path)}\nrename to {to.relative_to(project_path)}\n'
        for f in self.get_changed_files().values():
            text += f.get_diff()
        return text

    def apply(self) -> None:
        """
        Applies the whole refactoring to the files, which includes renames.
        """
        for f in self.get_changed_files().values():
            f.apply()
        for old, new in self.get_renames():
            old.rename(new)


def _calculate_rename(path: Optional[Path], new_name: str) -> Tuple[Optional[Path], Path]:
    if path is None:
        return (path, Path(new_name))
    dir_ = path.parent
    if path.name in ('__init__.py', '__init__.pyi'):
        return (path, dir_.parent.joinpath(new_name))
    return (path, dir_.joinpath(new_name + path.suffix))


def rename(
    inference_state: Any,
    definitions: Iterable[Any],
    new_name: str
) -> Refactoring:
    file_renames: set = set()
    file_tree_name_map: Dict[Optional[Path], Dict[Node, str]] = {}
    if not definitions:
        raise RefactoringError('There is no name under the cursor')
    for d in definitions:
        tree_name = d._name.tree_name
        if d.type == 'module' and tree_name is None:
            p: Optional[Path] = None if d.module_path is None else Path(d.module_path)
            file_renames.add(_calculate_rename(p, new_name))
        elif tree_name is not None:
            fmap = file_tree_name_map.setdefault(d.module_path, {})
            fmap[tree_name] = tree_name.prefix + new_name
    return Refactoring(inference_state, file_to_node_changes=file_tree_name_map, renames=file_renames)


def inline(
    inference_state: Any,
    names: Iterable[Any]
) -> Refactoring:
    if not names:
        raise RefactoringError('There is no name under the cursor')
    if any(n.api_type in ('module', 'namespace') for n in names):
        raise RefactoringError('Cannot inline imports, modules or namespaces')
    if any(n.tree_name is None for n in names):
        raise RefactoringError('Cannot inline builtins/extensions')
    definitions: List[Any] = [n for n in names if n.tree_name.is_definition()]
    if len(definitions) == 0:
        raise RefactoringError('No definition found to inline')
    if len(definitions) > 1:
        raise RefactoringError('Cannot inline a name with multiple definitions')
    if len(names) == 1:
        raise RefactoringError('There are no references to this name')
    tree_name = definitions[0].tree_name
    expr_stmt = tree_name.get_definition()
    if expr_stmt.type != 'expr_stmt':
        type_map = {'funcdef': 'function', 'classdef': 'class'}
        type_ = type_map.get(expr_stmt.type, expr_stmt.type)
        raise RefactoringError(f'Cannot inline a {type_}')
    if len(expr_stmt.get_defined_names(include_setitem=True)) > 1:
        raise RefactoringError('Cannot inline a statement with multiple definitions')
    first_child = expr_stmt.children[1]
    if first_child.type == 'annassign' and len(first_child.children) == 4:
        first_child = first_child.children[2]
    if first_child != '=':
        if first_child.type == 'annassign':
            raise RefactoringError('Cannot inline a statement that is defined by an annotation')
        else:
            raise RefactoringError(f'Cannot inline a statement with "{first_child.get_code(include_prefix=False)}"')
    rhs = expr_stmt.get_rhs()
    replace_code: str = rhs.get_code(include_prefix=False)
    references: List[Any] = [n for n in names if not n.tree_name.is_definition()]
    file_to_node_changes: Dict[Path, Dict[Node, str]] = {}
    for name in references:
        tree_name = name.tree_name
        path: Path = name.get_root_context().py__file__()
        s: str = replace_code
        if (
            rhs.type == 'testlist_star_expr'
            or tree_name.parent.type in EXPRESSION_PARTS
            or (
                tree_name.parent.type == 'trailer'
                and tree_name.parent.get_next_sibling() is not None
            )
        ):
            s = f'({replace_code})'
        of_path = file_to_node_changes.setdefault(path, {})
        n = tree_name
        prefix = n.prefix
        par = n.parent
        if par.type == 'trailer' and par.children[0] == '.':
            prefix = par.parent.children[0].prefix
            n = par
            for some_node in par.parent.children[:par.parent.children.index(par)]:
                of_path[some_node] = ''
        of_path[n] = prefix + s
    path = definitions[0].get_root_context().py__file__()
    changes = file_to_node_changes.setdefault(path, {})
    changes[expr_stmt] = _remove_indent_of_prefix(expr_stmt.get_first_leaf().prefix)
    next_leaf = expr_stmt.get_next_leaf()
    if next_leaf.prefix.strip(' \t') == '' and (next_leaf.type == 'newline' or next_leaf == ';'):
        changes[next_leaf] = ''
    return Refactoring(inference_state, file_to_node_changes=file_to_node_changes)


def _remove_indent_of_prefix(prefix: str) -> str:
    """
    Removes the last indentation of a prefix, e.g. " \n \n " becomes " \n \n".
    """
    lines = split_lines(prefix, keepends=True)
    return ''.join(lines[:-1])
