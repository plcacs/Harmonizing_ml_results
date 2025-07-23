import os
from pathlib import Path
from typing import Optional, Iterator, Dict, List, Tuple, Union
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import AbstractNameDefinition, ModuleName, SubModuleName
from jedi.inference.filters import GlobalNameFilter, ParserTreeFilter, DictFilter, MergedFilter
from jedi.inference import compiled
from jedi.inference.base_value import TreeValue, ValueSet
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.compiled import create_simple_object
from jedi.inference.context import ModuleContext
from jedi.inference.value import ModuleValue
from jedi.tree import TreeNode
from jedi.inference.state import InferenceState
from jedi.inference.imports import Importer


class _ModuleAttributeName(AbstractNameDefinition):
    """
    For module attributes like __file__, __str__ and so on.
    """
    api_type: str = 'instance'

    def __init__(self, parent_module: ModuleValue, string_name: str, string_value: Optional[str] = None) -> None:
        self.parent_context = parent_module
        self.string_name = string_name
        self._string_value = string_value

    def infer(self) -> ValueSet:
        if self._string_value is not None:
            s = self._string_value
            return ValueSet([create_simple_object(self.parent_context.inference_state, s)])
        return compiled.get_string_value_set(self.parent_context.inference_state)


class SubModuleDictMixin:

    @inference_state_method_cache()
    def sub_modules_dict(self) -> Dict[str, SubModuleName]:
        """
        Lists modules in the directory of this module (if this module is a
        package).
        """
        names: Dict[str, SubModuleName] = {}
        if self.is_package():
            path = self.py__path__()
            if path is not None:
                mods = self.inference_state.compiled_subprocess.iter_module_names(path)
                for name in mods:
                    names[name] = SubModuleName(self.as_context(), name)
        return names


class ModuleMixin(SubModuleDictMixin):
    _module_name_class: type = ModuleName

    def get_filters(self, origin_scope: Optional['FilterScope'] = None) -> Iterator['BaseFilter']:
        yield MergedFilter(
            ParserTreeFilter(parent_context=self.as_context(), origin_scope=origin_scope),
            GlobalNameFilter(self.as_context())
        )
        yield DictFilter(self.sub_modules_dict())
        yield DictFilter(self._module_attributes_dict())
        yield from self.iter_star_filters()

    def py__class__(self) -> 'Value':
        c, = values_from_qualified_names(self.inference_state, 'types', 'ModuleType')
        return c

    def is_module(self) -> bool:
        return True

    def is_stub(self) -> bool:
        return False

    @property
    @inference_state_method_cache()
    def name(self) -> ModuleName:
        return self._module_name_class(self, self.string_names[-1])

    @inference_state_method_cache()
    def _module_attributes_dict(self) -> Dict[str, _ModuleAttributeName]:
        names = ['__package__', '__doc__', '__name__']
        dct: Dict[str, _ModuleAttributeName] = {n: _ModuleAttributeName(self, n) for n in names}
        path = self.py__file__()
        if path is not None:
            dct['__file__'] = _ModuleAttributeName(self, '__file__', str(path))
        return dct

    def iter_star_filters(self) -> Iterator['BaseFilter']:
        for star_module in self.star_imports():
            f = next(star_module.get_filters(), None)
            assert f is not None
            yield f

    @inference_state_method_cache([])
    def star_imports(self) -> List[ModuleValue]:
        modules: List[ModuleValue] = []
        module_context = self.as_context()
        for i in self.tree_node.iter_imports():
            if i.is_star_import():
                new_modules = Importer(
                    self.inference_state,
                    import_path=i.get_paths()[-1],
                    module_context=module_context,
                    level=i.level
                ).follow()
                for module in new_modules:
                    if isinstance(module, ModuleValue):
                        modules.extend(module.star_imports())
                modules.extend(new_modules)
        return modules

    def get_qualified_names(self) -> Tuple[str, ...]:
        """
        A module doesn't have a qualified name, but it's important to note that
        it's reachable and not `None`. With this information we can add
        qualified names on top for all value children.
        """
        return ()


class ModuleValue(ModuleMixin, TreeValue):
    api_type: str = 'module'

    def __init__(
        self,
        inference_state: InferenceState,
        module_node: TreeNode,
        code_lines: List[str],
        file_io: Optional['FileIO'] = None,
        string_names: Optional[List[str]] = None,
        is_package: bool = False
    ) -> None:
        super().__init__(inference_state, parent_context=None, tree_node=module_node)
        self.file_io = file_io
        if file_io is None:
            self._path: Optional[Path] = None
        else:
            self._path = file_io.path
        self.string_names: Optional[List[str]] = string_names
        self.code_lines: List[str] = code_lines
        self._is_package: bool = is_package

    def is_stub(self) -> bool:
        if self._path is not None and self._path.suffix == '.pyi':
            return True
        return super().is_stub()

    def py__name__(self) -> Optional[str]:
        if self.string_names is None:
            return None
        return '.'.join(self.string_names)

    def py__file__(self) -> Optional[Path]:
        """
        In contrast to Python's __file__ can be None.
        """
        if self._path is None:
            return None
        return self._path.absolute()

    def is_package(self) -> bool:
        return self._is_package

    def py__package__(self) -> List[str]:
        if self.string_names is None:
            return []
        if self._is_package:
            return self.string_names
        return self.string_names[:-1]

    def py__path__(self) -> Optional[List[str]]:
        """
        In case of a package, this returns Python's __path__ attribute, which
        is a list of paths (strings).
        Returns None if the module is not a package.
        """
        if not self._is_package:
            return None
        first_few_lines = ''.join(self.code_lines[:50])
        options = ('declare_namespace(__name__)', 'extend_path(__path__')
        if any(option in first_few_lines for option in options):
            paths: set[str] = set()
            for s in self.inference_state.get_sys_path():
                other = os.path.join(s, self.name.string_name)
                if os.path.isdir(other):
                    paths.add(other)
            if paths:
                return list(paths)
        file = self.py__file__()
        assert file is not None
        return [os.path.dirname(file)]

    def _as_context(self) -> ModuleContext:
        return ModuleContext(self)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.py__name__()}@{self.tree_node.start_pos[0]}-{self.tree_node.end_pos[0]} is_stub={self.is_stub()}>'
