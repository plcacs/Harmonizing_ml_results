from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Set, Union
from dbt.clients.jinja import MacroGenerator, MacroStack
from dbt.contracts.graph.nodes import Macro
from dbt.exceptions import DuplicateMacroNameError, PackageNotFoundForMacroError
from dbt.include.global_project import PROJECT_NAME as GLOBAL_PROJECT_NAME

FlatNamespace = Dict[str, MacroGenerator]
NamespaceMember = Union[FlatNamespace, MacroGenerator]
FullNamespace = Dict[str, NamespaceMember]

class MacroNamespace(Mapping[str, MacroGenerator]):

    def __init__(
        self,
        global_namespace: FlatNamespace,
        local_namespace: FlatNamespace,
        global_project_namespace: FlatNamespace,
        packages: Dict[str, FlatNamespace]
    ) -> None:
        self.global_namespace: FlatNamespace = global_namespace
        self.local_namespace: FlatNamespace = local_namespace
        self.packages: Dict[str, FlatNamespace] = packages
        self.global_project_namespace: FlatNamespace = global_project_namespace

    def _search_order(self) -> Iterator[FlatNamespace]:
        yield self.local_namespace
        yield self.global_namespace
        yield self.packages
        yield {GLOBAL_PROJECT_NAME: self.global_project_namespace}
        yield self.global_project_namespace

    def _keys(self) -> Set[str]:
        keys: Set[str] = set()
        for search in self._search_order():
            keys.update(search)
        return keys

    def __iter__(self) -> Iterator[str]:
        for key in self._keys():
            yield key

    def __len__(self) -> int:
        return len(self._keys())

    def __getitem__(self, key: str) -> MacroGenerator:
        for dct in self._search_order():
            if key in dct:
                return dct[key]
        raise KeyError(key)

    def get_from_package(self, package_name: Optional[str], name: str) -> MacroGenerator:
        if package_name is None:
            return self.get(name)
        elif package_name == GLOBAL_PROJECT_NAME:
            macro = self.global_project_namespace.get(name)
            if macro is not None:
                return macro
        elif package_name in self.packages:
            macro = self.packages[package_name].get(name)
            if macro is not None:
                return macro
        raise PackageNotFoundForMacroError(package_name)


class MacroNamespaceBuilder:

    def __init__(
        self,
        root_package: str,
        search_package: str,
        thread_ctx: Any,
        internal_packages: List[str],
        node: Optional[Any] = None
    ) -> None:
        self.root_package: str = root_package
        self.search_package: str = search_package
        self.internal_package_names: Set[str] = set(internal_packages)
        self.internal_package_names_order: List[str] = internal_packages
        self.globals: FlatNamespace = {}
        self.locals: FlatNamespace = {}
        self.internal_packages: Dict[str, FlatNamespace] = {}
        self.packages: Dict[str, FlatNamespace] = {}
        self.thread_ctx: Any = thread_ctx
        self.node: Optional[Any] = node

    def _add_macro_to(
        self,
        hierarchy: Dict[str, FlatNamespace],
        macro: Macro,
        macro_func: MacroGenerator
    ) -> None:
        if macro.package_name in hierarchy:
            namespace = hierarchy[macro.package_name]
        else:
            namespace = {}
            hierarchy[macro.package_name] = namespace
        if macro.name in namespace:
            raise DuplicateMacroNameError(macro_func.macro, macro, macro.package_name)
        namespace[macro.name] = macro_func

    def add_macro(self, macro: Macro, ctx: Any) -> None:
        macro_name: str = macro.name
        macro_func: MacroGenerator = MacroGenerator(macro, ctx, self.node, self.thread_ctx)
        if macro.package_name in self.internal_package_names:
            self._add_macro_to(self.internal_packages, macro, macro_func)
        else:
            self._add_macro_to(self.packages, macro, macro_func)
            if macro.package_name == self.search_package:
                self.locals[macro_name] = macro_func
            elif macro.package_name == self.root_package:
                self.globals[macro_name] = macro_func

    def add_macros(self, macros: Iterable[Macro], ctx: Any) -> None:
        for macro in macros:
            self.add_macro(macro, ctx)

    def build_namespace(self, macros_by_package: Dict[str, Dict[str, Macro]], ctx: Any) -> MacroNamespace:
        for package_macros in macros_by_package.values():
            self.add_macros(package_macros.values(), ctx)
        global_project_namespace: FlatNamespace = {}
        for pkg in reversed(self.internal_package_names_order):
            package_macros = self.internal_packages.get(pkg)
            if package_macros:
                global_project_namespace.update(package_macros)
        return MacroNamespace(
            global_namespace=self.globals,
            local_namespace=self.locals,
            global_project_namespace=global_project_namespace,
            packages=self.packages
        )
