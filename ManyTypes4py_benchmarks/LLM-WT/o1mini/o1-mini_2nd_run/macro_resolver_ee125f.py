from typing import Dict, MutableMapping, Optional, List, Set
from dbt.clients.jinja import MacroGenerator
from dbt.contracts.graph.nodes import Macro
from dbt.exceptions import DuplicateMacroNameError, PackageNotFoundForMacroError
from dbt.include.global_project import PROJECT_NAME as GLOBAL_PROJECT_NAME

MacroNamespace = Dict[str, Macro]

class MacroResolver:

    def __init__(
        self,
        macros: Dict[str, Macro],
        root_project_name: str,
        internal_package_names: List[str]
    ) -> None:
        self.root_project_name: str = root_project_name
        self.macros: Dict[str, Macro] = macros
        self.internal_package_names: List[str] = internal_package_names
        self.internal_packages: Dict[str, MacroNamespace] = {}
        self.packages: Dict[str, MacroNamespace] = {}
        self.root_package_macros: Dict[str, Macro] = {}
        self.add_macros()
        self._build_internal_packages_namespace()
        self._build_macros_by_name()

    def _build_internal_packages_namespace(self) -> None:
        self.internal_packages_namespace: MacroNamespace = {}
        for pkg in reversed(self.internal_package_names):
            if pkg in self.internal_packages:
                self.internal_packages_namespace.update(self.internal_packages[pkg])

    def _build_macros_by_name(self) -> None:
        macros_by_name: Dict[str, Macro] = {}
        for macro in self.internal_packages_namespace.values():
            macros_by_name[macro.name] = macro
        for fnamespace in self.packages.values():
            for macro in fnamespace.values():
                macros_by_name[macro.name] = macro
        for macro in self.root_package_macros.values():
            macros_by_name[macro.name] = macro
        self.macros_by_name: Dict[str, Macro] = macros_by_name

    def _add_macro_to(self, package_namespaces: Dict[str, MacroNamespace], macro: Macro) -> None:
        if macro.package_name in package_namespaces:
            namespace: MacroNamespace = package_namespaces[macro.package_name]
        else:
            namespace = {}
            package_namespaces[macro.package_name] = namespace
        if macro.name in namespace:
            raise DuplicateMacroNameError(macro, macro, macro.package_name)
        namespace[macro.name] = macro

    def add_macro(self, macro: Macro) -> None:
        macro_name: str = macro.name
        if macro.package_name in self.internal_package_names:
            self._add_macro_to(self.internal_packages, macro)
        else:
            self._add_macro_to(self.packages, macro)
            if macro.package_name == self.root_project_name:
                self.root_package_macros[macro_name] = macro

    def add_macros(self) -> None:
        for macro in self.macros.values():
            self.add_macro(macro)

    def get_macro(self, local_package: str, macro_name: str) -> Optional[Macro]:
        local_package_macros: MacroNamespace = {}
        if local_package in self.internal_package_names:
            local_package_macros = self.internal_packages.get(local_package, {})
        elif local_package in self.packages:
            local_package_macros = self.packages.get(local_package, {})
        if macro_name in local_package_macros:
            return local_package_macros[macro_name]
        return self.macros_by_name.get(macro_name)

    def get_macro_id(self, local_package: str, macro_name: str) -> Optional[str]:
        macro: Optional[Macro] = self.get_macro(local_package, macro_name)
        if macro is None:
            return None
        else:
            return macro.unique_id

class TestMacroNamespace:

    def __init__(
        self,
        macro_resolver: MacroResolver,
        ctx: Dict,
        node: Dict,
        thread_ctx: Dict,
        depends_on_macros: Optional[List[str]]
    ) -> None:
        self.macro_resolver: MacroResolver = macro_resolver
        self.ctx: Dict = ctx
        self.node: Dict = node
        self.thread_ctx: Dict = thread_ctx
        self.local_namespace: Dict[str, MacroGenerator] = {}
        self.project_namespace: Dict[str, Dict[str, MacroGenerator]] = {}
        if depends_on_macros:
            dep_macros: List[str] = []
            self.recursively_get_depends_on_macros(depends_on_macros, dep_macros)
            for macro_unique_id in dep_macros:
                if macro_unique_id in self.macro_resolver.macros:
                    _, project_name, macro_name = macro_unique_id.split('.')
                    macro: Macro = self.macro_resolver.macros[macro_unique_id]
                    macro_gen: MacroGenerator = MacroGenerator(macro, self.ctx, self.node, self.thread_ctx)
                    self.local_namespace[macro_name] = macro_gen
                    if project_name not in self.project_namespace:
                        self.project_namespace[project_name] = {}
                    self.project_namespace[project_name][macro_name] = macro_gen

    def recursively_get_depends_on_macros(
        self,
        depends_on_macros: List[str],
        dep_macros: List[str]
    ) -> None:
        for macro_unique_id in depends_on_macros:
            if macro_unique_id in dep_macros:
                continue
            dep_macros.append(macro_unique_id)
            if macro_unique_id in self.macro_resolver.macros:
                macro: Macro = self.macro_resolver.macros[macro_unique_id]
                if macro.depends_on.macros:
                    self.recursively_get_depends_on_macros(macro.depends_on.macros, dep_macros)

    def get_from_package(
        self,
        package_name: Optional[str],
        name: str
    ) -> Optional[MacroGenerator]:
        macro: Optional[Macro] = None
        if package_name is None:
            macro = self.macro_resolver.macros_by_name.get(name)
        elif package_name == GLOBAL_PROJECT_NAME:
            macro = self.macro_resolver.internal_packages_namespace.get(name)
        elif package_name in self.macro_resolver.packages:
            macro = self.macro_resolver.packages[package_name].get(name)
        else:
            raise PackageNotFoundForMacroError(package_name)
        if not macro:
            return None
        macro_func: MacroGenerator = MacroGenerator(macro, self.ctx, self.node, self.thread_ctx)
        return macro_func
