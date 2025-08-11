from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Set, Union
from dbt.clients.jinja import MacroGenerator, MacroStack
from dbt.contracts.graph.nodes import Macro
from dbt.exceptions import DuplicateMacroNameError, PackageNotFoundForMacroError
from dbt.include.global_project import PROJECT_NAME as GLOBAL_PROJECT_NAME
FlatNamespace = Dict[str, MacroGenerator]
NamespaceMember = Union[FlatNamespace, MacroGenerator]
FullNamespace = Dict[str, NamespaceMember]

class MacroNamespace(Mapping):

    def __init__(self, global_namespace, local_namespace, global_project_namespace, packages) -> None:
        self.global_namespace = global_namespace
        self.local_namespace = local_namespace
        self.packages = packages
        self.global_project_namespace = global_project_namespace

    def _search_order(self) -> Union[typing.Generator, typing.Generator[dict]]:
        yield self.local_namespace
        yield self.global_namespace
        yield self.packages
        yield {GLOBAL_PROJECT_NAME: self.global_project_namespace}
        yield self.global_project_namespace

    def _keys(self) -> set:
        keys = set()
        for search in self._search_order():
            keys.update(search)
        return keys

    def __iter__(self) -> typing.Generator:
        for key in self._keys():
            yield key

    def __len__(self) -> int:
        return len(self._keys())

    def __getitem__(self, key: Union[str, None]):
        for dct in self._search_order():
            if key in dct:
                return dct[key]
        raise KeyError(key)

    def get_from_package(self, package_name: Union[str, None, int], name: str) -> Union[str, bool]:
        if package_name is None:
            return self.get(name)
        elif package_name == GLOBAL_PROJECT_NAME:
            return self.global_project_namespace.get(name)
        elif package_name in self.packages:
            return self.packages[package_name].get(name)
        else:
            raise PackageNotFoundForMacroError(package_name)

class MacroNamespaceBuilder:

    def __init__(self, root_package: Union[tuple[int], list[str], str], search_package: Union[dict[str, set[str]], depender.graph.dependency.DependencyGraph], thread_ctx: Union[typing.TextIO, tuple[typing.Union[str,bool]], int], internal_packages: Union[dict[str, set[str]], str], node: Union[None, bool, str]=None) -> None:
        self.root_package = root_package
        self.search_package = search_package
        self.internal_package_names = set(internal_packages)
        self.internal_package_names_order = internal_packages
        self.globals = {}
        self.locals = {}
        self.internal_packages = {}
        self.packages = {}
        self.thread_ctx = thread_ctx
        self.node = node

    def _add_macro_to(self, hierarchy: Union[str, dict[str, typing.Any], dict], macro: Union[str, T, typing.Mapping], macro_func: Union[str, dict, None]) -> None:
        if macro.package_name in hierarchy:
            namespace = hierarchy[macro.package_name]
        else:
            namespace = {}
            hierarchy[macro.package_name] = namespace
        if macro.name in namespace:
            raise DuplicateMacroNameError(macro_func.macro, macro, macro.package_name)
        hierarchy[macro.package_name][macro.name] = macro_func

    def add_macro(self, macro: mypy.nodes.Context, ctx: Any) -> None:
        macro_name = macro.name
        macro_func = MacroGenerator(macro, ctx, self.node, self.thread_ctx)
        if macro.package_name in self.internal_package_names:
            self._add_macro_to(self.internal_packages, macro, macro_func)
        else:
            self._add_macro_to(self.packages, macro, macro_func)
            if macro.package_name == self.search_package:
                self.locals[macro_name] = macro_func
            elif macro.package_name == self.root_package:
                self.globals[macro_name] = macro_func

    def add_macros(self, macros: Union[list, str], ctx: basilisp.lang.compiler.nodes.Node) -> None:
        for macro in macros:
            self.add_macro(macro, ctx)

    def build_namespace(self, macros_by_package: Any, ctx: Union[str, dict[str, typing.Any], typing.Type]) -> MacroNamespace:
        for package in macros_by_package.values():
            self.add_macros(package.values(), ctx)
        global_project_namespace = {}
        for pkg in reversed(self.internal_package_names_order):
            if pkg in self.internal_packages:
                global_project_namespace.update(self.internal_packages[pkg])
        return MacroNamespace(global_namespace=self.globals, local_namespace=self.locals, global_project_namespace=global_project_namespace, packages=self.packages)