from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

@dataclass
class ObsoleteImportMatch:
    reason: str
    constant: re.Pattern

_OBSOLETE_IMPORT: Dict[str, List[ObsoleteImportMatch]]

@dataclass
class NamespaceAlias:
    alias: str
    names: set

_FORCE_NAMESPACE_IMPORT: Dict[str, NamespaceAlias]

class HassImportsFormatChecker(BaseChecker):
    name: str
    priority: int
    msgs: dict[str, Tuple[str, str, str]]
    options: Tuple

    def __init__(self, linter: PyLinter):
        self.current_package: Optional[str]

    def visit_module(self, node: nodes.Module):
        pass

    def visit_import(self, node: nodes.Import):
        pass

    def _visit_importfrom_relative(self, current_package: str, node: nodes.ImportFrom):
        pass

    def _check_for_constant_alias(self, node: nodes.ImportFrom, current_component: str, imported_component: str) -> bool:
        pass

    def _check_for_component_root_import(self, node: nodes.ImportFrom, current_component: str, imported_parts: List[str], imported_component: str) -> bool:
        pass

    def _check_for_relative_import(self, current_package: str, node: nodes.ImportFrom, current_component: str) -> bool:
        pass

    def visit_importfrom(self, node: nodes.ImportFrom):
        pass

def register(linter: PyLinter):
    pass
