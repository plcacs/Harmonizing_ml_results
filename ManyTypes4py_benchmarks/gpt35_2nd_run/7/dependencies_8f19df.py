from __future__ import annotations
import ast
from collections import deque
import multiprocessing
from pathlib import Path
from homeassistant.const import Platform
from homeassistant.requirements import DISCOVERY_INTEGRATIONS
from . import ast_parse_module
from .model import Config, Integration
from typing import Set, Dict, Any, Tuple, List

class ImportCollector(ast.NodeVisitor):
    def __init__(self, integration: Integration) -> None:
        self.integration: Integration = integration
        self.referenced: Dict[Path, Set[str]] = {}
        self._cur_fil_dir: Path = None

    def collect(self) -> None:
        ...

    def _add_reference(self, reference_domain: str) -> None:
        ...

    def visit_If(self, node: ast.If) -> None:
        ...

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        ...

    def visit_Import(self, node: ast.Import) -> None:
        ...

    def visit_Attribute(self, node: ast.Attribute) -> None:
        ...

def calc_allowed_references(integration: Integration) -> Set[str]:
    ...

def find_non_referenced_integrations(integrations: Dict[str, Integration], integration: Integration, references: Dict[Path, Set[str]]) -> Set[str]:
    ...

def _compute_integration_dependencies(integration: Integration) -> Tuple[str, Dict[Path, Set[str]]]:
    ...

def _validate_dependency_imports(integrations: Dict[str, Integration]) -> None:
    ...

def _check_circular_deps(integrations: Dict[str, Integration], start_domain: str, integration: Integration, checked: Set[str], checking: deque) -> None:
    ...

def _validate_circular_dependencies(integrations: Dict[str, Integration]) -> None:
    ...

def _validate_dependencies(integrations: Dict[str, Integration]) -> None:
    ...

def validate(integrations: Dict[str, Integration], config: Config) -> None:
    ...
