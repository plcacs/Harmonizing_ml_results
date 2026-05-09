from __future__ import annotations
import ast
from collections import deque
import multiprocessing
from pathlib import Path
from homeassistant.const import Platform
from homeassistant.requirements import DISCOVERY_INTEGRATIONS
from . import ast_parse_module
from .model import Config, Integration

class ImportCollector(ast.NodeVisitor):
    """Collect all integrations referenced."""

    def __init__(self, integration: Integration) -> None:
        """Initialize the import collector."""
        self.integration: Integration = integration
        self.referenced: dict[Path, set[str]] = {}
        self._cur_fil_dir: Path | None = None

    def collect(self) -> None:
        """Collect imports from a source file."""
        for fil in self.integration.path.glob('**/*.py'):
            if not fil.is_file():
                continue
            self._cur_fil_dir = fil.relative_to(self.integration.path)
            self.referenced[self._cur_fil_dir] = set()
            try:
                self.visit(ast_parse_module(fil))
            except SyntaxError as e:
                e.add_note(f'File: {fil}')
                raise
            self._cur_fil_dir = None

    def _add_reference(self, reference_domain: str) -> None:
        """Add a reference."""
        assert self._cur_fil_dir
        self.referenced[self._cur_fil_dir].add(reference_domain)

    # ... (rest of the class remains the same)

def calc_allowed_references(integration: Integration) -> set[str]:
    """Return a set of allowed references."""
    manifest = integration.manifest
    allowed_references = ALLOWED_USED_COMPONENTS | set(manifest.get('dependencies', [])) | set(manifest.get('after_dependencies', []))
    if 'bluetooth_adapters' in allowed_references:
        allowed_references.add('bluetooth')
    for check_domain, to_check in DISCOVERY_INTEGRATIONS.items():
        if any((check in manifest for check in to_check)):
            allowed_references.add(check_domain)
    return allowed_references

def find_non_referenced_integrations(integrations: dict[str, Integration], integration: Integration, references: dict[Path, set[str]]) -> set[str]:
    """Find integrations that are not allowed to be referenced."""
    allowed_references = calc_allowed_references(integration)
    referenced = set()
    for path, refs in references.items():
        if len(path.parts) == 1:
            cur_fil_dir = path.stem
        else:
            cur_fil_dir = path.parts[0]
        is_platform_other_integration = cur_fil_dir in integrations
        for ref in refs:
            if ref == integration.domain:
                continue
            if ref in allowed_references:
                continue
            if (integration.domain, ref) in IGNORE_VIOLATIONS:
                continue
            if is_platform_other_integration and cur_fil_dir == ref:
                continue
            if not is_platform_other_integration and ((integration.path / f'{ref}.py').is_file() or (integration.path / ref).is_dir()):
                continue
            referenced.add(ref)
    return referenced

def _compute_integration_dependencies(integration: Integration) -> tuple[str, dict[Path, set[str]]]:
    """Compute integration dependencies."""
    if integration.domain in IGNORE_VIOLATIONS:
        return (integration.domain, None)
    collector = ImportCollector(integration)
    collector.collect()
    return (integration.domain, collector.referenced)

def _validate_dependency_imports(integrations: dict[str, Integration]) -> None:
    """Validate all dependencies."""
    with multiprocessing.Pool() as pool:
        integration_imports = dict(pool.imap_unordered(_compute_integration_dependencies, integrations.values(), chunksize=10))
    for integration in integrations.values():
        referenced = integration_imports[integration.domain]
        if not referenced:
            continue
        for domain in sorted(find_non_referenced_integrations(integrations, integration, referenced)):
            integration.add_error('dependencies', f"Using component {domain} but it's not in 'dependencies' or 'after_dependencies'")

def _validate_circular_dependencies(integrations: dict[str, Integration]) -> None:
    """Check for circular dependencies."""
    for integration in integrations.values():
        if integration.domain in IGNORE_VIOLATIONS:
            continue
        _check_circular_deps(integrations, integration.domain, integration, set(), deque())

def _validate_dependencies(integrations: dict[str, Integration]) -> None:
    """Check that all referenced dependencies exist and are not duplicated."""
    for integration in integrations.values():
        if not integration.manifest:
            continue
        after_deps = integration.manifest.get('after_dependencies', [])
        for dep in integration.manifest.get('dependencies', []):
            if dep in after_deps:
                integration.add_error('dependencies', f'Dependency {dep} is both in dependencies and after_dependencies')
            if dep not in integrations:
                integration.add_error('dependencies', f'Dependency {dep} does not exist')

def validate(integrations: dict[str, Integration], config: Config) -> None:
    """Handle dependencies for integrations."""
    _validate_dependency_imports(integrations)
    if not config.specific_integrations:
        _validate_dependencies(integrations)
        _validate_circular_dependencies(integrations)
