"""Validate dependencies."""
from __future__ import annotations
import ast
from collections import deque
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Deque, Any
from homeassistant.const import Platform
from homeassistant.requirements import DISCOVERY_INTEGRATIONS
from . import ast_parse_module
from .model import Config, Integration

class ImportCollector(ast.NodeVisitor):
    """Collect all integrations referenced."""

    def __init__(self, integration: Integration) -> None:
        """Initialize the import collector."""
        self.integration: Integration = integration
        self.referenced: Dict[Path, Set[str]] = {}
        self._cur_fil_dir: Optional[Path] = None

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

    def visit_If(self, node: ast.If) -> None:
        """Visit If node."""
        if isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING':
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit ImportFrom node."""
        if node.module is None:
            return
        if node.module == 'homeassistant.components.http.auth' and len(node.names) == 1 and (node.names[0].name == 'async_sign_path'):
            return
        if node.module.startswith('homeassistant.components.'):
            self._add_reference(node.module.split('.')[2])
        elif node.module == 'homeassistant.components':
            for name_node in node.names:
                self._add_reference(name_node.name)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit Import node."""
        for name_node in node.names:
            if name_node.name.startswith('homeassistant.components.'):
                self._add_reference(name_node.name.split('.')[2])

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit Attribute node."""
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'components' and (isinstance(node.value.value, ast.Name) and node.value.value.id == 'hass' or (isinstance(node.value.value, ast.Attribute) and node.value.value.attr in ('hass', '_hass'))):
            self._add_reference(node.attr)
        else:
            self.generic_visit(node)

ALLOWED_USED_COMPONENTS: Set[str] = {*{platform.value for platform in Platform}, 'alert', 'automation', 'conversation', 'default_config', 'device_automation', 'frontend', 'group', 'homeassistant', 'input_boolean', 'input_button', 'input_datetime', 'input_number', 'input_select', 'input_text', 'media_source', 'onboarding', 'panel_custom', 'persistent_notification', 'person', 'script', 'shopping_list', 'sun', 'system_health', 'system_log', 'timer', 'webhook', 'websocket_api', 'zone', 'mjpeg', 'stream'}
IGNORE_VIOLATIONS: Set[Union[str, Tuple[str, str]]] = {('sql', 'recorder'), ('lutron_caseta', 'lutron'), ('ffmpeg_noise', 'ffmpeg_motion'), ('demo', 'manual'), ('http', 'network'), ('http', 'cloud'), ('zha', 'homeassistant_hardware'), ('zha', 'homeassistant_sky_connect'), ('zha', 'homeassistant_yellow'), ('homeassistant_sky_connect', 'zha'), ('homeassistant_hardware', 'zha'), ('websocket_api', 'lovelace'), ('websocket_api', 'shopping_list'), 'logbook', ('conversation', 'assist_pipeline')}

def calc_allowed_references(integration: Integration) -> Set[str]:
    """Return a set of allowed references."""
    manifest = integration.manifest
    allowed_references: Set[str] = ALLOWED_USED_COMPONENTS | set(manifest.get('dependencies', [])) | set(manifest.get('after_dependencies', []))
    if 'bluetooth_adapters' in allowed_references:
        allowed_references.add('bluetooth')
    for check_domain, to_check in DISCOVERY_INTEGRATIONS.items():
        if any((check in manifest for check in to_check)):
            allowed_references.add(check_domain)
    return allowed_references

def find_non_referenced_integrations(integrations: Dict[str, Integration], integration: Integration, references: Dict[Path, Set[str]]) -> Set[str]:
    """Find integrations that are not allowed to be referenced."""
    allowed_references = calc_allowed_references(integration)
    referenced: Set[str] = set()
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

def _compute_integration_dependencies(integration: Integration) -> Tuple[str, Optional[Dict[Path, Set[str]]]]:
    """Compute integration dependencies."""
    if integration.domain in IGNORE_VIOLATIONS:
        return (integration.domain, None)
    collector = ImportCollector(integration)
    collector.collect()
    return (integration.domain, collector.referenced)

def _validate_dependency_imports(integrations: Dict[str, Integration]) -> None:
    """Validate all dependencies."""
    with multiprocessing.Pool() as pool:
        integration_imports = dict(pool.imap_unordered(_compute_integration_dependencies, integrations.values(), chunksize=10))
    for integration in integrations.values():
        referenced = integration_imports[integration.domain]
        if not referenced:
            continue
        for domain in sorted(find_non_referenced_integrations(integrations, integration, referenced)):
            integration.add_error('dependencies', f"Using component {domain} but it's not in 'dependencies' or 'after_dependencies'")

def _check_circular_deps(integrations: Dict[str, Integration], start_domain: str, integration: Integration, checked: Set[str], checking: Deque[str]) -> None:
    """Check for circular dependencies pointing at starting_domain."""
    if integration.domain in checked or integration.domain in checking:
        return
    checking.append(integration.domain)
    for domain in integration.manifest.get('dependencies', []):
        if domain == start_domain:
            integrations[start_domain].add_error('dependencies', f'Found a circular dependency with {integration.domain} ({", ".join(checking)})')
            break
        _check_circular_deps(integrations, start_domain, integrations[domain], checked, checking)
    else:
        for domain in integration.manifest.get('after_dependencies', []):
            if domain == start_domain:
                integrations[start_domain].add_error('dependencies', f'Found a circular dependency with after dependencies of {integration.domain} ({", ".join(checking)})')
                break
            _check_circular_deps(integrations, start_domain, integrations[domain], checked, checking)
    checked.add(integration.domain)
    checking.remove(integration.domain)

def _validate_circular_dependencies(integrations: Dict[str, Integration]) -> None:
    for integration in integrations.values():
        if integration.domain in IGNORE_VIOLATIONS:
            continue
        _check_circular_deps(integrations, integration.domain, integration, set(), deque())

def _validate_dependencies(integrations: Dict[str, Integration]) -> None:
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

def validate(integrations: Dict[str, Integration], config: Config) -> None:
    """Handle dependencies for integrations."""
    _validate_dependency_imports(integrations)
    if not config.specific_integrations:
        _validate_dependencies(integrations)
        _validate_circular_dependencies(integrations)
