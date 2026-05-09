from __future__ import annotations
from dataclasses import dataclass
import re
from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter

@dataclass
class ObsoleteImportMatch:
    """Class for pattern matching."""
    reason: str
    constant: re.Pattern

_OBSOLETE_IMPORT: dict[str, list[ObsoleteImportMatch]] = {
    'functools': [ObsoleteImportMatch(reason='replaced by propcache.api.cached_property', constant=re.compile('^cached_property$'))],
    # ... rest of the dictionary
}

_IGNORE_ROOT_IMPORT: tuple[str, ...] = ('assist_pipeline', 'automation', 'bluetooth', 'camera', 'cast', 'device_automation', 'device_tracker', 'ffmpeg', 'ffmpeg_motion', 'google_assistant', 'hardware', 'homeassistant', 'homeassistant_hardware', 'http', 'manual', 'plex', 'recorder', 'rest', 'script', 'sensor', 'stream', 'zha')

@dataclass
class NamespaceAlias:
    """Class for namespace imports."""
    alias: str
    names: set[str]

_FORCE_NAMESPACE_IMPORT: dict[str, NamespaceAlias] = {
    'homeassistant.helpers.area_registry': NamespaceAlias('ar', {'async_get'}),
    # ... rest of the dictionary
}

class HassImportsFormatChecker(BaseChecker):
    """Checker for imports."""
    name: str = 'hass_imports'
    priority: int = -1
    msgs: dict[str, tuple[str, str, str]] = {
        'W7421': ('Relative import should be used', 'hass-relative-import', 'Used when absolute import should be replaced with relative import'),
        'W7422': ('%s is deprecated, %s', 'hass-deprecated-import', 'Used when import is deprecated'),
        'W7423': ('Absolute import should be used', 'hass-absolute-import', 'Used when relative import should be replaced with absolute import'),
        'W7424': ('Import should be using the component root', 'hass-component-root-import', 'Used when an import from another component should be from the component root'),
        'W7425': ('`%s` should not be imported directly. Please import `%s` as `%s` and use `%s.%s`', 'hass-helper-namespace-import', 'Used when a helper should be used via the namespace'),
        'W7426': ('`%s` should be imported using an alias, such as `%s as %s`', 'hass-import-constant-alias', 'Used when a constant should be imported as an alias')
    }
    options: tuple = ()

    def __init__(self, linter: PyLinter) -> None:
        """Initialize the HassImportsFormatChecker."""
        super().__init__(linter)
        self.current_package: str | None = None

    def visit_module(self, node: nodes.Module) -> None:
        """Determine current package."""
        if node.package:
            self.current_package = node.name
        else:
            self.current_package = node.name[:node.name.rfind('.')]

    def visit_import(self, node: nodes.Import) -> None:
        """Check for improper `import _` invocations."""
        if self.current_package is None:
            return
        for module, _alias in node.names:
            if module.startswith(f'{self.current_package}.'):
                self.add_message('hass-relative-import', node=node)
                continue
            if module.startswith('homeassistant.components.') and len(module.split('.')) > 3:
                if self.current_package.startswith('tests.components.') and self.current_package.split('.')[2] == module.split('.')[2]:
                    continue
                self.add_message('hass-component-root-import', node=node)

    def _visit_importfrom_relative(self, current_package: str, node: nodes.ImportFrom) -> None:
        """Check for improper 'from ._ import _' invocations."""
        if node.level <= 1 or (not current_package.startswith('homeassistant.components.') and (not current_package.startswith('tests.components.'))):
            return
        split_package = current_package.split('.')
        if not node.modname and len(split_package) == node.level + 1:
            for name in node.names:
                if name[0] != split_package[2]:
                    self.add_message('hass-absolute-import', node=node)
                    return
            return
        if len(split_package) < node.level + 2:
            self.add_message('hass-absolute-import', node=node)

    def _check_for_constant_alias(self, node: nodes.ImportFrom, current_component: str, imported_component: str) -> bool:
        """Check for hass-import-constant-alias."""
        if current_component == imported_component:
            return True
        for name, alias in node.names:
            if name == 'DOMAIN' and (alias is None or alias == 'DOMAIN'):
                self.add_message('hass-import-constant-alias', node=node, args=('DOMAIN', 'DOMAIN', f'{imported_component.upper()}_DOMAIN'))
                return False
        return True

    def _check_for_component_root_import(self, node: nodes.ImportFrom, current_component: str, imported_parts: list[str], imported_component: str) -> bool:
        """Check for hass-component-root-import."""
        if current_component == imported_component or imported_component in _IGNORE_ROOT_IMPORT:
            return True
        if len(imported_parts) > 3:
            self.add_message('hass-component-root-import', node=node)
            return False
        for name, _ in node.names:
            if name == 'const':
                self.add_message('hass-component-root-import', node=node)
                return False
        return True

    def _check_for_relative_import(self, current_package: str, node: nodes.ImportFrom, current_component: str) -> bool:
        """Check for hass-relative-import."""
        if node.modname == current_package or node.modname.startswith(f'{current_package}.'):
            self.add_message('hass-relative-import', node=node)
            return False
        for root in ('homeassistant', 'tests'):
            if current_package.startswith(f'{root}.components.'):
                if node.modname == f'{root}.components':
                    for name in node.names:
                        if name[0] == current_component:
                            self.add_message('hass-relative-import', node=node)
                            return False
                elif node.modname.startswith(f'{root}.components.{current_component}.'):
                    self.add_message('hass-relative-import', node=node)
                    return False
        return True

    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        """Check for improper 'from _ import _' invocations."""
        if not self.current_package:
            return
        if node.level is not None:
            self._visit_importfrom_relative(self.current_package, node)
            return
        current_component: str | None = None
        for root in ('homeassistant', 'tests'):
            if self.current_package.startswith(f'{root}.components.'):
                current_component = self.current_package.split('.')[2]
        if not self._check_for_relative_import(self.current_package, node, current_component):
            return
        if node.modname.startswith('homeassistant.components.'):
            imported_parts: list[str] = node.modname.split('.')
            imported_component: str = imported_parts[2]
            if not self._check_for_component_root_import(node, current_component, imported_parts, imported_component):
                return
            if not self._check_for_constant_alias(node, current_component, imported_component):
                return
        if (_obsolete_imports := _OBSOLETE_IMPORT.get(node.modname)):
            for name_tuple in node.names:
                for obsolete_import in _obsolete_imports:
                    if (import_match := obsolete_import.constant.match(name_tuple[0])):
                        self.add_message('hass-deprecated-import', node=node, args=(import_match.string, obsolete_import.reason))
        if (_namespace_alias := _FORCE_NAMESPACE_IMPORT.get(node.modname)):
            for name in node.names:
                if name[0] in _namespace_alias.names:
                    self.add_message('hass-helper-namespace-import', node=node, args=(name[0], node.modname, _namespace_alias.alias, _namespace_alias.alias, name[0]))

def register(linter: PyLinter) -> None:
    """Register the checker."""
    linter.register_checker(HassImportsFormatChecker(linter))
