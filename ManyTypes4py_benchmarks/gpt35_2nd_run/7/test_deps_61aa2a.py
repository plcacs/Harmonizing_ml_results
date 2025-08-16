from typing import List, Dict, Any

class MockRegistry:
    def __init__(self, packages: Dict[str, Dict[str, Dict[str, Any]]):
        self.packages = packages

    def index_cached(self, registry_base_url: str = None) -> List[str]:
        return sorted(self.packages)

    def package(self, package_name: str, registry_base_url: str = None) -> List[Dict[str, Any]]:
        try:
            pkg = self.packages[package_name]
        except KeyError:
            return []
        return pkg

    def get_compatible_versions(self, package_name: str, dbt_version: str, should_version_check: bool) -> List[str]:
        packages = self.package(package_name)
        return [pkg_version for pkg_version, info in packages.items() if is_compatible_version(info, dbt_version)]

    def package_version(self, name: str, version: str) -> Dict[str, Any]:
        try:
            return self.packages[name][version]
        except KeyError:
            return None

def resolve_packages(packages: List[Dict[str, str]], project: Any, extra_context: Dict[str, Any]) -> List[Any]:
    pass

def is_compatible_version(info: Dict[str, Any], dbt_version: str) -> bool:
    pass
