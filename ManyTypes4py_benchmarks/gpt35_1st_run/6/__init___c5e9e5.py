from typing import Dict, Any, List, Tuple

def empty_profile_renderer() -> dbt.config.renderer.ProfileRenderer:
    return dbt.config.renderer.ProfileRenderer({})

def empty_project_renderer() -> dbt.config.renderer.DbtProjectYamlRenderer:
    return dbt.config.renderer.DbtProjectYamlRenderer()

def project_from_config_norender(cfg: Dict[str, Any], packages: Dict[str, Any] = None, project_root: str = '/invalid-root-path', verify_version: bool = False) -> dbt.config.project.Project:
    ...

def project_from_config_rendered(cfg: Dict[str, Any], packages: Dict[str, Any] = None, project_root: str = '/invalid-root-path', verify_version: bool = False, packages_specified_path: str = PACKAGES_FILE_NAME) -> dbt.config.project.Project:
    ...
