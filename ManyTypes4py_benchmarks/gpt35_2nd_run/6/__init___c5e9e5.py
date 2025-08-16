from typing import Dict, Any, List, Tuple

def empty_profile_renderer() -> dbt.config.renderer.ProfileRenderer:
    return dbt.config.renderer.ProfileRenderer({})

def empty_project_renderer() -> dbt.config.renderer.DbtProjectYamlRenderer:
    return dbt.config.renderer.DbtProjectYamlRenderer()

model_config: Dict[str, Dict[str, Any]] = {'my_package_name': {'enabled': True, 'adwords': {'adwords_ads': {'materialized': 'table', 'enabled': True, 'schema': 'analytics'}}, 'snowplow': {'snowplow_sessions': {'sort': 'timestamp', 'materialized': 'incremental', 'dist': 'user_id', 'unique_key': 'id'}, 'base': {'snowplow_events': {'sort': ['timestamp', 'userid'], 'materialized': 'table', 'sort_type': 'interleaved', 'dist': 'userid'}}}}
model_fqns: Tuple[Tuple[str, ...], ...] = frozenset((('my_package_name', 'snowplow', 'snowplow_sessions'), ('my_package_name', 'snowplow', 'base', 'snowplow_events'), ('my_package_name', 'adwords', 'adwords_ads')))

class Args:
    def __init__(self, profiles_dir: str = None, threads: int = None, profile: str = None, cli_vars: Dict[str, str] = None, version_check: bool = None, project_dir: str = None, target: str = None):
        self.profile = profile
        self.threads = threads
        self.target = target
        if profiles_dir is not None:
            self.profiles_dir = profiles_dir
            flags.PROFILES_DIR = profiles_dir
        if cli_vars is not None:
            self.vars = cli_vars
        if version_check is not None:
            self.version_check = version_check
        if project_dir is not None:
            self.project_dir = project_dir

def project_from_config_norender(cfg: Dict[str, Any], packages: Dict[str, Any] = None, project_root: str = '/invalid-root-path', verify_version: bool = False) -> Any:
    if packages is None:
        packages = {}
    partial = dbt.config.project.PartialProject.from_dicts(project_root, project_dict=cfg, packages_dict=packages, selectors_dict={}, verify_version=verify_version)
    partial.project_dict['project-root'] = project_root
    rendered = dbt.config.project.RenderComponents(project_dict=partial.project_dict, packages_dict=partial.packages_dict, selectors_dict=partial.selectors_dict)
    return partial.create_project(rendered)

def project_from_config_rendered(cfg: Dict[str, Any], packages: Dict[str, Any] = None, project_root: str = '/invalid-root-path', verify_version: bool = False, packages_specified_path: str = PACKAGES_FILE_NAME) -> Any:
    if packages is None:
        packages = {}
    partial = dbt.config.project.PartialProject.from_dicts(project_root, project_dict=cfg, packages_dict=packages, selectors_dict={}, verify_version=verify_version, packages_specified_path=packages_specified_path)
    return partial.render(empty_project_renderer())
