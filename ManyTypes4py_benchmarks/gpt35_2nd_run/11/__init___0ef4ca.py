from __future__ import annotations
import logging
import os
import sys
import typing
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collections import OrderedDict
    from IPython.core.interactiveshell import InteractiveShell

from IPython.core.getipython import get_ipython
from IPython.core.magic import needs_local_scope, register_line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

try:
    import rich.console as rich_console
    import rich.syntax as rich_syntax
except ImportError:
    pass

from kedro.framework.cli import load_entry_points
from kedro.framework.cli.project import CONF_SOURCE_HELP, PARAMS_ARG_HELP
from kedro.framework.cli.utils import ENV_HELP, _split_params
from kedro.framework.project import LOGGING, _ProjectPipelines, configure_project, pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.pipeline.node import Node
from kedro.utils import _find_kedro_project, _is_databricks

logger: logging.Logger = logging.getLogger(__name__)
FunctionParameters: MappingProxyType = MappingProxyType
RICH_INSTALLED: bool = True if importlib.util.find_spec('rich') is not None else False

def load_ipython_extension(ipython: InteractiveShell) -> None:
    ipython.register_magic_function(func=magic_reload_kedro, magic_name='reload_kedro')
    logger.info("Registered line magic '%reload_kedro'")
    ipython.register_magic_function(func=magic_load_node, magic_name='load_node')
    logger.info("Registered line magic '%load_node'")
    if _find_kedro_project(Path.cwd()) is None:
        logger.warning("Kedro extension was registered but couldn't find a Kedro project. Make sure you run '%reload_kedro <project_root>'.")
        return
    reload_kedro()

@typing.no_type_check
@needs_local_scope
@magic_arguments()
@argument('path', type=str, help='Path to the project root directory. If not given, use the previously setproject root.', nargs='?', default=None)
@argument('-e', '--env', type=str, default=None, help=ENV_HELP)
@argument('--params', type=lambda value: _split_params(None, None, value), default=None, help=PARAMS_ARG_HELP)
@argument('--conf-source', type=str, default=None, help=CONF_SOURCE_HELP)
def magic_reload_kedro(line: str, local_ns: dict = None, conf_source: Any = None) -> None:
    args = parse_argstring(magic_reload_kedro, line)
    reload_kedro(args.path, args.env, args.params, local_ns, args.conf_source)

def reload_kedro(path: str = None, env: str = None, extra_params: Any = None, local_namespace: dict = None, conf_source: Any = None) -> None:
    project_path = _resolve_project_path(path, local_namespace)
    metadata = bootstrap_project(project_path)
    _remove_cached_modules(metadata.package_name)
    configure_project(metadata.package_name)
    session = KedroSession.create(project_path, env=env, extra_params=extra_params, conf_source=conf_source)
    context = session.load_context()
    catalog = context.catalog
    get_ipython().push(variables={'context': context, 'catalog': catalog, 'session': session, 'pipelines': pipelines})
    logger.info('Kedro project %s', str(metadata.project_name))
    logger.info("Defined global variable 'context', 'session', 'catalog' and 'pipelines'")
    for line_magic in load_entry_points('line_magic'):
        register_line_magic(needs_local_scope(line_magic))
        logger.info("Registered line magic '%s'", line_magic.__name__)

def _resolve_project_path(path: str = None, local_namespace: dict = None) -> Path:
    if path:
        project_path = Path(path).expanduser().resolve()
    else:
        if local_namespace and local_namespace.get('context') and hasattr(local_namespace['context'], 'project_path'):
            project_path = local_namespace['context'].project_path
        else:
            project_path = _find_kedro_project(Path.cwd())
        if project_path:
            logger.info("Resolved project path as: %s.\nTo set a different path, run '%%reload_kedro <project_root>'", project_path)
    if project_path and local_namespace and local_namespace.get('context') and hasattr(local_namespace['context'], 'project_path') and (project_path != local_namespace['context'].project_path):
        logger.info('Updating path to Kedro project: %s...', project_path)
    return project_path

def _remove_cached_modules(package_name: str) -> None:
    to_remove = [mod for mod in sys.modules if mod.startswith(package_name)]
    for module in to_remove:
        del sys.modules[module]

def _guess_run_environment() -> str:
    if os.environ.get('VSCODE_PID') or os.environ.get('VSCODE_CWD'):
        return 'vscode'
    elif _is_databricks():
        return 'databricks'
    elif hasattr(get_ipython(), 'kernel'):
        return 'jupyter'
    else:
        return 'ipython'
