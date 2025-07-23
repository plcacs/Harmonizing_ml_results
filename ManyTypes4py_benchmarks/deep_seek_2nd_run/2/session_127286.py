"""This module implements Kedro session responsible for project lifecycle."""
from __future__ import annotations
import getpass
import logging
import logging.config
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
import click
from kedro import __version__ as kedro_version
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.hooks.manager import _register_hooks, _register_hooks_entry_points
from kedro.framework.project import pipelines, settings, validate_settings
from kedro.io.core import generate_timestamp
from kedro.runner import AbstractRunner, SequentialRunner
from kedro.utils import _find_kedro_project

if TYPE_CHECKING:
    from collections.abc import Iterable
    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore
    from pluggy import PluginManager

def _describe_git(project_path: Path) -> Dict[str, Any]:
    path = str(project_path)
    try:
        res = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=path, stderr=subprocess.STDOUT)
        git_data = {'commit_sha': res.decode().strip()}
        git_status_res = subprocess.check_output(['git', 'status', '--short'], cwd=path, stderr=subprocess.STDOUT)
        git_data['dirty'] = bool(git_status_res.decode().strip())
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug('Unable to git describe %s', path)
        logger.debug(traceback.format_exc())
        return {}
    return {'git': git_data}

def _jsonify_cli_context(ctx: click.Context) -> Dict[str, Any]:
    return {
        'args': ctx.args,
        'params': ctx.params,
        'command_name': ctx.command.name,
        'command_path': ' '.join(['kedro'] + sys.argv[1:])
    }

class KedroSessionError(Exception):
    """``KedroSessionError`` raised by ``KedroSession``
    in the case that multiple runs are attempted in one session.
    """
    pass

class KedroSession:
    """``KedroSession`` is the object that is responsible for managing the lifecycle
    of a Kedro run."""

    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = None,
        project_path: Optional[Union[str, Path]] = None,
        save_on_close: bool = False,
        conf_source: Optional[str] = None
    ):
        self._project_path = Path(project_path or _find_kedro_project(Path.cwd()) or Path.cwd()).resolve()
        self.session_id = session_id
        self.save_on_close = save_on_close
        self._package_name = package_name
        self._store = self._init_store()
        self._run_called = False
        hook_manager = _create_hook_manager()
        _register_hooks(hook_manager, settings.HOOKS)
        _register_hooks_entry_points(hook_manager, settings.DISABLE_HOOKS_FOR_PLUGINS)
        self._hook_manager = hook_manager
        self._conf_source = conf_source or str(self._project_path / settings.CONF_SOURCE)

    @classmethod
    def create(
        cls,
        project_path: Optional[Union[str, Path]] = None,
        save_on_close: bool = True,
        env: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        conf_source: Optional[str] = None
    ) -> KedroSession:
        """Create a new instance of ``KedroSession``."""
        validate_settings()
        session = cls(
            project_path=project_path,
            session_id=generate_timestamp(),
            save_on_close=save_on_close,
            conf_source=conf_source
        )
        session_data: Dict[str, Any] = {
            'project_path': session._project_path,
            'session_id': session.session_id
        }
        ctx = click.get_current_context(silent=True)
        if ctx:
            session_data['cli'] = _jsonify_cli_context(ctx)
        env = env or os.getenv('KEDRO_ENV')
        if env:
            session_data['env'] = env
        if extra_params:
            session_data['extra_params'] = extra_params
        try:
            session_data['username'] = getpass.getuser()
        except Exception as exc:
            logging.getLogger(__name__).debug('Unable to get username. Full exception: %s', exc)
        session_data.update(**_describe_git(session._project_path))
        session._store.update(session_data)
        return session

    def _init_store(self) -> BaseSessionStore:
        store_class = settings.SESSION_STORE_CLASS
        classpath = f'{store_class.__module__}.{store_class.__qualname__}'
        store_args = deepcopy(settings.SESSION_STORE_ARGS)
        store_args.setdefault('path', (self._project_path / 'sessions').as_posix())
        store_args['session_id'] = self.session_id
        try:
            return store_class(**store_args)
        except TypeError as err:
            raise ValueError(f"\n{err}.\nStore config must only contain arguments valid for the constructor of '{classpath}'.") from err
        except Exception as err:
            raise ValueError(f"\n{err}.\nFailed to instantiate session store of type '{classpath}'.") from err

    def _log_exception(self, exc_type: Type[BaseException], exc_value: BaseException, exc_tb: traceback) -> None:
        type_ = [] if exc_type.__module__ == 'builtins' else [exc_type.__module__]
        type_.append(exc_type.__qualname__)
        exc_data = {
            'type': '.'.join(type_),
            'value': str(exc_value),
            'traceback': traceback.format_tb(exc_tb)
        }
        self._store['exception'] = exc_data

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @property
    def store(self) -> Dict[str, Any]:
        """Return a copy of internal store."""
        return dict(self._store)

    def load_context(self) -> KedroContext:
        """An instance of the project context."""
        env = self.store.get('env')
        extra_params = self.store.get('extra_params')
        config_loader = self._get_config_loader()
        context_class = settings.CONTEXT_CLASS
        context = context_class(
            package_name=self._package_name,
            project_path=self._project_path,
            config_loader=config_loader,
            env=env,
            extra_params=extra_params,
            hook_manager=self._hook_manager
        )
        self._hook_manager.hook.after_context_created(context=context)
        return context

    def _get_config_loader(self) -> AbstractConfigLoader:
        """An instance of the config loader."""
        env = self.store.get('env')
        extra_params = self.store.get('extra_params')
        config_loader_class = settings.CONFIG_LOADER_CLASS
        return config_loader_class(
            conf_source=self._conf_source,
            env=env,
            runtime_params=extra_params,
            **settings.CONFIG_LOADER_ARGS
        )

    def close(self) -> None:
        """Close the current session."""
        if self.save_on_close:
            self._store.save()

    def __enter__(self) -> KedroSession:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb_: Optional[traceback]
    ) -> None:
        if exc_type:
            self._log_exception(exc_type, exc_value, tb_)
        self.close()

    def run(
        self,
        pipeline_name: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
        runner: Optional[AbstractRunner] = None,
        node_names: Optional[List[str]] = None,
        from_nodes: Optional[List[str]] = None,
        to_nodes: Optional[List[str]] = None,
        from_inputs: Optional[List[str]] = None,
        to_outputs: Optional[List[str]] = None,
        load_versions: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the pipeline with a specified runner."""
        self._logger.info('Kedro project %s', self._project_path.name)
        if self._run_called:
            raise KedroSessionError(
                'A run has already been completed as part of the active KedroSession. '
                'KedroSession has a 1-1 mapping with runs, and thus only one run should '
                'be executed per session.'
            )
        session_id = self.store['session_id']
        save_version = session_id
        extra_params = self.store.get('extra_params') or {}
        context = self.load_context()
        name = pipeline_name or '__default__'
        try:
            pipeline = pipelines[name]
        except KeyError as exc:
            raise ValueError(
                f"Failed to find the pipeline named '{name}'. It needs to be generated "
                "and returned by the 'register_pipelines' function."
            ) from exc
        filtered_pipeline = pipeline.filter(
            tags=tags,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            node_names=node_names,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            node_namespace=namespace
        )
        record_data = {
            'session_id': session_id,
            'project_path': self._project_path.as_posix(),
            'env': context.env,
            'kedro_version': kedro_version,
            'tags': tags,
            'from_nodes': from_nodes,
            'to_nodes': to_nodes,
            'node_names': node_names,
            'from_inputs': from_inputs,
            'to_outputs': to_outputs,
            'load_versions': load_versions,
            'extra_params': extra_params,
            'pipeline_name': pipeline_name,
            'namespace': namespace,
            'runner': getattr(runner, '__name__', str(runner))
        }
        catalog = context._get_catalog(save_version=save_version, load_versions=load_versions)
        hook_manager = self._hook_manager
        runner = runner or SequentialRunner()
        if not isinstance(runner, AbstractRunner):
            raise KedroSessionError(
                'KedroSession expect an instance of Runner instead of a class.'
                'Have you forgotten the `()` at the end of the statement?'
            )
        hook_manager.hook.before_pipeline_run(
            run_params=record_data,
            pipeline=filtered_pipeline,
            catalog=catalog
        )
        try:
            run_result = runner.run(filtered_pipeline, catalog, hook_manager, session_id)
            self._run_called = True
        except Exception as error:
            hook_manager.hook.on_pipeline_error(
                error=error,
                run_params=record_data,
                pipeline=filtered_pipeline,
                catalog=catalog
            )
            raise
        hook_manager.hook.after_pipeline_run(
            run_params=record_data,
            run_result=run_result,
            pipeline=filtered_pipeline,
            catalog=catalog
        )
        return run_result
