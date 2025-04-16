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
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, Optional, Type

import click

from kedro import __version__ as kedro_version
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.hooks.manager import _register_hooks, _register_hooks_entry_points
from kedro.framework.project import (
    pipelines,
    settings,
    validate_settings,
)
from kedro.io.core import generate_timestamp
from kedro.runner import AbstractRunner, SequentialRunner
from kedro.utils import _find_kedro_project

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore


def _describe_git(project_path: Path) -> Dict[str, Dict[str, Any]]:
    path = str(project_path)
    try:
        res = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=path,
            stderr=subprocess.STDOUT,
        )
        git_data: Dict[str, Any] = {"commit_sha": res.decode().strip()}
        git_status_res = subprocess.check_output(  # noqa: S603
            ["git", "status", "--short"],  # noqa: S607
            cwd=path,
            stderr=subprocess.STDOUT,
        )
        git_data["dirty"] = bool(git_status_res.decode().strip())

    # `subprocess.check_output()` raises `NotADirectoryError` on Windows
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Unable to git describe %s", path)
        logger.debug(traceback.format_exc())
        return {}

    return {"git": git_data}


def _jsonify_cli_context(ctx: click.core.Context) -> Dict[str, Any]:
    return {
        "args": ctx.args,
        "params": ctx.params,
        "command_name": ctx.command.name,
        "command_path": " ".join(["kedro"] + sys.argv[1:]),
    }


class KedroSessionError(Exception):
    """``KedroSessionError`` raised by ``KedroSession``
    in the case that multiple runs are attempted in one session.
    """

    pass


class KedroSession:
    """``KedroSession`` is the object that is responsible for managing the lifecycle
    of a Kedro run. Use `KedroSession.create()` as
    a context manager to construct a new KedroSession with session data
    provided (see the example below).

    Example:
    ::

        >>> from kedro.framework.session import KedroSession
        >>> from kedro.framework.startup import bootstrap_project
        >>> from pathlib import Path

        >>> # If you are creating a session outside of a Kedro project (i.e. not using
        >>> # `kedro run` or `kedro jupyter`), you need to run `bootstrap_project` to
        >>> # let Kedro find your configuration.
        >>> bootstrap_project(Path("<project_root>"))
        >>> with KedroSession.create() as session:
        >>>     session.run()
    """

    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = None,
        project_path: Optional[Path | str] = None,
        save_on_close: bool = False,
        conf_source: Optional[str] = None,
    ) -> None:
        self._project_path: Path = Path(
            project_path or _find_kedro_project(Path.cwd()) or Path.cwd()
        ).resolve()
        self.session_id: str = session_id
        self.save_on_close: bool = save_on_close
        self._package_name: Optional[str] = package_name
        self._store: BaseSessionStore = self._init_store()
        self._run_called: bool = False

        hook_manager = _create_hook_manager()
        _register_hooks(hook_manager, settings.HOOKS)
        _register_hooks_entry_points(hook_manager, settings.DISABLE_HOOKS_FOR_PLUGINS)
        self._hook_manager: _create_hook_manager().__class__ = hook_manager

        self._conf_source: str = conf_source or str(
            self._project_path / settings.CONF_SOURCE
        )

    @classmethod
    def create(
        cls,
        project_path: Optional[Path | str] = None,
        save_on_close: bool = True,
        env: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        conf_source: Optional[str] = None,
    ) -> KedroSession:
        """Create a new instance of ``KedroSession`` with the session data.

        Args:
            project_path: Path to the project root directory. Default is
                current working directory Path.cwd().
            save_on_close: Whether or not to save the session when it's closed.
            conf_source: Path to a directory containing configuration
            env: Environment for the KedroContext.
            extra_params: Optional dictionary containing extra project parameters
                for underlying KedroContext. If specified, will update (and therefore
                take precedence over) the parameters retrieved from the project
                configuration.

        Returns:
            A new ``KedroSession`` instance.
        """
        validate_settings()

        session = cls(
            project_path=project_path,
            session_id=generate_timestamp(),
            save_on_close=save_on_close,
            conf_source=conf_source,
        )

        # have to explicitly type session_data otherwise mypy will complain
        # possibly related to this: https://github.com/python/mypy/issues/1430
        session_data: Dict[str, Any] = {
            "project_path": session._project_path,
            "session_id": session.session_id,
        }

        ctx: Optional[click.core.Context] = click.get_current_context(silent=True)
        if ctx:
            session_data["cli"] = _jsonify_cli_context(ctx)

        env = env or os.getenv("KEDRO_ENV")
        if env:
            session_data["env"] = env

        if extra_params:
            session_data["extra_params"] = extra_params

        try:
            session_data["username"] = getpass.getuser()
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Unable to get username. Full exception: %s", exc
            )

        session_data.update(**_describe_git(session._project_path))
        session._store.update(session_data)

        return session

    def _init_store(self) -> BaseSessionStore:
        store_class: Type[BaseSessionStore] = settings.SESSION_STORE_CLASS
        classpath: str = f"{store_class.__module__}.{store_class.__qualname__}"
        store_args: Dict[str, Any] = deepcopy(settings.SESSION_STORE_ARGS)
        store_args.setdefault("path", (self._project_path / "sessions").as_posix())
        store_args["session_id"] = self.session_id

        try:
            return store_class(**store_args)  # type: ignore[no-any-return]
        except TypeError as err:
            raise ValueError(
                f"\n{err}.\nStore config must only contain arguments valid "
                f"for the constructor of '{classpath}'."
            ) from err
        except Exception as err:
            raise ValueError(
                f"\n{err}.\nFailed to instantiate session store of type '{classpath}'."
            ) from err

    def _log_exception(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is None:
            return
        type_parts: list[str] = [] if exc_type.__module__ == "builtins" else [exc_type.__module__]
        type_parts.append(exc_type.__qualname__)

        exc_data: Dict[str, Any] = {
            "type": ".".join(type_parts),
            "value": str(exc_value),
            "traceback": traceback.format_tb(exc_tb),
        }
        self._store["exception"] = exc_data

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @property
    def store(self) -> Dict[str, Any]:
        """Return a copy of internal store."""
        return dict(self._store)

    def load_context(self) -> KedroContext:
        """An instance of the project context."""
        env: Optional[str] = self.store.get("env")
        extra_params: Optional[Dict[str, Any]] = self.store.get("extra_params")
        config_loader: AbstractConfigLoader = self._get_config_loader()
        context_class = settings.CONTEXT_CLASS
        context: KedroContext = context_class(
            package_name=self._package_name,
            project_path=self._project_path,
            config_loader=config_loader,
            env=env,
            extra_params=extra_params,
            hook_manager=self._hook_manager,
        )
        self._hook_manager.hook.after_context_created(context=context)

        return context  # type: ignore[return-value]

    def _get_config_loader(self) -> AbstractConfigLoader:
        """An instance of the config loader."""
        env: Optional[str] = self.store.get("env")
        extra_params: Optional[Dict[str, Any]] = self.store.get("extra_params")

        config_loader_class: Type[AbstractConfigLoader] = settings.CONFIG_LOADER_CLASS
        return config_loader_class(  # type: ignore[no-any-return]
            conf_source=self._conf_source,
            env=env,
            runtime_params=extra_params,
            **settings.CONFIG_LOADER_ARGS,
        )

    def close(self) -> None:
        """Close the current session and save its store to disk
        if `save_on_close` attribute is True.
        """
        if self.save_on_close:
            self._store.save()

    def __enter__(self) -> KedroSession:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb_: Optional[TracebackType],
    ) -> None:
        if exc_type:
            self._log_exception(exc_type, exc_value, tb_)
        self.close()

    def run(  # noqa: PLR0913
        self,
        pipeline_name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        runner: Optional[AbstractRunner] = None,
        node_names: Optional[Iterable[str]] = None,
        from_nodes: Optional[Iterable[str]] = None,
        to_nodes: Optional[Iterable[str]] = None,
        from_inputs: Optional[Iterable[str]] = None,
        to_outputs: Optional[Iterable[str]] = None,
        load_versions: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the pipeline with a specified runner.

        Args:
            pipeline_name: Name of the pipeline that is being run.
            tags: An optional list of node tags which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                containing *any* of these tags will be run.
            runner: An optional parameter specifying the runner that you want to run
                the pipeline with.
            node_names: An optional list of node names which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                with these names will be run.
            from_nodes: An optional list of node names which should be used as a
                starting point of the new ``Pipeline``.
            to_nodes: An optional list of node names which should be used as an
                end point of the new ``Pipeline``.
            from_inputs: An optional list of input datasets which should be
                used as a starting point of the new ``Pipeline``.
            to_outputs: An optional list of output datasets which should be
                used as an end point of the new ``Pipeline``.
            load_versions: An optional flag to specify a particular dataset
                version timestamp to load.
            namespace: The namespace of the nodes that is being run.
        Raises:
            ValueError: If the named or `__default__` pipeline is not
                defined by `register_pipelines`.
            Exception: Any uncaught exception during the run will be re-raised
                after being passed to ``on_pipeline_error`` hook.
            KedroSessionError: If more than one run is attempted to be executed during
                a single session.
        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.
        """
        # Report project name
        self._logger.info("Kedro project %s", self._project_path.name)

        if self._run_called:
            raise KedroSessionError(
                "A run has already been completed as part of the"
                " active KedroSession. KedroSession has a 1-1 mapping with"
                " runs, and thus only one run should be executed per session."
            )

        session_id: str = self.store["session_id"]
        save_version: str = session_id
        extra_params: Dict[str, Any] = self.store.get("extra_params") or {}
        context: KedroContext = self.load_context()

        name: str = pipeline_name or "__default__"

        try:
            pipeline = pipelines[name]
        except KeyError as exc:
            raise ValueError(
                f"Failed to find the pipeline named '{name}'. "
                f"It needs to be generated and returned "
                f"by the 'register_pipelines' function."
            ) from exc

        filtered_pipeline = pipeline.filter(
            tags=tags,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            node_names=node_names,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            node_namespace=namespace,
        )

        record_data: Dict[str, Any] = {
            "session_id": session_id,
            "project_path": self._project_path.as_posix(),
            "env": context.env,
            "kedro_version": kedro_version,
            "tags": tags,
            "from_nodes": from_nodes,
            "to_nodes": to_nodes,
            "node_names": node_names,
            "from_inputs": from_inputs,
            "to_outputs": to_outputs,
            "load_versions": load_versions,
            "extra_params": extra_params,
            "pipeline_name": pipeline_name,
            "namespace": namespace,
            "runner": getattr(runner, "__name__", str(runner)),
        }

        catalog = context._get_catalog(
            save_version=save_version,
            load_versions=load_versions,
        )

        # Run the runner
        hook_manager = self._hook_manager
        runner_instance: AbstractRunner = runner or SequentialRunner()
        if not isinstance(runner_instance, AbstractRunner):
            raise KedroSessionError(
                "KedroSession expect an instance of Runner instead of a class."
                "Have you forgotten the `()` at the end of the statement?"
            )
        hook_manager.hook.before_pipeline_run(
            run_params=record_data, pipeline=filtered_pipeline, catalog=catalog
        )
        try:
            run_result: Dict[str, Any] = runner_instance.run(
                filtered_pipeline, catalog, hook_manager, session_id
            )
            self._run_called = True
        except Exception as error:
            hook_manager.hook.on_pipeline_error(
                error=error,
                run_params=record_data,
                pipeline=filtered_pipeline,
                catalog=catalog,
            )
            raise

        hook_manager.hook.after_pipeline_run(
            run_params=record_data,
            run_result=run_result,
            pipeline=filtered_pipeline,
            catalog=catalog,
        )
        return run_result
