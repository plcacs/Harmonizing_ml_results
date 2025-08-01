#!/usr/bin/env python3
"""
This module load custom objects
"""
import importlib.util
import inspect
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


class PathModifier:
    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "PathModifier":
        """Inject path to allow importing with relative imports."""
        sys.path.insert(0, str(self.path))
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Any) -> None:
        """Undo insertion of local path."""
        str_path = str(self.path)
        if str_path in sys.path:
            sys.path.remove(str_path)


class IResolver:
    """
    This class contains all the logic to load custom classes
    """
    # Class variables - expected to be initialized by subclasses, if needed.
    user_subdir: Optional[str] = None
    initial_search_path: Optional[Path] = None
    extra_path: Optional[str] = None
    object_type: Any  # Subclasses should set this to a valid type.
    object_type_str: str  # Subclasses should set this to a valid string.

    @classmethod
    def build_search_paths(
        cls,
        config: Dict[str, Any],
        user_subdir: Optional[str] = None,
        extra_dirs: Optional[List[str]] = None
    ) -> List[Path]:
        abs_paths: List[Path] = []
        if cls.initial_search_path:
            abs_paths.append(cls.initial_search_path)
        if user_subdir:
            abs_paths.insert(0, config['user_data_dir'].joinpath(user_subdir))
        if extra_dirs:
            for directory in extra_dirs:
                abs_paths.insert(0, Path(directory).resolve())
        if cls.extra_path and (extra := config.get(cls.extra_path)):
            abs_paths.insert(0, Path(extra).resolve())
        return abs_paths

    @classmethod
    def _get_valid_object(
        cls,
        module_path: Path,
        object_name: Optional[str],
        enum_failed: bool = False
    ) -> Iterator[Optional[Tuple[type, str]]]:
        """
        Generator returning objects with matching object_type and object_name in the path given.
        Tuple format: (object class, source)
        """
        with PathModifier(module_path.parent):
            module_name: str = module_path.stem or ''
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if not spec:
                return iter([None])
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore
            except (AttributeError, ModuleNotFoundError, SyntaxError, ImportError, NameError) as err:
                logger.warning(f"Could not import {module_path} due to '{err}'")
                if enum_failed:
                    return iter([None])
            def is_valid_class(obj: Any) -> bool:
                try:
                    return (
                        inspect.isclass(obj) and 
                        issubclass(obj, cls.object_type) and 
                        (obj is not cls.object_type) and 
                        (obj.__module__ == module_name)
                    )
                except TypeError:
                    return False
            valid_objects_gen = (
                (obj, inspect.getsource(module))
                for name, obj in inspect.getmembers(module, is_valid_class)
                if object_name is None or object_name == name
            )
            return valid_objects_gen

    @classmethod
    def _search_object(
        cls,
        directory: Path,
        *,
        object_name: str,
        add_source: bool = False
    ) -> Tuple[Optional[type], Optional[Path]]:
        """
        Search for the objectname in the given directory
        :return: Tuple containing object class and module path
        """
        logger.debug(f"Searching for {cls.object_type.__name__} {object_name} in '{directory}'")
        for entry in directory.iterdir():
            if entry.suffix != '.py':
                logger.debug('Ignoring %s', entry)
                continue
            if entry.is_symlink() and (not entry.is_file()):
                logger.debug('Ignoring broken symlink %s', entry)
                continue
            module_path: Path = entry.resolve()
            obj = next(cls._get_valid_object(module_path, object_name), None)
            if obj:
                obj[0].__file__ = str(entry)
                if add_source:
                    obj[0].__source__ = obj[1]
                return (obj[0], module_path)
        return (None, None)

    @classmethod
    def _load_object(
        cls,
        paths: List[Path],
        *,
        object_name: str,
        add_source: bool = False,
        kwargs: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Try to load object from path list.
        """
        for _path in paths:
            try:
                module, module_path = cls._search_object(directory=_path, object_name=object_name, add_source=add_source)
                if module:
                    logger.info(f"Using resolved {cls.object_type.__name__.lower()[1:]} {object_name} from '{module_path}'...")
                    return module(**kwargs)
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist.', _path.resolve())
        return None

    @classmethod
    def load_object(
        cls,
        object_name: str,
        config: Dict[str, Any],
        *,
        kwargs: Dict[str, Any],
        extra_dir: Optional[str] = None
    ) -> Any:
        """
        Search and loads the specified object as configured in the child class.
        :raises: OperationalException if the class is invalid or does not exist.
        :return: Object instance
        """
        extra_dirs: List[str] = []
        if extra_dir:
            extra_dirs.append(extra_dir)
        abs_paths: List[Path] = cls.build_search_paths(config, user_subdir=cls.user_subdir, extra_dirs=extra_dirs)
        found_object: Optional[Any] = cls._load_object(paths=abs_paths, object_name=object_name, kwargs=kwargs)
        if found_object:
            return found_object
        raise OperationalException(f"Impossible to load {cls.object_type_str} '{object_name}'. This class does not exist or contains Python code errors.")

    @classmethod
    def search_all_objects(
        cls,
        config: Dict[str, Any],
        enum_failed: bool,
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Searches for valid objects.
        :return: List of dicts containing 'name', 'class', 'location', and 'location_rel' entries.
        """
        result: List[Dict[str, Any]] = []
        abs_paths: List[Path] = cls.build_search_paths(config, user_subdir=cls.user_subdir)
        for path in abs_paths:
            result.extend(cls._search_all_objects(path, enum_failed, recursive))
        return result

    @classmethod
    def _build_rel_location(cls, directory: Path, entry: Path) -> str:
        builtin: bool = cls.initial_search_path == directory
        return f'<builtin>/{entry.relative_to(directory)}' if builtin else str(entry.relative_to(directory))

    @classmethod
    def _search_all_objects(
        cls,
        directory: Path,
        enum_failed: bool,
        recursive: bool = False,
        basedir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches a directory for valid objects.
        :return: List of dicts containing 'name', 'class', 'location', and 'location_rel' entries.
        """
        logger.debug(f"Searching for {cls.object_type.__name__} '{directory}'")
        objects: List[Dict[str, Any]] = []
        if not directory.is_dir():
            logger.info(f"'{directory}' is not a directory, skipping.")
            return objects
        for entry in directory.iterdir():
            if recursive and entry.is_dir() and (not entry.name.startswith('__')) and (not entry.name.startswith('.')):
                objects.extend(cls._search_all_objects(entry, enum_failed, recursive, basedir or directory))
            if entry.suffix != '.py':
                logger.debug('Ignoring %s', entry)
                continue
            module_path: Path = entry.resolve()
            logger.debug(f'Path {module_path}')
            for obj in cls._get_valid_object(module_path, object_name=None, enum_failed=enum_failed):
                objects.append({
                    'name': obj[0].__name__ if obj is not None else '',
                    'class': obj[0] if obj is not None else None,
                    'location': entry,
                    'location_rel': cls._build_rel_location(basedir or directory, entry)
                })
        return objects
