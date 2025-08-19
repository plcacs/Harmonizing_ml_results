from __future__ import annotations
import json
import logging
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, TypeAlias

logger: logging.Logger = logging.getLogger()

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def sanitize_json(d: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return {k: v for k, v in d.items() if v is not None}


def _filter_recursive(data: JSONValue, blacklist: Collection[Any]) -> JSONValue:
    if isinstance(data, dict):
        return {k: _filter_recursive(v, blacklist) for k, v in data.items() if v not in blacklist}
    if isinstance(data, list):
        return [_filter_recursive(v, blacklist) for v in data]
    return data


def json_load(path: str | Path | os.PathLike[str]) -> JSONValue:
    file_path = Path(path).resolve()
    if file_path.is_file():
        try:
            data = file_path.read_text()
            if data.strip():
                result: JSONValue = json.loads(data, object_hook=sanitize_json)
                return result
        except Exception:
            backup_path = f'{file_path}.{datetime.now().isoformat()}.backup'
            logger.exception('Error opening JSON file "%s"', file_path)
            logger.warning('Moving invalid JSON file to "%s"', backup_path)
            shutil.move(str(file_path), backup_path)
    return {}


def json_stringify(
    data: JSONValue,
    indent: int | str | None = None,
    sort_keys: bool = True,
    value_blacklist: Collection[Any] | None = None,
) -> str:
    if value_blacklist is None:
        value_blacklist = [[], {}, None, '']
    filtered_data: JSONValue = _filter_recursive(data, value_blacklist)
    return json.dumps(filtered_data, indent=indent, sort_keys=sort_keys)


def json_save(
    data: JSONValue,
    path: str | Path | os.PathLike[str],
    indent: int | str | None = 2,
    sort_keys: bool = True,
    value_blacklist: Collection[Any] | None = None,
) -> bool:
    """Save self to file path"""
    file_path = Path(path).resolve()
    if file_path:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                json_stringify(data, indent=indent, sort_keys=sort_keys, value_blacklist=value_blacklist)
            )
        except Exception:
            logger.exception('Could not write to JSON file "%s"', file_path)
        else:
            return True
    return False