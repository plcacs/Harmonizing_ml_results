from __future__ import annotations
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
logger = logging.getLogger()

def sanitize_json(d: Union[dict, dict[str, typing.Any]]) -> dict[tuple[typing.Union[str,typing.Any]], tuple[typing.Union[str,typing.Any]]]:
    return {k: v for k, v in d.items() if v is not None}

def _filter_recursive(data: Union[dict[str, typing.Any], typing.MutableMapping, dict[str, str]], blacklist: Union[typing.Mapping, dict, dict[str, typing.Any]]) -> Union[dict, list, dict[str, typing.Any], typing.MutableMapping, dict[str, str]]:
    if isinstance(data, dict):
        return {k: _filter_recursive(v, blacklist) for k, v in data.items() if v not in blacklist}
    if isinstance(data, list):
        return [_filter_recursive(v, blacklist) for v in data]
    return data

def json_load(path: Union[pathlib.Path, str]) -> dict:
    file_path = Path(path).resolve()
    if file_path.is_file():
        try:
            data = file_path.read_text()
            if data.strip():
                return json.loads(data, object_hook=sanitize_json)
        except Exception:
            backup_path = f'{file_path}.{datetime.now().isoformat()}.backup'
            logger.exception('Error opening JSON file "%s"', file_path)
            logger.warning('Moving invalid JSON file to "%s"', backup_path)
            shutil.move(str(file_path), backup_path)
    return {}

def json_stringify(data: Union[bytes, bool, None], indent: Union[None, bool, dict[str, bytes], tuple[typing.Union[int,str]]]=None, sort_keys: bool=True, value_blacklist: Union[None, str, list[str], typing.Any]=None):
    if value_blacklist is None:
        value_blacklist = [[], {}, None, '']
    filtered_data = _filter_recursive(data, value_blacklist)
    return json.dumps(filtered_data, indent=indent, sort_keys=sort_keys)

def json_save(data: Union[str, pathlib.Path, bool], path: Union[str, None, bool], indent: int=2, sort_keys: bool=True, value_blacklist: Union[None, str, pathlib.Path, bool]=None) -> bool:
    """Save self to file path"""
    file_path = Path(path).resolve()
    if file_path:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(json_stringify(data, indent=indent, sort_keys=sort_keys, value_blacklist=value_blacklist))
        except Exception:
            logger.exception('Could not write to JSON file "%s"', file_path)
        else:
            return True
    return False