from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger()


def sanitize_json(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _filter_recursive(data: Any, blacklist: List[Any]) -> Any:
    if isinstance(data, dict):
        return {k: _filter_recursive(v, blacklist) for k, v in data.items() if v not in blacklist}
    if isinstance(data, list):
        return [_filter_recursive(v, blacklist) for v in data]
    return data


def json_load(path: Union[str, Path]) -> Any:
    file_path = Path(path).resolve()
    if file_path.is_file():
        try:
            data = file_path.read_text()
            if data.strip():
                return json.loads(data, object_hook=sanitize_json)
        except Exception:
            backup_path = f"{file_path}.{datetime.now().isoformat()}.backup"
            logger.exception('Error opening JSON file "%s"', file_path)
            logger.warning('Moving invalid JSON file to "%s"', backup_path)
            shutil.move(str(file_path), backup_path)
    return {}


def json_stringify(
    data: Any, indent: Optional[Union[int, str]] = None, sort_keys: bool = True, value_blacklist: Optional[List[Any]] = None
) -> str:
    if value_blacklist is None:
        value_blacklist = [[], {}, None, ""]
    filtered_data = _filter_recursive(data, value_blacklist)
    return json.dumps(filtered_data, indent=indent, sort_keys=sort_keys)


def json_save(
    data: Any,
    path: Union[str, Path],
    indent: Optional[Union[int, str]] = 2,
    sort_keys: bool = True,
    value_blacklist: Optional[List[Any]] = None,
) -> bool:
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
