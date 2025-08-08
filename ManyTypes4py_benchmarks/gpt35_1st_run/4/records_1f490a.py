import enum
from datetime import date, datetime, time
from typing import Any, Dict, List, Tuple

from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.row import Row as SQLRow
from sqlalchemy.sql.compiler import _CompileLabel
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import JSON
from sqlalchemy.types import TypeEngine
from databases.interfaces import Record as RecordInterface

DIALECT_EXCLUDE: set = {'postgresql'}

class Record(RecordInterface):
    __slots__: Tuple[str] = ('_row', '_result_columns', '_dialect', '_column_map', '_column_map_int', '_column_map_full')

    def __init__(self, row: Any, result_columns: List[Any], dialect: Dialect, column_maps: Tuple[Dict[str, Tuple[int, TypeEngine], Dict[int, Tuple[int, TypeEngine], Dict[str, Tuple[int, TypeEngine]]]]):
        self._row = row
        self._result_columns = result_columns
        self._dialect = dialect
        self._column_map, self._column_map_int, self._column_map_full = column_maps

    @property
    def _mapping(self) -> Any:
        return self._row

    def keys(self) -> List[str]:
        return list(self._mapping.keys())

    def values(self) -> List[Any]:
        return list(self._mapping.values())

    def __getitem__(self, key: Any) -> Any:
        if len(self._column_map) == 0:
            return self._row[key]
        elif isinstance(key, Column):
            idx, datatype = self._column_map_full[str(key)]
        elif isinstance(key, int):
            idx, datatype = self._column_map_int[key]
        else:
            idx, datatype = self._column_map[key]
        raw = self._row[idx]
        processor = datatype._cached_result_processor(self._dialect, None)
        if self._dialect.name in DIALECT_EXCLUDE:
            if processor is not None and isinstance(raw, (int, str, float)):
                return processor(raw)
        return raw

    def __iter__(self) -> Any:
        return iter(self._row.keys())

    def __len__(self) -> int:
        return len(self._row)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__getitem__(name)
        except KeyError as e:
            raise AttributeError(e.args[0]) from e

class Row(SQLRow):

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return super().__getitem__(key)
        idx = self._key_to_index[key][0]
        return super().__getitem__(idx)

    def keys(self) -> List[str]:
        return list(self._mapping.keys())

    def values(self) -> List[Any]:
        return list(self._mapping.values())

def create_column_maps(result_columns: List[Tuple[str, Any, Any, TypeEngine]]) -> Tuple[Dict[str, Tuple[int, TypeEngine]], Dict[int, Tuple[int, TypeEngine]], Dict[str, Tuple[int, TypeEngine]]]:
    column_map, column_map_int, column_map_full = ({}, {}, {})
    for idx, (column_name, _, column, datatype) in enumerate(result_columns):
        column_map[column_name] = (idx, datatype)
        column_map_int[idx] = (idx, datatype)
        if isinstance(column[0], _CompileLabel):
            column_map_full[str(column[2])] = (idx, datatype)
        else:
            column_map_full[str(column[0])] = (idx, datatype)
    return (column_map, column_map_int, column_map_full)
