import enum
import typing
from datetime import date, datetime, time

from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.row import Row as SQLRow
from sqlalchemy.sql.compiler import _CompileLabel
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import JSON
from sqlalchemy.types import TypeEngine

from databases.interfaces import Record as RecordInterface

DIALECT_EXCLUDE: typing.Set[str] = {"postgresql"}


class Record(RecordInterface):
    __slots__ = (
        "_row",
        "_result_columns",
        "_dialect",
        "_column_map",
        "_column_map_int",
        "_column_map_full",
    )

    def __init__(
        self,
        row: typing.Mapping[str, typing.Any],
        result_columns: tuple,
        dialect: Dialect,
        column_maps: typing.Tuple[
            typing.Mapping[typing.Any, typing.Tuple[int, TypeEngine]],
            typing.Mapping[int, typing.Tuple[int, TypeEngine]],
            typing.Mapping[str, typing.Tuple[int, TypeEngine]],
        ],
    ) -> None:
        self._row: typing.Mapping[str, typing.Any] = row
        self._result_columns: tuple = result_columns
        self._dialect: Dialect = dialect
        self._column_map, self._column_map_int, self._column_map_full = column_maps

    @property
    def _mapping(self) -> typing.Mapping[str, typing.Any]:
        return self._row

    def keys(self) -> typing.KeysView[str]:
        return self._mapping.keys()

    def values(self) -> typing.ValuesView[typing.Any]:
        return self._mapping.values()

    def __getitem__(self, key: typing.Union[str, Column, int]) -> typing.Any:
        if len(self._column_map) == 0:
            return self._row[key]
        elif isinstance(key, Column):
            key_str: str = str(key)
            idx, datatype = self._column_map_full[key_str]
        elif isinstance(key, int):
            idx, datatype = self._column_map_int[key]
        else:
            idx, datatype = self._column_map[key]

        raw: typing.Any = self._row[idx]
        processor: typing.Optional[typing.Callable[[typing.Any], typing.Any]] = datatype._cached_result_processor(self._dialect, None)

        if self._dialect.name in DIALECT_EXCLUDE:
            if processor is not None and isinstance(raw, (int, str, float)):
                return processor(raw)

        return raw

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._row.keys())

    def __len__(self) -> int:
        return len(self._row)

    def __getattr__(self, name: str) -> typing.Any:
        try:
            return self.__getitem__(name)
        except KeyError as e:
            raise AttributeError(e.args[0]) from e


class Row(SQLRow):
    def __getitem__(self, key: typing.Union[str, int]) -> typing.Any:
        """
        An instance of a Row in SQLAlchemy allows the access
        to the Row._fields as tuple and the Row._mapping for
        the values.
        """
        if isinstance(key, int):
            return super().__getitem__(key)

        idx: int = self._key_to_index[key][0]
        return super().__getitem__(idx)

    def keys(self) -> typing.KeysView[str]:
        return self._mapping.keys()

    def values(self) -> typing.ValuesView[typing.Any]:
        return self._mapping.values()


def create_column_maps(
    result_columns: typing.Sequence[typing.Tuple[str, typing.Any, Column, TypeEngine]],
) -> typing.Tuple[
    typing.Mapping[typing.Any, typing.Tuple[int, TypeEngine]],
    typing.Mapping[int, typing.Tuple[int, TypeEngine]],
    typing.Mapping[str, typing.Tuple[int, TypeEngine]],
]:
    """
    Generate column -> datatype mappings from the column definitions.

    These mappings are used throughout PostgresConnection methods
    to initialize Record-s. The underlying DB driver does not do type
    conversion for us so we have wrap the returned asyncpg.Record-s.

    :return: Three mappings from different ways to address a column to \
                corresponding column indexes and datatypes: \
                1. by column identifier; \
                2. by column index; \
                3. by column name in Column sqlalchemy objects.
    """
    column_map: typing.Dict[typing.Any, typing.Tuple[int, TypeEngine]] = {}
    column_map_int: typing.Dict[int, typing.Tuple[int, TypeEngine]] = {}
    column_map_full: typing.Dict[str, typing.Tuple[int, TypeEngine]] = {}
    for idx, (column_name, _, column, datatype) in enumerate(result_columns):
        column_map[column_name] = (idx, datatype)
        column_map_int[idx] = (idx, datatype)

        # Added in SQLA 2.0 and _CompileLabels do not have _annotations
        # When this happens, the mapping is on the second position
        if isinstance(column[0], _CompileLabel):
            key_str: str = str(column[2])
        else:
            key_str = str(column[0])
        column_map_full[key_str] = (idx, datatype)
    return column_map, column_map_int, column_map_full
