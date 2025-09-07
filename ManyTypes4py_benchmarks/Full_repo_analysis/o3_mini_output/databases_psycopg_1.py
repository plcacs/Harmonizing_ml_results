import typing
from sqlalchemy import types, util
from sqlalchemy.dialects.postgresql.base import PGDialect, PGExecutionContext
from sqlalchemy.engine import processors
from sqlalchemy.types import Float, Numeric
from sqlalchemy.engine.interfaces import Dialect

class PGExecutionContext_psycopg(PGExecutionContext):
    ...  # type: ignore

class PGNumeric(Numeric):

    def bind_processor(self, dialect: Dialect) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
        return processors.to_str

    def result_processor(self, dialect: Dialect, coltype: typing.Any) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
        if self.asdecimal:
            return None
        else:
            return processors.to_float

class PGDialect_psycopg(PGDialect):
    colspecs: typing.Dict[type, type] = util.update_copy(
        PGDialect.colspecs,
        {types.Numeric: PGNumeric, types.Float: Float}
    )
    execution_ctx_cls = PGExecutionContext_psycopg

dialect: type = PGDialect_psycopg