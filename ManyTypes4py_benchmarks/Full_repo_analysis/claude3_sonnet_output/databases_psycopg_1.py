"""
All the unique changes for the databases package
with the custom Numeric as the deprecated pypostgresql
for backwards compatibility and to make sure the
package can go to SQLAlchemy 2.0+.
"""
import typing
from sqlalchemy import types, util
from sqlalchemy.dialects.postgresql.base import PGDialect, PGExecutionContext
from sqlalchemy.engine import processors
from sqlalchemy.types import Float, Numeric

class PGExecutionContext_psycopg(PGExecutionContext):
    ...

class PGNumeric(Numeric):

    def bind_processor(self, dialect: PGDialect) -> typing.Callable[[typing.Any], str]:
        return processors.to_str

    def result_processor(self, dialect: PGDialect, coltype: typing.Any) -> typing.Optional[typing.Callable[[typing.Any], float]]:
        if self.asdecimal:
            return None
        else:
            return processors.to_float

class PGDialect_psycopg(PGDialect):
    colspecs: typing.Dict[typing.Type[typing.Any], typing.Type[typing.Any]] = util.update_copy(
        PGDialect.colspecs, {types.Numeric: PGNumeric, types.Float: Float}
    )
    execution_ctx_cls = PGExecutionContext_psycopg

dialect = PGDialect_psycopg
