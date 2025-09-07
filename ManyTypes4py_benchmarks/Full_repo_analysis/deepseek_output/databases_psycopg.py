from sqlalchemy.dialects.postgresql.psycopg import PGCompiler_psycopg
from sqlalchemy.sql.base import ColumnDefault
from sqlalchemy.dialects.postgresql.base import PGDialect
from typing import Any, Dict

class APGCompiler_psycopg2(PGCompiler_psycopg):

    def construct_params(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        pd = super().construct_params(*args, **kwargs)
        for column in self.prefetch:
            pd[column.key] = self._exec_default(column.default)
        return pd

    def _exec_default(self, default: ColumnDefault) -> Any:
        if default.is_callable:
            return default.arg(self.dialect)
        else:
            return default.arg
