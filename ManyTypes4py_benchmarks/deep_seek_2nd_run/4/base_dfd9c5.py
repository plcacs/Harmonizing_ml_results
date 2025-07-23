import threading
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set, DefaultDict
import psycopg2
from flask import current_app
from psycopg2.extensions import AsIs, adapt, register_adapter
from psycopg2.extras import Json, NamedTupleCursor, register_composite
from alerta.app import alarm_model
from alerta.database.base import Database
from alerta.exceptions import NoCustomerMatch
from alerta.models.enums import ADMIN_SCOPES
from alerta.models.heartbeat import HeartbeatStatus
from alerta.utils.format import DateTime
from alerta.utils.response import absolute_url
from .utils import Query

MAX_RETRIES = 5

class HistoryAdapter:

    def __init__(self, history: Any) -> None:
        self.history = history
        self.conn: Optional[psycopg2.extensions.connection] = None

    def prepare(self, conn: psycopg2.extensions.connection) -> None:
        self.conn = conn

    def getquoted(self) -> str:
        def quoted(o: Any) -> str:
            a = adapt(o)
            if hasattr(a, 'prepare'):
                a.prepare(self.conn)
            return a.getquoted().decode('utf-8')
        return '({}, {}, {}, {}, {}, {}, {}, {}::timestamp, {}, {})::history'.format(
            quoted(self.history.id), quoted(self.history.event), 
            quoted(self.history.severity), quoted(self.history.status),
            quoted(self.history.value), quoted(self.history.text),
            quoted(self.history.change_type), quoted(self.history.update_time),
            quoted(self.history.user), quoted(self.history.timeout)
        )

    def __str__(self) -> str:
        return str(self.getquoted())

Record = namedtuple('Record', [
    'id', 'resource', 'event', 'environment', 'severity', 'status', 'service', 
    'group', 'value', 'text', 'tags', 'attributes', 'origin', 'update_time', 
    'user', 'timeout', 'type', 'customer'
])

class Backend(Database):

    def create_engine(self, app: Any, uri: str, dbname: Optional[str] = None, 
                     schema: str = 'public', raise_on_error: bool = True) -> None:
        self.uri = uri
        self.dbname = dbname
        self.schema = schema
        lock = threading.Lock()
        with lock:
            conn = self.connect()
            with app.open_resource('sql/schema.sql') as f:
                try:
                    conn.cursor().execute(f.read())
                    conn.commit()
                except Exception as e:
                    if raise_on_error:
                        raise
                    app.logger.warning(e)
        register_adapter(dict, Json)
        register_adapter(datetime, self._adapt_datetime)
        register_composite(schema + '.history' if schema else 'history', conn, globally=True)
        from alerta.models.alert import History
        register_adapter(History, HistoryAdapter)

    def connect(self) -> psycopg2.extensions.connection:
        retry = 0
        while True:
            try:
                conn = psycopg2.connect(
                    dsn=self.uri, 
                    dbname=self.dbname, 
                    cursor_factory=NamedTupleCursor
                )
                conn.set_client_encoding('UTF8')
                break
            except Exception as e:
                print(e)
                retry += 1
                if retry > MAX_RETRIES:
                    conn = None
                    break
                else:
                    backoff = 2 ** retry
                    print(f'Retry attempt {retry}/{MAX_RETRIES} (wait={backoff}s)...')
                    time.sleep(backoff)
        if conn:
            conn.cursor().execute('SET search_path TO {}'.format(self.schema))
            conn.commit()
            return conn
        else:
            raise RuntimeError(f'Database connect error. Failed to connect after {MAX_RETRIES} retries.')

    @staticmethod
    def _adapt_datetime(dt: datetime) -> AsIs:
        return AsIs(f'{adapt(DateTime.iso8601(dt))}')

    @property
    def name(self) -> str:
        cursor = self.get_db().cursor()
        cursor.execute('SELECT current_database()')
        return cursor.fetchone()[0]

    @property
    def version(self) -> str:
        cursor = self.get_db().cursor()
        cursor.execute('SHOW server_version')
        return cursor.fetchone()[0]

    @property
    def is_alive(self) -> bool:
        cursor = self.get_db().cursor()
        cursor.execute('SELECT true')
        return cursor.fetchone()

    def close(self, db: psycopg2.extensions.connection) -> None:
        db.close()

    def destroy(self) -> None:
        conn = self.connect()
        cursor = conn.cursor()
        for table in ['alerts', 'blackouts', 'customers', 'groups', 'heartbeats', 
                     'keys', 'metrics', 'perms', 'users']:
            cursor.execute(f'DROP TABLE IF EXISTS {table}')
        conn.commit()
        conn.close()

    def get_severity(self, alert: Any) -> str:
        select = '\n            SELECT severity FROM alerts\n             WHERE environment=%(environment)s AND resource=%(resource)s\n               AND ((event=%(event)s AND severity!=%(severity)s)\n                OR (event!=%(event)s AND %(event)s=ANY(correlate)))\n               AND {customer}\n            '.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).severity

    def get_status(self, alert: Any) -> str:
        select = '\n            SELECT status FROM alerts\n             WHERE environment=%(environment)s AND resource=%(resource)s\n              AND (event=%(event)s OR %(event)s=ANY(correlate))\n              AND {customer}\n            '.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).status

    def is_duplicate(self, alert: Any) -> Any:
        select = '\n            SELECT * FROM alerts\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND event=%(event)s\n               AND severity=%(severity)s\n               AND {customer}\n            '.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert))

    def is_correlated(self, alert: Any) -> Any:
        select = '\n            SELECT * FROM alerts\n             WHERE environment=%(environment)s AND resource=%(resource)s\n               AND ((event=%(event)s AND severity!=%(severity)s)\n                OR (event!=%(event)s AND %(event)s=ANY(correlate)))\n               AND {customer}\n        '.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert))

    def is_flapping(self, alert: Any, window: int = 1800, count: int = 2) -> bool:
        select = "\n            SELECT COUNT(*)\n              FROM alerts, unnest(history) h\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND h.event=%(event)s\n               AND h.update_time > (NOW() at time zone 'utc' - INTERVAL '{window} seconds')\n               AND h.type='severity'\n               AND {customer}\n        ".format(window=window, customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).count > count

    def dedup_alert(self, alert: Any, history: Any) -> Any:
        alert.history = history
        update = '\n            UPDATE alerts\n               SET status=%(status)s, service=%(service)s, value=%(value)s, text=%(text)s,\n                   timeout=%(timeout)s, raw_data=%(raw_data)s, repeat=%(repeat)s,\n                   last_receive_id=%(last_receive_id)s, last_receive_time=%(last_receive_time)s,\n                   tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)), attributes=attributes || %(attributes)s,\n                   duplicate_count=duplicate_count + 1, {update_time}, history=(%(history)s || history)[1:{limit}]\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND event=%(event)s\n               AND severity=%(severity)s\n               AND {customer}\n         RETURNING *\n        '.format(limit=current_app.config['HISTORY_LIMIT'], update_time='update_time=%(update_time)s' if alert.update_time else 'update_time=update_time', customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._updateone(update, vars(alert), returning=True)

    def correlate_alert(self, alert: Any, history: Any) -> Any:
        alert.history = history
        update = '\n            UPDATE alerts\n               SET event=%(event)s, severity=%(severity)s, status=%(status)s, service=%(service)s, value=%(value)s,\n                   text=%(text)s, create_time=%(create_time)s, timeout=%(timeout)s, raw_data=%(raw_data)s,\n                   duplicate_count=%(duplicate_count)s, repeat=%(repeat)s, previous_severity=%(previous_severity)s,\n                   trend_indication=%(trend_indication)s, receive_time=%(receive_time)s, last_receive_id=%(last_receive_id)s,\n                   last_receive_time=%(last_receive_time)s, tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)),\n                   attributes=attributes || %(attributes)s, {update_time}, history=(%(history)s || history)[1:{limit}]\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND ((event=%(event)s AND severity!=%(severity)s) OR (event!=%(event)s AND %(event)s=ANY(correlate)))\n               AND {customer}\n         RETURNING *\n        '.format(limit=current_app.config['HISTORY_LIMIT'], update_time='update_time=%(update_time)s' if alert.update_time else 'update_time=update_time', customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._updateone(update, vars(alert), returning=True)

    def create_alert(self, alert: Any) -> Any:
        insert = '\n            INSERT INTO alerts (id, resource, event, environment, severity, correlate, status, service, "group",\n                value, text, tags, attributes, origin, type, create_time, timeout, raw_data, customer,\n                duplicate_count, repeat, previous_severity, trend_indication, receive_time, last_receive_id,\n                last_receive_time, update_time, history)\n            VALUES (%(id)s, %(resource)s, %(event)s, %(environment)s, %(severity)s, %(correlate)s, %(status)s,\n                %(service)s, %(group)s, %(value)s, %(text)s, %(tags)s, %(attributes)s, %(origin)s,\n                %(event_type)s, %(create_time)s, %(timeout)s, %(raw_data)s, %(customer)s, %(duplicate_count)s,\n                %(repeat)s, %(previous_severity)s, %(trend_indication)s, %(receive_time)s, %(last_receive_id)s,\n                %(last_receive_time)s, %(update_time)s, %(history)s::history[])\n            RETURNING *\n        '
        return self._insert(insert, vars(alert))

    def set_alert(self, id: str, severity: str, status: str, tags: List[str], 
                 attributes: Dict[str, Any], timeout: int, previous_severity: str, 
                 update_time: datetime, history: Optional[Any] = None) -> Any:
        update = '\n            UPDATE alerts\n               SET severity=%(severity)s, status=%(status)s, tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)),\n                   attributes=%(attributes)s, timeout=%(timeout)s, previous_severity=%(previous_severity)s,\n                   update_time=%(update_time)s, history=(%(change)s || history)[1:{limit}]\n             WHERE id=%(id)s OR id LIKE %(like_id)s\n         RETURNING *\n        '.format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {
            'id': id, 'like_id': id + '%', 'severity': severity, 'status': status, 
            'tags': tags, 'attributes': attributes, 'timeout': timeout, 
            'previous_severity': previous_severity, 'update_time': update_time, 
            'change': history
        }, returning=True)

    def get_alert(self, id: str, customers: Optional[List[str]] = None) -> Any:
        select = '\n            SELECT * FROM alerts\n             WHERE (id ~* (%(id)s) OR last_receive_id ~* (%(id)s))\n               AND {customer}\n        '.format(customer='customer=ANY(%(customers)s)' if customers else '1=1')
        return self._fetchone(select, {'id': '^' + id, 'customers': customers})

    def set_status(self, id: str, status: str, timeout: int, 
                  update_time: datetime, history: Optional[Any] = None) -> Any:
        update = '\n            UPDATE alerts\n            SET status=%(status)s, timeout=%(timeout)s, update_time=%(update_time)s, history=(%(change)s || history)[1:{limit}]\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '.format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {
            'id': id, 'like_id': id + '%', 'status': status, 
            'timeout': timeout, 'update_time': update_time, 'change': history
        }, returning=True)

    def tag_alert(self, id: str, tags: List[str]) -> Any:
        update = '\n            UPDATE alerts\n            SET tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s))\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def untag_alert(self, id: str, tags: List[str]) -> Any:
        update = '\n            UPDATE alerts\n            SET tags=(select array_agg(t) FROM unnest(tags) AS t WHERE NOT t=ANY(%(tags)s) )\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def update_tags(self, id: str, tags: List[str]) -> Any:
        update = '\n            UPDATE alerts\n            SET tags=%(tags)s\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def update_attributes(self, id: str, old_attrs: Dict[str, Any], 
                         new_attrs: Dict[str, Any]) -> Dict[str, Any]:
        old_attrs.update(new_attrs)
        attrs = {k: v for k, v in old_attrs.items() if v is not None}
        update = '\n            UPDATE alerts\n            SET attributes=%(attrs)s\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING attributes\n        '
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'attrs': attrs}, returning=True).attributes

    def delete_alert(self, id: str) -> Any:
        delete = '\n            DELETE FROM alerts\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING id\n        '
        return self._deleteone(delete, {'id': id, 'like_id': id + '%'}, returning=True)

    def tag_alerts(self, query: Optional[Query] = None, tags: Optional[List[str]] = None) -> List[str]:
        query = query or Query()
        update = f'\n            UPDATE alerts\n            SET tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(_tags)s))\n            WHERE {query.where}\n            RETURNING id\n        '
        return [row[0] for row in self._updateall(update, {**query.vars, **{'_tags': tags}}, returning=True)]

    def untag_alerts(self, query: Optional[Query] = None, tags: Optional[List[str]] = None) -> List[str]:
        query = query or Query()
        update = '\n            UPDATE alerts\n            SET tags=(select array_agg(t) FROM unnest(tags) AS t WHERE NOT t=ANY(%(_tags)s) )\n            WHERE {where}\n            RETURNING id\n        '.format(where=query.where)
        return [row[0] for row in self._updateall(update, {**query.vars, **{'_tags': tags}}, returning=True)]

    def update_attributes_by_query(self, query: Optional[Query] = None, 
                                