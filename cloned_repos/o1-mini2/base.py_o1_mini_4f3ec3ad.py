import threading
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

MAX_RETRIES: int = 5


class HistoryAdapter:
    def __init__(self, history: Any) -> None:
        self.history: Any = history
        self.conn: Optional[Any] = None

    def prepare(self, conn: Any) -> None:
        self.conn = conn

    def getquoted(self) -> str:
        def quoted(o: Any) -> str:
            a = adapt(o)
            if hasattr(a, 'prepare'):
                a.prepare(self.conn)
            return a.getquoted().decode('utf-8')

        return '({}, {}, {}, {}, {}, {}, {}, {}::timestamp, {}, {})::history'.format(
            quoted(self.history.id),
            quoted(self.history.event),
            quoted(self.history.severity),
            quoted(self.history.status),
            quoted(self.history.value),
            quoted(self.history.text),
            quoted(self.history.change_type),
            quoted(self.history.update_time),
            quoted(self.history.user),
            quoted(self.history.timeout)
        )

    def __str__(self) -> str:
        return str(self.getquoted())


class Record(namedtuple('Record', [
    'id', 'resource', 'event', 'environment', 'severity', 'status', 'service',
    'group', 'value', 'text', 'tags', 'attributes', 'origin', 'update_time',
    'user', 'timeout', 'type', 'customer'
])):
    id: str
    resource: str
    event: str
    environment: str
    severity: str
    status: str
    service: List[str]
    group: Optional[str]
    value: str
    text: str
    tags: List[str]
    attributes: Dict[str, Any]
    origin: str
    update_time: datetime
    user: Optional[str]
    timeout: Optional[int]
    type: str
    customer: Optional[str]


class Backend(Database):

    def create_engine(
        self,
        app: Any,
        uri: str,
        dbname: Optional[str] = None,
        schema: str = 'public',
        raise_on_error: bool = True
    ) -> None:
        self.uri: str = uri
        self.dbname: Optional[str] = dbname
        self.schema: str = schema

        lock = threading.Lock()
        with lock:
            conn: Any = self.connect()

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
        register_composite(
            f'{self.schema}.history' if self.schema else 'history',
            conn,
            globally=True
        )
        from alerta.models.alert import History
        register_adapter(History, HistoryAdapter)

    def connect(self) -> Any:
        retry: int = 0
        while True:
            try:
                conn: Any = psycopg2.connect(
                    dsn=self.uri,
                    dbname=self.dbname,
                    cursor_factory=NamedTupleCursor
                )

                conn.set_client_encoding('UTF8')
                break
            except Exception as e:
                print(e)  # FIXME - should log this error instead of printing, but current_app is unavailable here
                retry += 1
                if retry > MAX_RETRIES:
                    conn = None
                    break
                else:
                    backoff: int = 2 ** retry
                    print(f'Retry attempt {retry}/{MAX_RETRIES} (wait={backoff}s)...')
                    time.sleep(backoff)

        if conn:
            conn.cursor().execute(f'SET search_path TO {self.schema}')
            conn.commit()
            return conn
        else:
            raise RuntimeError(f'Database connect error. Failed to connect after {MAX_RETRIES} retries.')

    @staticmethod
    def _adapt_datetime(dt: datetime) -> AsIs:
        return AsIs(f'{adapt(DateTime.iso8601(dt))}')

    @property
    def name(self) -> str:
        cursor: Any = self.get_db().cursor()
        cursor.execute('SELECT current_database()')
        return cursor.fetchone()[0]

    @property
    def version(self) -> str:
        cursor: Any = self.get_db().cursor()
        cursor.execute('SHOW server_version')
        return cursor.fetchone()[0]

    @property
    def is_alive(self) -> bool:
        cursor: Any = self.get_db().cursor()
        cursor.execute('SELECT true')
        return bool(cursor.fetchone()[0])

    def close(self, db: Any) -> None:
        db.close()

    def destroy(self) -> None:
        conn: Any = self.connect()
        cursor: Any = conn.cursor()
        for table in ['alerts', 'blackouts', 'customers', 'groups', 'heartbeats', 'keys', 'metrics', 'perms', 'users']:
            cursor.execute(f'DROP TABLE IF EXISTS {table}')
        conn.commit()
        conn.close()

    # ALERTS

    def get_severity(self, alert: Any) -> str:
        select: str = """
            SELECT severity FROM alerts
             WHERE environment=%(environment)s AND resource=%(resource)s
               AND ((event=%(event)s AND severity!=%(severity)s)
                OR (event!=%(event)s AND %(event)s=ANY(correlate)))
               AND {customer}
            """.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).severity

    def get_status(self, alert: Any) -> str:
        select: str = """
            SELECT status FROM alerts
             WHERE environment=%(environment)s AND resource=%(resource)s
              AND (event=%(event)s OR %(event)s=ANY(correlate))
              AND {customer}
            """.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).status

    def is_duplicate(self, alert: Any) -> Optional[Any]:
        select: str = """
            SELECT * FROM alerts
             WHERE environment=%(environment)s
               AND resource=%(resource)s
               AND event=%(event)s
               AND severity=%(severity)s
               AND {customer}
            """.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert))

    def is_correlated(self, alert: Any) -> Optional[Any]:
        select: str = """
            SELECT * FROM alerts
             WHERE environment=%(environment)s AND resource=%(resource)s
               AND ((event=%(event)s AND severity!=%(severity)s)
                OR (event!=%(event)s AND %(event)s=ANY(correlate)))
               AND {customer}
        """.format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert))

    def is_flapping(self, alert: Any, window: int = 1800, count: int = 2) -> bool:
        """
        Return true if alert severity has changed more than X times in Y seconds
        """
        select: str = """
            SELECT COUNT(*)
              FROM alerts, unnest(history) h
             WHERE environment=%(environment)s
               AND resource=%(resource)s
               AND h.event=%(event)s
               AND h.update_time > (NOW() at time zone 'utc' - INTERVAL '{window} seconds')
               AND h.type='severity'
               AND {customer}
        """.format(window=window, customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).count > count

    def dedup_alert(self, alert: Any, history: Any) -> Optional[Any]:
        """
        Update alert status, service, value, text, timeout and rawData, increment duplicate count and set
        repeat=True, and keep track of last receive id and time but don't append to history unless status changes.
        """
        alert.history = history
        update: str = """
            UPDATE alerts
               SET status=%(status)s, service=%(service)s, value=%(value)s, text=%(text)s,
                   timeout=%(timeout)s, raw_data=%(raw_data)s, repeat=%(repeat)s,
                   last_receive_id=%(last_receive_id)s, last_receive_time=%(last_receive_time)s,
                   tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)), attributes=attributes || %(attributes)s,
                   duplicate_count=duplicate_count + 1, {update_time}, history=(%(history)s || history)[1:{limit}]
             WHERE environment=%(environment)s
               AND resource=%(resource)s
               AND event=%(event)s
               AND severity=%(severity)s
               AND {customer}
         RETURNING *
        """.format(
            limit=current_app.config['HISTORY_LIMIT'],
            update_time='update_time=%(update_time)s' if alert.update_time else 'update_time=update_time',
            customer='customer=%(customer)s' if alert.customer else 'customer IS NULL'
        )
        return self._updateone(update, vars(alert), returning=True)

    def correlate_alert(self, alert: Any, history: Any) -> Optional[Any]:
        alert.history = history
        update: str = """
            UPDATE alerts
               SET event=%(event)s, severity=%(severity)s, status=%(status)s, service=%(service)s, value=%(value)s,
                   text=%(text)s, create_time=%(create_time)s, timeout=%(timeout)s, raw_data=%(raw_data)s,
                   duplicate_count=%(duplicate_count)s, repeat=%(repeat)s, previous_severity=%(previous_severity)s,
                   trend_indication=%(trend_indication)s, receive_time=%(receive_time)s, last_receive_id=%(last_receive_id)s,
                   last_receive_time=%(last_receive_time)s, tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)),
                   attributes=attributes || %(attributes)s, {update_time}, history=(%(history)s || history)[1:{limit}]
             WHERE environment=%(environment)s
               AND resource=%(resource)s
               AND ((event=%(event)s AND severity!=%(severity)s) OR (event!=%(event)s AND %(event)s=ANY(correlate)))
               AND {customer}
         RETURNING *
        """.format(
            limit=current_app.config['HISTORY_LIMIT'],
            update_time='update_time=%(update_time)s' if alert.update_time else 'update_time=update_time',
            customer='customer=%(customer)s' if alert.customer else 'customer IS NULL'
        )
        return self._updateone(update, vars(alert), returning=True)

    def create_alert(self, alert: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO alerts (id, resource, event, environment, severity, correlate, status, service, "group",
                value, text, tags, attributes, origin, type, create_time, timeout, raw_data, customer,
                duplicate_count, repeat, previous_severity, trend_indication, receive_time, last_receive_id,
                last_receive_time, update_time, history)
            VALUES (%(id)s, %(resource)s, %(event)s, %(environment)s, %(severity)s, %(correlate)s, %(status)s,
                %(service)s, %(group)s, %(value)s, %(text)s, %(tags)s, %(attributes)s, %(origin)s,
                %(event_type)s, %(create_time)s, %(timeout)s, %(raw_data)s, %(customer)s, %(duplicate_count)s,
                %(repeat)s, %(previous_severity)s, %(trend_indication)s, %(receive_time)s, %(last_receive_id)s,
                %(last_receive_time)s, %(update_time)s, %(history)s::history[])
            RETURNING *
        """
        return self._insert(insert, vars(alert))

    def set_alert(
        self,
        id: str,
        severity: str,
        status: str,
        tags: List[str],
        attributes: Dict[str, Any],
        timeout: int,
        previous_severity: str,
        update_time: datetime,
        history: Optional[Any] = None
    ) -> Optional[Any]:
        update: str = """
            UPDATE alerts
               SET severity=%(severity)s, status=%(status)s, tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)),
                   attributes=%(attributes)s, timeout=%(timeout)s, previous_severity=%(previous_severity)s,
                   update_time=%(update_time)s, history=(%(change)s || history)[1:{limit}]
             WHERE id=%(id)s OR id LIKE %(like_id)s
         RETURNING *
        """.format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'severity': severity, 'status': status,
                                        'tags': tags, 'attributes': attributes, 'timeout': timeout,
                                        'previous_severity': previous_severity, 'update_time': update_time,
                                        'change': history}, returning=True)

    def get_alert(
        self,
        id: str,
        customers: Optional[List[str]] = None
    ) -> Optional[Any]:
        select: str = """
            SELECT * FROM alerts
             WHERE (id ~* (%(id)s) OR last_receive_id ~* (%(id)s))
               AND {customer}
        """.format(customer='customer=ANY(%(customers)s)' if customers else '1=1')
        return self._fetchone(select, {'id': '^' + id, 'customers': customers})

    # STATUS, TAGS, ATTRIBUTES

    def set_status(
        self,
        id: str,
        status: str,
        timeout: int,
        update_time: datetime,
        history: Optional[Any] = None
    ) -> Optional[Any]:
        update: str = """
            UPDATE alerts
            SET status=%(status)s, timeout=%(timeout)s, update_time=%(update_time)s, history=(%(change)s || history)[1:{limit}]
            WHERE id=%(id)s OR id LIKE %(like_id)s
            RETURNING *
        """.format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'status': status, 'timeout': timeout, 'update_time': update_time, 'change': history}, returning=True)

    def tag_alert(self, id: str, tags: List[str]) -> Optional[Any]:
        update: str = """
            UPDATE alerts
            SET tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s))
            WHERE id=%(id)s OR id LIKE %(like_id)s
            RETURNING *
        """
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def untag_alert(self, id: str, tags: List[str]) -> Optional[Any]:
        update: str = """
            UPDATE alerts
            SET tags=(select array_agg(t) FROM unnest(tags) AS t WHERE NOT t=ANY(%(tags)s) )
            WHERE id=%(id)s OR id LIKE %(like_id)s
            RETURNING *
        """
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def update_tags(self, id: str, tags: List[str]) -> Optional[Any]:
        update: str = """
            UPDATE alerts
            SET tags=%(tags)s
            WHERE id=%(id)s OR id LIKE %(like_id)s
            RETURNING *
        """
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def update_attributes(
        self,
        id: str,
        old_attrs: Dict[str, Any],
        new_attrs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        old_attrs.update(new_attrs)
        attrs: Dict[str, Any] = {k: v for k, v in old_attrs.items() if v is not None}

        update: str = """
            UPDATE alerts
               SET attributes=%(attrs)s
            WHERE id=%(id)s OR id LIKE %(like_id)s
            RETURNING attributes
        """
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'attrs': attrs}, returning=True).attributes

    def delete_alert(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM alerts
            WHERE id=%(id)s OR id LIKE %(like_id)s
            RETURNING id
        """
        return self._deleteone(delete, {'id': id, 'like_id': id + '%'}, returning=True)

    # BULK

    def tag_alerts(
        self,
        query: Optional[Query] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        query = query or Query()
        update: str = f"""
            UPDATE alerts
            SET tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(_tags)s))
            WHERE {query.where}
            RETURNING id
        """
        return [row[0] for row in self._updateall(update, {**query.vars, **{'_tags': tags}}, returning=True)]

    def untag_alerts(
        self,
        query: Optional[Query] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        query = query or Query()
        update: str = """
            UPDATE alerts
            SET tags=(select array_agg(t) FROM unnest(tags) AS t WHERE NOT t=ANY(%(_tags)s) )
            WHERE {where}
            RETURNING id
        """.format(where=query.where)
        return [row[0] for row in self._updateall(update, {**query.vars, **{'_tags': tags}}, returning=True)]

    def update_attributes_by_query(
        self,
        query: Optional[Query] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        update: str = f"""
            UPDATE alerts
            SET attributes=attributes || %(_attributes)s
            WHERE {query.where}
            RETURNING id
        """
        return [row[0] for row in self._updateall(update, {**query.vars, **{'_attributes': attributes}}, returning=True)]

    def delete_alerts(
        self,
        query: Optional[Query] = None
    ) -> List[str]:
        query = query or Query()
        delete: str = f"""
            DELETE FROM alerts
            WHERE {query.where}
            RETURNING id
        """
        return [row[0] for row in self._deleteall(delete, query.vars, returning=True)]

    # SEARCH & HISTORY

    def add_history(self, id: str, history: Any) -> Optional[Any]:
        update: str = """
            UPDATE alerts
               SET history=(%(history)s || history)[1:{limit}]
             WHERE id=%(id)s OR id LIKE %(like_id)s
         RETURNING *
        """.format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'history': history}, returning=True)

    def get_alerts(
        self,
        query: Optional[Query] = None,
        raw_data: bool = False,
        history: bool = False,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        if raw_data and history:
            select: str = '*'
        else:
            select: str = (
                'id, resource, event, environment, severity, correlate, status, service, "group", value, "text",'
                + 'tags, attributes, origin, type, create_time, timeout, {raw_data}, customer, duplicate_count, repeat,'
                + 'previous_severity, trend_indication, receive_time, last_receive_id, last_receive_time, update_time,'
                + '{history}'
            ).format(
                raw_data='raw_data' if raw_data else 'NULL as raw_data',
                history='history' if history else 'array[]::history[] as history'
            )

        join: str = ''
        if 's.code' in query.sort:
            join += 'JOIN (VALUES {}) AS s(sev, code) ON alerts.severity = s.sev '.format(
                ', '.join((f"('{k}', {v})" for k, v in alarm_model.Severity.items()))
            )
        if 'st.state' in query.sort:
            join += 'JOIN (VALUES {}) AS st(sts, state) ON alerts.status = st.sts '.format(
                ', '.join((f"('{k}', '{v}')" for k, v in alarm_model.Status.items()))
            )
        select_query: str = f"""
            SELECT {select}
              FROM alerts {join}
             WHERE {query.where}
          ORDER BY {query.sort or 'last_receive_time'}
        """
        return self._fetchall(select_query, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_alert_history(
        self,
        alert: Any,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Record]:
        select: str = """
            SELECT resource, environment, service, "group", tags, attributes, origin, customer, h.*
              FROM alerts, unnest(history[1:{limit}]) h
             WHERE environment=%(environment)s AND resource=%(resource)s
               AND (h.event=%(event)s OR %(event)s=ANY(correlate))
               AND {customer}
          ORDER BY update_time DESC
            """.format(
            customer='customer=%(customer)s' if alert.customer else 'customer IS NULL',
            limit=current_app.config['HISTORY_LIMIT']
        )
        records: List[Record] = [
            Record(
                id=h.id,
                resource=h.resource,
                event=h.event,
                environment=h.environment,
                severity=h.severity,
                status=h.status,
                service=h.service,
                group=h.group,
                value=h.value,
                text=h.text,
                tags=h.tags,
                attributes=h.attributes,
                origin=h.origin,
                update_time=h.update_time,
                user=getattr(h, 'user', None),
                timeout=getattr(h, 'timeout', None),
                type=h.type,
                customer=h.customer
            ) for h in self._fetchall(select, vars(alert), limit=page_size, offset=(page - 1) * page_size)
        ]
        return records

    def get_history(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Record]:
        query = query or Query()
        if 'id' in query.vars:
            select: str = """
                SELECT a.id
                  FROM alerts a, unnest(history[1:{limit}]) h
                 WHERE h.id LIKE %(id)s
            """.format(limit=current_app.config['HISTORY_LIMIT'])
            query.vars['id'] = self._fetchone(select, query.vars)

        select_query: str = """
            SELECT resource, environment, service, "group", tags, attributes, origin, customer, history, h.*
              FROM alerts, unnest(history[1:{limit}]) h
             WHERE {where}
          ORDER BY update_time DESC
        """.format(where=query.where, limit=current_app.config['HISTORY_LIMIT'])

        records: List[Record] = [
            Record(
                id=h.id,
                resource=h.resource,
                event=h.event,
                environment=h.environment,
                severity=h.severity,
                status=h.status,
                service=h.service,
                group=h.group,
                value=h.value,
                text=h.text,
                tags=h.tags,
                attributes=h.attributes,
                origin=h.origin,
                update_time=h.update_time,
                user=getattr(h, 'user', None),
                timeout=getattr(h, 'timeout', None),
                type=h.type,
                customer=h.customer
            ) for h in self._fetchall(select_query, query.vars, limit=page_size, offset=(page - 1) * page_size)
        ]
        return records

    # COUNTS

    def get_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM alerts
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def get_counts(self, query: Optional[Query] = None, group: Optional[str] = None) -> Dict[str, int]:
        query = query or Query()
        if group is None:
            raise ValueError('Must define a group')
        select: str = """
            SELECT {group}, COUNT(*) FROM alerts
             WHERE {where}
            GROUP BY {group}
        """.format(where=query.where, group=group)
        return {s[group]: s.count for s in self._fetchall(select, query.vars)}

    def get_counts_by_severity(self, query: Optional[Query] = None) -> Dict[str, int]:
        query = query or Query()
        select: str = f"""
            SELECT severity, COUNT(*) FROM alerts
             WHERE {query.where}
            GROUP BY severity
        """
        return {s.severity: s.count for s in self._fetchall(select, query.vars)}

    def get_counts_by_status(self, query: Optional[Query] = None) -> Dict[str, int]:
        query = query or Query()
        select: str = f"""
            SELECT status, COUNT(*) FROM alerts
            WHERE {query.where}
            GROUP BY status
        """
        return {s.status: s.count for s in self._fetchall(select, query.vars)}

    def get_topn_count(
        self,
        query: Optional[Query] = None,
        topn: int = 100
    ) -> List[Dict[str, Any]]:
        query = query or Query()
        group: str = 'event'
        if query and query.group:
            group = query.group[0]

        select: str = """
            SELECT {group}, COUNT(1) as count, SUM(duplicate_count) AS duplicate_count,
                   array_agg(DISTINCT environment) AS environments, array_agg(DISTINCT svc) AS services,
                   array_agg(DISTINCT ARRAY[id, resource]) AS resources
              FROM alerts, UNNEST (service) svc
             WHERE {where}
          GROUP BY {group}
          ORDER BY count DESC
        """.format(where=query.where, group=group)
        return [
            {
                'count': t.count,
                'duplicateCount': t.duplicate_count,
                'environments': t.environments,
                'services': t.services,
                group: getattr(t, group),
                'resources': [{'id': r[0], 'resource': r[1], 'href': absolute_url(f'/alert/{r[0]}')} for r in t.resources]
            } for t in self._fetchall(select, query.vars, limit=topn)
        ]

    def get_topn_flapping(
        self,
        query: Optional[Query] = None,
        topn: int = 100
    ) -> List[Dict[str, Any]]:
        query = query or Query()
        group: str = 'event'
        if query and query.group:
            group = query.group[0]
        select: str = """
            WITH topn AS (SELECT * FROM alerts WHERE {where})
            SELECT topn.{group}, COUNT(1) as count, SUM(duplicate_count) AS duplicate_count,
                   array_agg(DISTINCT environment) AS environments, array_agg(DISTINCT svc) AS services,
                   array_agg(DISTINCT ARRAY[topn.id, resource]) AS resources
              FROM topn, UNNEST (service) svc, UNNEST (history) hist
             WHERE hist.type='severity'
          GROUP BY topn.{group}
          ORDER BY count DESC
        """.format(where=query.where, group=group)
        return [
            {
                'count': t.count,
                'duplicateCount': t.duplicate_count,
                'environments': t.environments,
                'services': t.services,
                group: getattr(t, group),
                'resources': [{'id': r[0], 'resource': r[1], 'href': absolute_url(f'/alert/{r[0]}')} for r in t.resources]
            } for t in self._fetchall(select, query.vars, limit=topn)
        ]

    def get_topn_standing(
        self,
        query: Optional[Query] = None,
        topn: int = 100
    ) -> List[Dict[str, Any]]:
        query = query or Query()
        group: str = 'event'
        if query and query.group:
            group = query.group[0]
        select: str = """
            WITH topn AS (SELECT * FROM alerts WHERE {where})
            SELECT topn.{group}, COUNT(1) as count, SUM(duplicate_count) AS duplicate_count,
                   SUM(last_receive_time - create_time) as life_time,
                   array_agg(DISTINCT environment) AS environments, array_agg(DISTINCT svc) AS services,
                   array_agg(DISTINCT ARRAY[topn.id, resource]) AS resources
              FROM topn, UNNEST (service) svc, UNNEST (history) hist
             WHERE hist.type='severity'
          GROUP BY topn.{group}
          ORDER BY life_time DESC
        """.format(where=query.where, group=group)
        return [
            {
                'count': t.count,
                'duplicateCount': t.duplicate_count,
                'environments': t.environments,
                'services': t.services,
                group: getattr(t, group),
                'resources': [{'id': r[0], 'resource': r[1], 'href': absolute_url(f'/alert/{r[0]}')} for r in t.resources]
            } for t in self._fetchall(select, query.vars, limit=topn)
        ]

    # ENVIRONMENTS

    def get_environments(
        self,
        query: Optional[Query] = None,
        topn: int = 1000
    ) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = f"""
            SELECT environment, severity, status, count(1) FROM alerts
            WHERE {query.where}
            GROUP BY environment, CUBE(severity, status)
        """
        result: List[Any] = self._fetchall(select, query.vars, limit=topn)

        severity_count: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        status_count: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        total_count: Dict[str, int] = defaultdict(int)

        for row in result:
            if row.severity and not row.status:
                severity_count[row.environment].append((row.severity, row.count))
            if not row.severity and row.status:
                status_count[row.environment].append((row.status, row.count))
            if not row.severity and not row.status:
                total_count[row.environment] = row.count

        select_env: str = """SELECT DISTINCT environment FROM alerts"""
        environments: List[Any] = self._fetchall(select_env, {})
        return [
            {
                'environment': e.environment,
                'severityCounts': dict(severity_count[e.environment]),
                'statusCounts': dict(status_count[e.environment]),
                'count': total_count[e.environment]
            } for e in environments
        ]

    # SERVICES

    def get_services(
        self,
        query: Optional[Query] = None,
        topn: int = 1000
    ) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = """
            SELECT environment, svc, severity, status, count(1) FROM alerts, UNNEST(service) svc
            WHERE {where}
            GROUP BY environment, svc, CUBE(severity, status)
        """.format(where=query.where)
        result: List[Any] = self._fetchall(select, query.vars, limit=topn)

        severity_count: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
        status_count: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
        total_count: Dict[Tuple[str, str], int] = defaultdict(int)

        for row in result:
            if row.severity and not row.status:
                severity_count[(row.environment, row.svc)].append((row.severity, row.count))
            if not row.severity and row.status:
                status_count[(row.environment, row.svc)].append((row.status, row.count))
            if not row.severity and not row.status:
                total_count[(row.environment, row.svc)] = row.count

        select_svc: str = """SELECT DISTINCT environment, svc FROM alerts, UNNEST(service) svc"""
        services: List[Any] = self._fetchall(select_svc, {})
        return [
            {
                'environment': s.environment,
                'service': s.svc,
                'severityCounts': dict(severity_count[(s.environment, s.svc)]),
                'statusCounts': dict(status_count[(s.environment, s.svc)]),
                'count': total_count[(s.environment, s.svc)]
            } for s in services
        ]

    # ALERT GROUPS

    def get_alert_groups(
        self,
        query: Optional[Query] = None,
        topn: int = 1000
    ) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = f"""
            SELECT environment, "group", count(1) FROM alerts
            WHERE {query.where}
            GROUP BY environment, "group"
        """
        return [
            {
                'environment': g.environment,
                'group': g.group,
                'count': g.count
            } for g in self._fetchall(select, query.vars, limit=topn)
        ]

    # ALERT TAGS

    def get_alert_tags(
        self,
        query: Optional[Query] = None,
        topn: int = 1000
    ) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = """
            SELECT environment, tag, count(1) FROM alerts, UNNEST(tags) tag
            WHERE {where}
            GROUP BY environment, tag
        """.format(where=query.where)
        return [{'environment': t.environment, 'tag': t.tag, 'count': t.count} for t in self._fetchall(select, query.vars, limit=topn)]

    # BLACKOUTS

    def create_blackout(self, blackout: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO blackouts (id, priority, environment, service, resource, event,
                "group", tags, origin, customer, start_time, end_time,
                duration, "user", create_time, text)
            VALUES (%(id)s, %(priority)s, %(environment)s, %(service)s, %(resource)s, %(event)s,
                %(group)s, %(tags)s, %(origin)s, %(customer)s, %(start_time)s, %(end_time)s,
                %(duration)s, %(user)s, %(create_time)s, %(text)s)
            RETURNING *, duration AS remaining
        """
        return self._insert(insert, vars(blackout))

    def get_blackout(
        self,
        id: str,
        customers: Optional[List[str]] = None
    ) -> Optional[Any]:
        select: str = """
            SELECT *, GREATEST(EXTRACT(EPOCH FROM (end_time - GREATEST(start_time, NOW() at time zone 'utc'))), 0) AS remaining
            FROM blackouts
            WHERE id=%(id)s
              AND {customer}
        """.format(customer='customer=ANY(%(customers)s)' if customers else '1=1')
        return self._fetchone(select, {'id': id, 'customers': customers})

    def get_blackouts(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = """
            SELECT *, GREATEST(EXTRACT(EPOCH FROM (end_time - GREATEST(start_time, NOW() at time zone 'utc'))), 0) AS remaining
              FROM blackouts
             WHERE {where}
          ORDER BY {order}
        """.format(where=query.where, order=query.sort)
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_blackouts_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM blackouts
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def is_blackout_period(self, alert: Any) -> bool:
        select: str = """
            SELECT *
            FROM blackouts
            WHERE start_time <= %(create_time)s AND end_time > %(create_time)s
              AND environment=%(environment)s
              AND (
                 ( resource IS NULL AND service='{}' AND event IS NULL AND "group" IS NULL AND tags='{}' AND origin IS NULL )
              OR ( resource IS NULL AND service='{}' AND event IS NULL AND "group" IS NULL AND tags='{}' AND origin=%(origin)s )
              OR ( resource IS NULL AND service='{}' AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource IS NULL AND service='{}' AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )
              OR ( resource IS NULL AND service='{}' AND event IS NULL AND "group"=%(group)s AND tags='{}' AND origin IS NULL )
              OR ( resource IS NULL AND service='{}' AND event IS NULL AND "group"=%(group)s AND tags='{}' AND origin=%(origin)s )
              OR ( resource IS NULL AND service='{}' AND event IS NULL AND "group"=%(group)s AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource IS NULL AND service='{}' AND event IS NULL AND "group"=%(group)s AND tags <@ %(tags)s AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group" IS NULL AND tags='{}' AND origin IS NULL )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group" IS NULL AND tags='{}' AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group"=%(group)s AND tags='{}' AND origin IS NULL )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group"=%(group)s AND tags='{}' AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group"=%(group)s AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource=%(resource)s AND service='{}' AND event=%(event)s AND "group"=%(group)s AND tags <@ %(tags)s AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags='{}' AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags='{}' AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group"=%(group)s AND tags='{}' AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group"=%(group)s AND tags='{}' AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group"=%(group)s AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event IS NULL AND "group"=%(group)s AND tags <@ %(tags)s AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags='{}' AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags='{}' AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags='{}' AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags='{}' AND origin=%(origin)s )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags <@ %(tags)s AND origin IS NULL )
              OR ( resource=%(resource)s AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags <@ %(tags)s AND origin=%(origin)s )
                 )
        """
        if current_app.config['CUSTOMER_VIEWS']:
            select += ' AND (customer IS NULL OR customer=%(customer)s)'
        if self._fetchone(select, vars(alert)):
            return True
        return False

    def update_blackout(self, id: str, **kwargs: Any) -> Optional[Any]:
        update: str = """
            UPDATE blackouts
            SET
        """
        if kwargs.get('environment') is not None:
            update += 'environment=%(environment)s, '
        if 'service' in kwargs:
            update += 'service=%(service)s, '
        if 'resource' in kwargs:
            update += 'resource=%(resource)s, '
        if 'event' in kwargs:
            update += 'event=%(event)s, '
        if 'group' in kwargs:
            update += '"group"=%(group)s, '
        if 'tags' in kwargs:
            update += 'tags=%(tags)s, '
        if 'origin' in kwargs:
            update += 'origin=%(origin)s, '
        if 'customer' in kwargs:
            update += 'customer=%(customer)s, '
        if kwargs.get('startTime') is not None:
            update += 'start_time=%(startTime)s, '
        if kwargs.get('endTime') is not None:
            update += 'end_time=%(endTime)s, '
        if 'duration' in kwargs:
            update += 'duration=%(duration)s, '
        if 'text' in kwargs:
            update += 'text=%(text)s, '
        update += """
            "user"=COALESCE(%(user)s, "user")
            WHERE id=%(id)s
            RETURNING *, GREATEST(EXTRACT(EPOCH FROM (end_time - GREATEST(start_time, NOW() at time zone 'utc'))), 0) AS remaining
        """
        kwargs['id'] = id
        kwargs['user'] = kwargs.get('user')
        return self._updateone(update, kwargs, returning=True)

    def delete_blackout(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM blackouts
            WHERE id=%s
            RETURNING id
        """
        return self._deleteone(delete, (id,), returning=True)

    # HEARTBEATS

    def upsert_heartbeat(self, heartbeat: Any) -> Optional[Any]:
        upsert: str = """
            INSERT INTO heartbeats (id, origin, tags, attributes, type, create_time, timeout, receive_time, customer)
            VALUES (%(id)s, %(origin)s, %(tags)s, %(attributes)s, %(event_type)s, %(create_time)s, %(timeout)s, %(receive_time)s, %(customer)s)
            ON CONFLICT (origin, COALESCE(customer, '')) DO UPDATE
                SET tags=%(tags)s, attributes=%(attributes)s, create_time=%(create_time)s, timeout=%(timeout)s, receive_time=%(receive_time)s
            RETURNING *,
                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,
                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since
        """
        return self._upsert(upsert, vars(heartbeat))

    def get_heartbeat(
        self,
        id: str,
        customers: Optional[List[str]] = None
    ) -> Optional[Any]:
        select: str = """
            SELECT *,
                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,
                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since
              FROM heartbeats
             WHERE (id=%(id)s OR id LIKE %(like_id)s)
               AND {customer}
        """.format(customer='customer=ANY(%(customers)s)' if customers else '1=1')
        return self._fetchone(select, {'id': id, 'like_id': id + '%', 'customers': customers})

    def get_heartbeats(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = """
            SELECT *,
                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,
                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since
              FROM heartbeats
             WHERE {where}
          ORDER BY {order}
        """.format(where=query.where, order=query.sort)
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_heartbeats_by_status(
        self,
        status: Optional[List[HeartbeatStatus]] = None,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        status = status or []
        query = query or Query()

        swhere: str = ''
        if status:
            q: List[str] = []
            if HeartbeatStatus.OK in status:
                q.append(
                    """
                    (EXTRACT(EPOCH FROM (NOW() at time zone 'utc' - receive_time)) <= timeout
                    AND EXTRACT(EPOCH FROM (receive_time - create_time)) * 1000 <= {max_latency})
                    """.format(max_latency=current_app.config['HEARTBEAT_MAX_LATENCY'])
                )
            if HeartbeatStatus.Expired in status:
                q.append("(EXTRACT(EPOCH FROM (NOW() at time zone 'utc' - receive_time)) > timeout)")
            if HeartbeatStatus.Slow in status:
                q.append(
                    """
                    (EXTRACT(EPOCH FROM (NOW() at time zone 'utc' - receive_time)) <= timeout
                    AND EXTRACT(EPOCH FROM (receive_time - create_time)) * 1000 > {max_latency})
                    """.format(max_latency=current_app.config['HEARTBEAT_MAX_LATENCY'])
                )
            if q:
                swhere = 'AND (' + ' OR '.join(q) + ')'

        select: str = """
            SELECT *,
                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,
                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since
              FROM heartbeats
             WHERE {where}
             {swhere}
          ORDER BY {order}
        """.format(where=query.where, swhere=swhere, order=query.sort)
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_heartbeats_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM heartbeats
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def delete_heartbeat(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM heartbeats
            WHERE id=%(id)s OR id LIKE %(like_id)s
            RETURNING id
        """
        return self._deleteone(delete, {'id': id, 'like_id': id + '%'}, returning=True)

    # API KEYS

    def create_key(self, key: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO keys (id, key, "user", scopes, text, expire_time, "count", last_used_time, customer)
            VALUES (%(id)s, %(key)s, %(user)s, %(scopes)s, %(text)s, %(expire_time)s, %(count)s, %(last_used_time)s, %(customer)s)
            RETURNING *
        """
        return self._insert(insert, vars(key))

    def get_key(self, key: str, user: Optional[str] = None) -> Optional[Any]:
        select: str = f"""
            SELECT * FROM keys
             WHERE (id=%(key)s OR key=%(key)s)
               AND {'"user"=%(user)s' if user else '1=1'}
        """
        return self._fetchone(select, {'key': key, 'user': user})

    def get_keys(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = f"""
            SELECT * FROM keys
             WHERE {query.where}
          ORDER BY {query.sort}
        """
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_keys_by_user(self, user: str) -> List[Any]:
        select: str = """
            SELECT * FROM keys
             WHERE "user"=%s
        """
        return self._fetchall(select, (user,))

    def get_keys_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM keys
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def update_key(self, key: str, **kwargs: Any) -> Optional[Any]:
        update: str = """
            UPDATE keys
            SET
        """
        if 'user' in kwargs:
            update += '"user"=%(user)s, '
        if 'scopes' in kwargs:
            update += 'scopes=%(scopes)s, '
        if 'text' in kwargs:
            update += 'text=%(text)s, '
        if 'expireTime' in kwargs:
            update += 'expire_time=%(expireTime)s, '
        if 'customer' in kwargs:
            update += 'customer=%(customer)s, '
        update += """
            id=id
            WHERE (id=%(key)s OR key=%(key)s)
            RETURNING *
        """
        kwargs['key'] = key
        return self._updateone(update, kwargs, returning=True)

    def update_key_last_used(self, key: str) -> Optional[Any]:
        update: str = """
            UPDATE keys
            SET last_used_time=NOW() at time zone 'utc', count=count + 1
            WHERE id=%s OR key=%s
        """
        return self._updateone(update, (key, key))

    def delete_key(self, key: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM keys
            WHERE id=%s OR key=%s
            RETURNING key
        """
        return self._deleteone(delete, (key, key), returning=True)

    # USERS

    def create_user(self, user: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO users (id, name, login, password, email, status, roles, attributes,
                create_time, last_login, text, update_time, email_verified)
            VALUES (%(id)s, %(name)s, %(login)s, %(password)s, %(email)s, %(status)s, %(roles)s, %(attributes)s, %(create_time)s,
                %(last_login)s, %(text)s, %(update_time)s, %(email_verified)s)
            RETURNING *
        """
        return self._insert(insert, vars(user))

    def get_user(self, id: str) -> Optional[Any]:
        select: str = """SELECT * FROM users WHERE id=%s"""
        return self._fetchone(select, (id,))

    def get_users(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = f"""
            SELECT * FROM users
             WHERE {query.where}
          ORDER BY {query.sort}
        """
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_users_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM users
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def get_user_by_username(self, username: str) -> Optional[Any]:
        select: str = """SELECT * FROM users WHERE login=%s OR email=%s"""
        return self._fetchone(select, (username, username))

    def get_user_by_email(self, email: str) -> Optional[Any]:
        select: str = """SELECT * FROM users WHERE email=%s"""
        return self._fetchone(select, (email,))

    def get_user_by_hash(self, hash: str) -> Optional[Any]:
        select: str = """SELECT * FROM users WHERE hash=%s"""
        return self._fetchone(select, (hash,))

    def update_last_login(self, id: str) -> Optional[Any]:
        update: str = """
            UPDATE users
            SET last_login=NOW() at time zone 'utc'
            WHERE id=%s
        """
        return self._updateone(update, (id,))

    def update_user(self, id: str, **kwargs: Any) -> Optional[Any]:
        update: str = """
            UPDATE users
            SET
        """
        if kwargs.get('name', None) is not None:
            update += 'name=%(name)s, '
        if kwargs.get('login', None) is not None:
            update += 'login=%(login)s, '
        if kwargs.get('password', None) is not None:
            update += 'password=%(password)s, '
        if kwargs.get('email', None) is not None:
            update += 'email=%(email)s, '
        if kwargs.get('status', None) is not None:
            update += 'status=%(status)s, '
        if kwargs.get('roles', None) is not None:
            update += 'roles=%(roles)s, '
        if kwargs.get('attributes', None) is not None:
            update += 'attributes=attributes || %(attributes)s, '
        if kwargs.get('text', None) is not None:
            update += 'text=%(text)s, '
        if kwargs.get('email_verified', None) is not None:
            update += 'email_verified=%(email_verified)s, '
        update += """
            update_time=NOW() at time zone 'utc'
            WHERE id=%(id)s
            RETURNING *
        """
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def update_user_attributes(
        self,
        id: str,
        old_attrs: Dict[str, Any],
        new_attrs: Dict[str, Any]
    ) -> bool:
        from alerta.utils.collections import merge
        merge(old_attrs, new_attrs)
        attrs: Dict[str, Any] = {k: v for k, v in old_attrs.items() if v is not None}
        update: str = """
            UPDATE users
               SET attributes=%(attrs)s, update_time=NOW() at time zone 'utc'
             WHERE id=%(id)s
            RETURNING id
        """
        return bool(self._updateone(update, {'id': id, 'attrs': attrs}, returning=True))

    def delete_user(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM users
            WHERE id=%s
            RETURNING id
        """
        return self._deleteone(delete, (id,), returning=True)

    def set_email_hash(self, id: str, hash: str) -> Optional[Any]:
        update: str = """
            UPDATE users
            SET hash=%s, update_time=NOW() at time zone 'utc'
            WHERE id=%s
        """
        return self._updateone(update, (hash, id))

    # GROUPS

    def create_group(self, group: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO groups (id, name, text)
            VALUES (%(id)s, %(name)s, %(text)s)
            RETURNING *, 0 AS count
        """
        return self._insert(insert, vars(group))

    def get_group(self, id: str) -> Optional[Any]:
        select: str = """SELECT *, COALESCE(CARDINALITY(users), 0) AS count FROM groups WHERE id=%s"""
        return self._fetchone(select, (id,))

    def get_groups(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = """
            SELECT *, COALESCE(CARDINALITY(users), 0) AS count FROM groups
             WHERE {where}
          ORDER BY {order}
        """.format(where=query.where, order=query.sort)
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_groups_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM groups
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def get_group_users(self, id: str) -> List[Any]:
        select: str = """
            SELECT u.id, u.login, u.email, u.name, u.status
              FROM (SELECT id, UNNEST(users) as uid FROM groups) g
            INNER JOIN users u on g.uid = u.id
            WHERE g.id = %s
        """
        return self._fetchall(select, (id,))

    def update_group(self, id: str, **kwargs: Any) -> Optional[Any]:
        update: str = """
            UPDATE groups
            SET
        """
        if kwargs.get('name', None) is not None:
            update += 'name=%(name)s, '
        if kwargs.get('text', None) is not None:
            update += 'text=%(text)s, '
        update += """
            update_time=NOW() at time zone 'utc'
            WHERE id=%(id)s
            RETURNING *, COALESCE(CARDINALITY(users), 0) AS count
        """
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def add_user_to_group(self, group: str, user: str) -> Optional[Any]:
        update: str = """
            UPDATE groups
            SET users=ARRAY(SELECT DISTINCT UNNEST(users || %(users)s))
            WHERE id=%(id)s
            RETURNING *
        """
        return self._updateone(update, {'id': group, 'users': [user]}, returning=True)

    def remove_user_from_group(self, group: str, user: str) -> Optional[Any]:
        update: str = """
            UPDATE groups
            SET users=(select array_agg(u) FROM unnest(users) AS u WHERE NOT u=%(user)s )
            WHERE id=%(id)s
            RETURNING *
        """
        return self._updateone(update, {'id': group, 'user': user}, returning=True)

    def delete_group(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM groups
            WHERE id=%s
            RETURNING id
        """
        return self._deleteone(delete, (id,), returning=True)

    def get_groups_by_user(self, user: str) -> List[Any]:
        select: str = """
            SELECT *, COALESCE(CARDINALITY(users), 0) AS count
              FROM groups
            WHERE %s=ANY(users)
        """
        return self._fetchall(select, (user,))

    # PERMISSIONS

    def create_perm(self, perm: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO perms (id, match, scopes)
            VALUES (%(id)s, %(match)s, %(scopes)s)
            RETURNING *
        """
        return self._insert(insert, vars(perm))

    def get_perm(self, id: str) -> Optional[Any]:
        select: str = """SELECT * FROM perms WHERE id=%s"""
        return self._fetchone(select, (id,))

    def get_perms(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = f"""
            SELECT * FROM perms
             WHERE {query.where}
          ORDER BY {query.sort}
        """
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_perms_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM perms
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def update_perm(self, id: str, **kwargs: Any) -> Optional[Any]:
        update: str = """
            UPDATE perms
            SET
        """
        if 'match' in kwargs:
            update += 'match=%(match)s, '
        if 'scopes' in kwargs:
            update += 'scopes=%(scopes)s, '
        update += """
            id=%(id)s
            WHERE id=%(id)s
            RETURNING *
        """
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def delete_perm(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM perms
            WHERE id=%s
            RETURNING id
        """
        return self._deleteone(delete, (id,), returning=True)

    def get_scopes_by_match(self, login: str, matches: List[str]) -> List[str]:
        if login in current_app.config['ADMIN_USERS']:
            return ADMIN_SCOPES

        scopes: List[str] = []
        for match in matches:
            if match in current_app.config['ADMIN_ROLES']:
                return ADMIN_SCOPES
            if match in current_app.config['USER_ROLES']:
                scopes.extend(current_app.config['USER_DEFAULT_SCOPES'])
            if match in current_app.config['GUEST_ROLES']:
                scopes.extend(current_app.config['GUEST_DEFAULT_SCOPES'])
            select: str = """SELECT scopes FROM perms WHERE match=%s"""
            response: Optional[Any] = self._fetchone(select, (match,))
            if response:
                scopes.extend(response.scopes)
        return sorted(set(scopes))

    # CUSTOMERS

    def create_customer(self, customer: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO customers (id, match, customer)
            VALUES (%(id)s, %(match)s, %(customer)s)
            RETURNING *
        """
        return self._insert(insert, vars(customer))

    def get_customer(self, id: str) -> Optional[Any]:
        select: str = """SELECT * FROM customers WHERE id=%s"""
        return self._fetchone(select, (id,))

    def get_customers(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = f"""
            SELECT * FROM customers
             WHERE {query.where}
          ORDER BY {query.sort}
        """
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_customers_count(self, query: Optional[Query] = None) -> int:
        query = query or Query()
        select: str = f"""
            SELECT COUNT(1) FROM customers
             WHERE {query.where}
        """
        return self._fetchone(select, query.vars).count

    def update_customer(self, id: str, **kwargs: Any) -> Optional[Any]:
        update: str = """
            UPDATE customers
            SET
        """
        if 'match' in kwargs:
            update += 'match=%(match)s, '
        if 'customer' in kwargs:
            update += 'customer=%(customer)s, '
        update += """
            id=%(id)s
            WHERE id=%(id)s
            RETURNING *
        """
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def delete_customer(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM customers
            WHERE id=%s
            RETURNING id
        """
        return self._deleteone(delete, (id,), returning=True)

    def get_customers_by_match(self, login: str, matches: List[str]) -> Union[List[str], str]:
        if login in current_app.config['ADMIN_USERS']:
            return '*'  # all customers

        customers: List[str] = []
        for match in [login] + matches:
            select: str = """SELECT customer FROM customers WHERE match=%s"""
            response: List[Any] = self._fetchall(select, (match,))
            if response:
                customers.extend([r.customer for r in response])

        if customers:
            if '*' in customers:
                return '*'  # all customers
            return customers

        raise NoCustomerMatch(f"No customer lookup configured for user '{login}' or '{','.join(matches)}'")

    # NOTES

    def create_note(self, note: Any) -> Optional[Any]:
        insert: str = """
            INSERT INTO notes (id, text, "user", attributes, type,
                create_time, update_time, alert, customer)
            VALUES (%(id)s, %(text)s, %(user)s, %(attributes)s, %(note_type)s,
                %(create_time)s, %(update_time)s, %(alert)s, %(customer)s)
            RETURNING *
        """
        return self._insert(insert, vars(note))

    def get_note(self, id: str) -> Optional[Any]:
        select: str = """
            SELECT * FROM notes
            WHERE id=%s
        """
        return self._fetchone(select, (id,))

    def get_notes(
        self,
        query: Optional[Query] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        query = query or Query()
        select: str = f"""
            SELECT * FROM notes
             WHERE {query.where}
          ORDER BY {query.sort or 'create_time'}
        """
        return self._fetchall(select, query.vars, limit=page_size, offset=(page - 1) * page_size)

    def get_alert_notes(self, id: str, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        select: str = """
            SELECT * FROM notes
             WHERE alert ~* (%s)
        """
        return self._fetchall(select, (id,), limit=page_size, offset=(page - 1) * page_size)

    def get_customer_notes(
        self,
        customer: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None
    ) -> List[Any]:
        select: str = """
            SELECT * FROM notes
             WHERE customer=%s
        """
        return self._fetchall(select, (customer,), limit=page_size, offset=(page - 1) * page_size)

    def update_note(self, id: str, **kwargs: Any) -> Optional[Any]:
        update: str = """
            UPDATE notes
            SET
        """
        if kwargs.get('text', None) is not None:
            update += 'text=%(text)s, '
        if kwargs.get('attributes', None) is not None:
            update += 'attributes=attributes || %(attributes)s, '
        update += """
            "user"=COALESCE(%(user)s, "user"),
            update_time=NOW() at time zone 'utc'
            WHERE id=%(id)s
            RETURNING *
        """
        kwargs['id'] = id
        kwargs['user'] = kwargs.get('user')
        return self._updateone(update, kwargs, returning=True)

    def delete_note(self, id: str) -> Optional[Any]:
        delete: str = """
            DELETE FROM notes
            WHERE id=%s
            RETURNING id
        """
        return self._deleteone(delete, (id,), returning=True)

    # METRICS

    def get_metrics(self, type: Optional[str] = None) -> List[Any]:
        select: str = """SELECT * FROM metrics"""
        if type:
            select += ' WHERE type=%s'
        return self._fetchall(select, (type,) if type else ())

    def set_gauge(self, gauge: Any) -> Optional[Any]:
        upsert: str = """
            INSERT INTO metrics ("group", name, title, description, value, type)
            VALUES (%(group)s, %(name)s, %(title)s, %(description)s, %(value)s, %(type)s)
            ON CONFLICT ("group", name, type) DO UPDATE
                SET value=%(value)s
            RETURNING *
        """
        return self._upsert(upsert, vars(gauge))

    def inc_counter(self, counter: Any) -> Optional[Any]:
        upsert: str = """
            INSERT INTO metrics ("group", name, title, description, count, type)
            VALUES (%(group)s, %(name)s, %(title)s, %(description)s, %(count)s, %(type)s)
            ON CONFLICT ("group", name, type) DO UPDATE
                SET count=metrics.count + %(count)s
            RETURNING *
        """
        return self._upsert(upsert, vars(counter))

    def update_timer(self, timer: Any) -> Optional[Any]:
        upsert: str = """
            INSERT INTO metrics ("group", name, title, description, count, total_time, type)
            VALUES (%(group)s, %(name)s, %(title)s, %(description)s, %(count)s, %(total_time)s, %(type)s)
            ON CONFLICT ("group", name, type) DO UPDATE
                SET count=metrics.count + %(count)s, total_time=metrics.total_time + %(total_time)s
            RETURNING *
        """
        return self._upsert(upsert, vars(timer))

    # HOUSEKEEPING

    def get_expired(
        self,
        expired_threshold: Optional[int],
        info_threshold: Optional[int]
    ) -> List[Any]:
        # delete 'closed' or 'expired' alerts older than "expired_threshold" seconds
        # and 'informational' alerts older than "info_threshold" seconds

        if expired_threshold:
            delete: str = """
                DELETE FROM alerts
                 WHERE (status IN ('closed', 'expired')
                        AND last_receive_time < (NOW() at time zone 'utc' - INTERVAL '%(expired_threshold)s seconds'))
            """
            self._deleteall(delete, {'expired_threshold': expired_threshold})

        if info_threshold:
            delete: str = """
                DELETE FROM alerts
                 WHERE (severity=%(inform_severity)s
                        AND last_receive_time < (NOW() at time zone 'utc' - INTERVAL '%(info_threshold)s seconds'))
            """
            self._deleteall(delete, {'inform_severity': alarm_model.DEFAULT_INFORM_SEVERITY, 'info_threshold': info_threshold})

        # get list of alerts to be newly expired
        select: str = """
            SELECT *
              FROM alerts
             WHERE status NOT IN ('expired') AND COALESCE(timeout, {timeout})!=0
               AND (last_receive_time + INTERVAL '1 second' * timeout) < NOW() at time zone 'utc'
        """.format(timeout=current_app.config['ALERT_TIMEOUT'])

        return self._fetchall(select, {})

    def get_unshelve(self) -> List[Any]:
        # get list of alerts to be unshelved
        select: str = """
            SELECT DISTINCT ON (a.id) a.*
              FROM alerts a, UNNEST(history) h
             WHERE a.status='shelved'
               AND h.type='shelve'
               AND h.status='shelved'
               AND COALESCE(h.timeout, {timeout})!=0
               AND (a.update_time + INTERVAL '1 second' * h.timeout) < NOW() at time zone 'utc'
          ORDER BY a.id, a.update_time DESC
        """.format(timeout=current_app.config['SHELVE_TIMEOUT'])
        return self._fetchall(select, {})

    def get_unack(self) -> List[Any]:
        # get list of alerts to be unack'ed
        select: str = """
            SELECT DISTINCT ON (a.id) a.*
              FROM alerts a, UNNEST(history) h
             WHERE a.status='ack'
               AND h.type='ack'
               AND h.status='ack'
               AND COALESCE(h.timeout, {timeout})!=0
               AND (a.update_time + INTERVAL '1 second' * h.timeout) < NOW() at time zone 'utc'
          ORDER BY a.id, a.update_time DESC
        """.format(timeout=current_app.config['ACK_TIMEOUT'])
        return self._fetchall(select, {})

    # SQL HELPERS

    def _insert(self, query: str, vars: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional[Any]:
        """
        Insert, with return.
        """
        cursor: Any = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        return cursor.fetchone()

    def _fetchone(self, query: str, vars: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional[Any]:
        """
        Return none or one row.
        """
        cursor: Any = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        return cursor.fetchone()

    def _fetchall(
        self,
        query: str,
        vars: Union[Dict[str, Any], Tuple[Any, ...]],
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Any]:
        """
        Return multiple rows.
        """
        if limit is None:
            limit = current_app.config['DEFAULT_PAGE_SIZE']
        query += f' LIMIT {limit} OFFSET {offset}'
        cursor: Any = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        return cursor.fetchall()

    def _updateone(
        self,
        query: str,
        vars: Union[Dict[str, Any], Tuple[Any, ...]],
        returning: bool = False
    ) -> Optional[Any]:
        """
        Update, with optional return.
        """
        cursor: Any = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        if returning:
            return cursor.fetchone()
        return None

    def _updateall(
        self,
        query: str,
        vars: Union[Dict[str, Any], Tuple[Any, ...]],
        returning: bool = False
    ) -> Optional[List[Any]]:
        """
        Update, with optional return.
        """
        cursor: Any = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        if returning:
            return cursor.fetchall()
        return None

    def _upsert(self, query: str, vars: Union[Dict[str, Any], Tuple[Any, ...]]) -> Optional[Any]:
        """
        Insert or update, with return.
        """
        return self._insert(query, vars)

    def _deleteone(
        self,
        query: str,
        vars: Union[Dict[str, Any], Tuple[Any, ...]],
        returning: bool = False
    ) -> Optional[Any]:
        """
        Delete, with optional return.
        """
        cursor: Any = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        if returning:
            return cursor.fetchone()
        return None

    def _deleteall(
        self,
        query: str,
        vars: Union[Dict[str, Any], Tuple[Any, ...]],
        returning: bool = False
    ) -> Optional[List[Any]]:
        """
        Delete multiple rows, with optional return.
        """
        cursor: Any = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        if returning:
            return cursor.fetchall()
        return None

    def _log(self, cursor: Any, query: str, vars: Union[Dict[str, Any], Tuple[Any, ...]]) -> None:
        current_app.logger.debug('{stars}\n{query}\n{stars}'.format(
            stars='*' * 40, query=cursor.mogrify(query, vars).decode('utf-8')))
