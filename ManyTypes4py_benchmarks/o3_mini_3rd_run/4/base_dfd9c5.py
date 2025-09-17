#!/usr/bin/env python3
from __future__ import annotations
import threading
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2.extensions import AsIs, adapt, register_adapter, connection as PgConnection, cursor as PgCursor
from psycopg2.extras import Json, NamedTupleCursor, register_composite
from flask import current_app
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
        self.conn: Optional[PgConnection] = None

    def prepare(self, conn: PgConnection) -> None:
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

Record = namedtuple('Record', ['id', 'resource', 'event', 'environment', 'severity', 'status', 'service', 'group', 
                                 'value', 'text', 'tags', 'attributes', 'origin', 'update_time', 'user', 'timeout', 'type', 'customer'])

class Backend(Database):

    def create_engine(self, app: Any, uri: str, dbname: Optional[str] = None, schema: str = 'public', raise_on_error: bool = True) -> None:
        self.uri = uri
        self.dbname = dbname
        self.schema = schema
        lock: threading.Lock = threading.Lock()
        with lock:
            conn: PgConnection = self.connect()
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

    def connect(self) -> PgConnection:
        retry: int = 0
        conn: Optional[PgConnection] = None
        while True:
            try:
                conn = psycopg2.connect(dsn=self.uri, dbname=self.dbname, cursor_factory=NamedTupleCursor)
                conn.set_client_encoding('UTF8')
                break
            except Exception as e:
                print(e)
                retry += 1
                if retry > MAX_RETRIES:
                    conn = None
                    break
                else:
                    backoff: int = 2 ** retry
                    print(f'Retry attempt {retry}/{MAX_RETRIES} (wait={backoff}s)...')
                    time.sleep(backoff)
        if conn:
            cur: PgCursor = conn.cursor()
            cur.execute('SET search_path TO {}'.format(self.schema))
            conn.commit()
            return conn
        else:
            raise RuntimeError(f'Database connect error. Failed to connect after {MAX_RETRIES} retries.')

    @staticmethod
    def _adapt_datetime(dt: datetime) -> AsIs:
        return AsIs(f'{adapt(DateTime.iso8601(dt))}')

    @property
    def name(self) -> str:
        cursor: PgCursor = self.get_db().cursor()
        cursor.execute('SELECT current_database()')
        return cursor.fetchone()[0]

    @property
    def version(self) -> str:
        cursor: PgCursor = self.get_db().cursor()
        cursor.execute('SHOW server_version')
        return cursor.fetchone()[0]

    @property
    def is_alive(self) -> Any:
        cursor: PgCursor = self.get_db().cursor()
        cursor.execute('SELECT true')
        return cursor.fetchone()

    def close(self, db: PgConnection) -> None:
        db.close()

    def destroy(self) -> None:
        conn: PgConnection = self.connect()
        cursor: PgCursor = conn.cursor()
        for table in ['alerts', 'blackouts', 'customers', 'groups', 'heartbeats', 'keys', 'metrics', 'perms', 'users']:
            cursor.execute(f'DROP TABLE IF EXISTS {table}')
        conn.commit()
        conn.close()

    def get_severity(self, alert: Any) -> Any:
        select: str = (
            '\n            SELECT severity FROM alerts\n             WHERE environment=%(environment)s AND resource=%(resource)s\n               AND ((event=%(event)s AND severity!=%(severity)s)\n                OR (event!=%(event)s AND %(event)s=ANY(correlate)))\n               AND {customer}\n            '
        ).format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).severity

    def get_status(self, alert: Any) -> Any:
        select: str = (
            '\n            SELECT status FROM alerts\n             WHERE environment=%(environment)s AND resource=%(resource)s\n              AND (event=%(event)s OR %(event)s=ANY(correlate))\n              AND {customer}\n            '
        ).format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).status

    def is_duplicate(self, alert: Any) -> Any:
        select: str = (
            '\n            SELECT * FROM alerts\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND event=%(event)s\n               AND severity=%(severity)s\n               AND {customer}\n            '
        ).format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert))

    def is_correlated(self, alert: Any) -> Any:
        select: str = (
            '\n            SELECT * FROM alerts\n             WHERE environment=%(environment)s AND resource=%(resource)s\n               AND ((event=%(event)s AND severity!=%(severity)s)\n                OR (event!=%(event)s AND %(event)s=ANY(correlate)))\n               AND {customer}\n        '
        ).format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert))

    def is_flapping(self, alert: Any, window: int = 1800, count: int = 2) -> bool:
        """
        Return true if alert severity has changed more than X times in Y seconds
        """
        select: str = (
            "\n            SELECT COUNT(*)\n              FROM alerts, unnest(history) h\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND h.event=%(event)s\n               AND h.update_time > (NOW() at time zone 'utc' - INTERVAL '{window} seconds')\n               AND h.type='severity'\n               AND {customer}\n        "
        ).format(window=window, customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._fetchone(select, vars(alert)).count > count

    def dedup_alert(self, alert: Any, history: Any) -> Any:
        alert.history = history
        update: str = (
            '\n            UPDATE alerts\n               SET status=%(status)s, service=%(service)s, value=%(value)s, text=%(text)s,\n                   timeout=%(timeout)s, raw_data=%(raw_data)s, repeat=%(repeat)s,\n                   last_receive_id=%(last_receive_id)s, last_receive_time=%(last_receive_time)s,\n                   tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)), attributes=attributes || %(attributes)s,\n                   duplicate_count=duplicate_count + 1, {update_time}, history=(%(history)s || history)[1:{limit}]\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND event=%(event)s\n               AND severity=%(severity)s\n               AND {customer}\n         RETURNING *\n        '
        ).format(limit=current_app.config['HISTORY_LIMIT'],
                 update_time='update_time=%(update_time)s' if alert.update_time else 'update_time=update_time',
                 customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._updateone(update, vars(alert), returning=True)

    def correlate_alert(self, alert: Any, history: Any) -> Any:
        alert.history = history
        update: str = (
            '\n            UPDATE alerts\n               SET event=%(event)s, severity=%(severity)s, status=%(status)s, service=%(service)s, value=%(value)s,\n                   text=%(text)s, create_time=%(create_time)s, timeout=%(timeout)s, raw_data=%(raw_data)s,\n                   duplicate_count=%(duplicate_count)s, repeat=%(repeat)s, previous_severity=%(previous_severity)s,\n                   trend_indication=%(trend_indication)s, receive_time=%(receive_time)s, last_receive_id=%(last_receive_id)s,\n                   last_receive_time=%(last_receive_time)s, tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)),\n                   attributes=attributes || %(attributes)s, {update_time}, history=(%(history)s || history)[1:{limit}]\n             WHERE environment=%(environment)s\n               AND resource=%(resource)s\n               AND ((event=%(event)s AND severity!=%(severity)s) OR (event!=%(event)s AND %(event)s=ANY(correlate)))\n               AND {customer}\n         RETURNING *\n        '
        ).format(limit=current_app.config['HISTORY_LIMIT'],
                 update_time='update_time=%(update_time)s' if alert.update_time else 'update_time=update_time',
                 customer='customer=%(customer)s' if alert.customer else 'customer IS NULL')
        return self._updateone(update, vars(alert), returning=True)

    def create_alert(self, alert: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO alerts (id, resource, event, environment, severity, correlate, status, service, "group",\n                value, text, tags, attributes, origin, type, create_time, timeout, raw_data, customer,\n                duplicate_count, repeat, previous_severity, trend_indication, receive_time, last_receive_id,\n                last_receive_time, update_time, history)\n            VALUES (%(id)s, %(resource)s, %(event)s, %(environment)s, %(severity)s, %(correlate)s, %(status)s,\n                %(service)s, %(group)s, %(value)s, %(text)s, %(tags)s, %(attributes)s, %(origin)s,\n                %(event_type)s, %(create_time)s, %(timeout)s, %(raw_data)s, %(customer)s, %(duplicate_count)s,\n                %(repeat)s, %(previous_severity)s, %(trend_indication)s, %(receive_time)s, %(last_receive_id)s,\n                %(last_receive_time)s, %(update_time)s, %(history)s::history[])\n            RETURNING *\n        '
        return self._insert(insert, vars(alert))

    def set_alert(self, id: str, severity: Any, status: Any, tags: List[Any], attributes: Dict[str, Any], timeout: Any, previous_severity: Any, update_time: Any, history: Optional[Any] = None) -> Any:
        update: str = (
            '\n            UPDATE alerts\n               SET severity=%(severity)s, status=%(status)s, tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s)),\n                   attributes=%(attributes)s, timeout=%(timeout)s, previous_severity=%(previous_severity)s,\n                   update_time=%(update_time)s, history=(%(change)s || history)[1:{limit}]\n             WHERE id=%(id)s OR id LIKE %(like_id)s\n         RETURNING *\n        '
        ).format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'severity': severity, 'status': status, 'tags': tags, 'attributes': attributes, 'timeout': timeout, 'previous_severity': previous_severity, 'update_time': update_time, 'change': history}, returning=True)

    def get_alert(self, id: str, customers: Optional[List[str]] = None) -> Any:
        select: str = (
            '\n            SELECT * FROM alerts\n             WHERE (id ~* (%(id)s) OR last_receive_id ~* (%(id)s))\n               AND {customer}\n        '
        ).format(customer='customer=ANY(%(customers)s)' if customers else '1=1')
        return self._fetchone(select, {'id': '^' + id, 'customers': customers})

    def set_status(self, id: str, status: Any, timeout: Any, update_time: Any, history: Optional[Any] = None) -> Any:
        update: str = (
            '\n            UPDATE alerts\n            SET status=%(status)s, timeout=%(timeout)s, update_time=%(update_time)s, history=(%(change)s || history)[1:{limit}]\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '
        ).format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'status': status, 'timeout': timeout, 'update_time': update_time, 'change': history}, returning=True)

    def tag_alert(self, id: str, tags: List[Any]) -> Any:
        update: str = (
            '\n            UPDATE alerts\n            SET tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(tags)s))\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '
        )
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def untag_alert(self, id: str, tags: List[Any]) -> Any:
        update: str = (
            '\n            UPDATE alerts\n            SET tags=(select array_agg(t) FROM unnest(tags) AS t WHERE NOT t=ANY(%(tags)s) )\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '
        )
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def update_tags(self, id: str, tags: List[Any]) -> Any:
        update: str = (
            '\n            UPDATE alerts\n            SET tags=%(tags)s\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING *\n        '
        )
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'tags': tags}, returning=True)

    def update_attributes(self, id: str, old_attrs: Dict[str, Any], new_attrs: Dict[str, Any]) -> Any:
        old_attrs.update(new_attrs)
        attrs: Dict[str, Any] = {k: v for k, v in old_attrs.items() if v is not None}
        update: str = (
            '\n            UPDATE alerts\n            SET attributes=%(attrs)s\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING attributes\n        '
        )
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'attrs': attrs}, returning=True).attributes

    def delete_alert(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM alerts\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, {'id': id, 'like_id': id + '%'}, returning=True)

    def tag_alerts(self, query: Optional[Query] = None, tags: Optional[List[Any]] = None) -> List[Any]:
        query = query or Query()
        update: str = f'\n            UPDATE alerts\n            SET tags=ARRAY(SELECT DISTINCT UNNEST(tags || %(_tags)s))\n            WHERE {query.where}\n            RETURNING id\n        '
        rows: List[Any] = self._updateall(update, {**query.vars, **{'_tags': tags}}, returning=True)
        return [row[0] for row in rows]

    def untag_alerts(self, query: Optional[Query] = None, tags: Optional[List[Any]] = None) -> List[Any]:
        query = query or Query()
        update: str = (
            '\n            UPDATE alerts\n            SET tags=(select array_agg(t) FROM unnest(tags) AS t WHERE NOT t=ANY(%(_tags)s) )\n            WHERE {where}\n            RETURNING id\n        '.format(where=query.where)
        )
        rows: List[Any] = self._updateall(update, {**query.vars, **{'_tags': tags}}, returning=True)
        return [row[0] for row in rows]

    def update_attributes_by_query(self, query: Optional[Query] = None, attributes: Optional[Dict[str, Any]] = None) -> List[Any]:
        update: str = f'\n            UPDATE alerts\n            SET attributes=attributes || %(_attributes)s\n            WHERE {query.where}\n            RETURNING id\n        '
        rows: List[Any] = self._updateall(update, {**query.vars, **{'_attributes': attributes}}, returning=True)
        return [row[0] for row in rows]

    def delete_alerts(self, query: Optional[Query] = None) -> List[Any]:
        query = query or Query()
        delete: str = f'\n            DELETE FROM alerts\n            WHERE {query.where}\n            RETURNING id\n        '
        rows: List[Any] = self._deleteall(delete, query.vars, returning=True)
        return [row[0] for row in rows]

    def add_history(self, id: str, history: Any) -> Any:
        update: str = (
            '\n            UPDATE alerts\n               SET history=(%(history)s || history)[1:{limit}]\n             WHERE id=%(id)s OR id LIKE %(like_id)s\n         RETURNING *\n        '
        ).format(limit=current_app.config['HISTORY_LIMIT'])
        return self._updateone(update, {'id': id, 'like_id': id + '%', 'history': history}, returning=True)

    def get_alerts(self, query: Optional[Query] = None, raw_data: bool = False, history: bool = False, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        if raw_data and history:
            select_fields: str = '*'
        else:
            select_fields = ('id, resource, event, environment, severity, correlate, status, service, "group", value, "text",'
                             + 'tags, attributes, origin, type, create_time, timeout, {raw_data}, customer, duplicate_count, repeat,'
                             + 'previous_severity, trend_indication, receive_time, last_receive_id, last_receive_time, update_time,'
                             + '{history}').format(raw_data='raw_data' if raw_data else 'NULL as raw_data', history='history' if history else 'array[]::history[] as history')
        join: str = ''
        if 's.code' in query.sort:
            join += 'JOIN (VALUES {}) AS s(sev, code) ON alerts.severity = s.sev '.format(', '.join((f"('{k}', {v})" for k, v in alarm_model.Severity.items())))
        if 'st.state' in query.sort:
            join += 'JOIN (VALUES {}) AS st(sts, state) ON alerts.status = st.sts '.format(', '.join((f"('{k}', '{v}')" for k, v in alarm_model.Status.items())))
        select: str = f'\n            SELECT {select_fields}\n              FROM alerts {join}\n             WHERE {query.where}\n          ORDER BY {query.sort or "last_receive_time"}\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_alert_history(self, alert: Any, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Record]:
        select: str = (
            '\n            SELECT resource, environment, service, "group", tags, attributes, origin, customer, h.*\n              FROM alerts, unnest(history[1:{limit}]) h\n             WHERE environment=%(environment)s AND resource=%(resource)s\n               AND (h.event=%(event)s OR %(event)s=ANY(correlate))\n               AND {customer}\n          ORDER BY update_time DESC\n            '
        ).format(customer='customer=%(customer)s' if alert.customer else 'customer IS NULL', limit=current_app.config['HISTORY_LIMIT'])
        rows: List[Any] = self._fetchall(select, vars(alert), limit=page_size, offset=((page - 1) * page_size) if (page and page_size) else 0)
        return [
            Record(
                id=h.id, resource=h.resource, event=h.event, environment=h.environment, severity=h.severity, status=h.status,
                service=h.service, group=h.group, value=h.value, text=h.text, tags=h.tags, attributes=h.attributes,
                origin=h.origin, update_time=h.update_time, user=getattr(h, 'user', None), timeout=getattr(h, 'timeout', None),
                type=h.type, customer=h.customer
            ) for h in rows
        ]

    def get_history(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Record]:
        query = query or Query()
        if 'id' in query.vars:
            select: str = (
                '\n                SELECT a.id\n                  FROM alerts a, unnest(history[1:{limit}]) h\n                 WHERE h.id LIKE %(id)s\n            '
            ).format(limit=current_app.config['HISTORY_LIMIT'])
            query.vars['id'] = self._fetchone(select, query.vars)
        select: str = (
            '\n            SELECT resource, environment, service, "group", tags, attributes, origin, customer, history, h.*\n              FROM alerts, unnest(history[1:{limit}]) h\n             WHERE {where}\n          ORDER BY update_time DESC\n        '
        ).format(where=query.where, limit=current_app.config['HISTORY_LIMIT'])
        rows: List[Any] = self._fetchall(select, query.vars, limit=page_size, offset=((page - 1) * page_size) if (page and page_size) else 0)
        return [
            Record(
                id=h.id, resource=h.resource, event=h.event, environment=h.environment, severity=h.severity, status=h.status,
                service=h.service, group=h.group, value=h.value, text=h.text, tags=h.tags, attributes=h.attributes,
                origin=h.origin, update_time=h.update_time, user=getattr(h, 'user', None), timeout=getattr(h, 'timeout', None),
                type=h.type, customer=h.customer
            ) for h in rows
        ]

    def get_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM alerts\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def get_counts(self, query: Optional[Query] = None, group: Optional[str] = None) -> Dict[Any, Any]:
        query = query or Query()
        if group is None:
            raise ValueError('Must define a group')
        select: str = (
            '\n            SELECT {group}, COUNT(*) FROM alerts\n             WHERE {where}\n            GROUP BY {group}\n        '
        ).format(where=query.where, group=group)
        rows: List[Any] = self._fetchall(select, query.vars)
        return {s['group']: s.count for s in rows}

    def get_counts_by_severity(self, query: Optional[Query] = None) -> Dict[Any, Any]:
        query = query or Query()
        select: str = f'\n            SELECT severity, COUNT(*) FROM alerts\n             WHERE {query.where}\n            GROUP BY severity\n        '
        rows: List[Any] = self._fetchall(select, query.vars)
        return {s.severity: s.count for s in rows}

    def get_counts_by_status(self, query: Optional[Query] = None) -> Dict[Any, Any]:
        query = query or Query()
        select: str = f'\n            SELECT status, COUNT(*) FROM alerts\n            WHERE {query.where}\n            GROUP BY status\n        '
        rows: List[Any] = self._fetchall(select, query.vars)
        return {s.status: s.count for s in rows}

    def get_topn_count(self, query: Optional[Query] = None, topn: int = 100) -> List[Dict[str, Any]]:
        query = query or Query()
        group: str = 'event'
        if query and query.group:
            group = query.group[0]
        select: str = (
            '\n            SELECT {group}, COUNT(1) as count, SUM(duplicate_count) AS duplicate_count,\n                   array_agg(DISTINCT environment) AS environments, array_agg(DISTINCT svc) AS services,\n                   array_agg(DISTINCT ARRAY[id, resource]) AS resources\n              FROM alerts, UNNEST (service) svc\n             WHERE {where}\n          GROUP BY {group}\n          ORDER BY count DESC\n        '
        ).format(where=query.where, group=group)
        rows: List[Any] = self._fetchall(select, query.vars, limit=topn)
        result: List[Dict[str, Any]] = []
        for t in rows:
            result.append({
                'count': t.count,
                'duplicateCount': t.duplicate_count,
                'environments': t.environments,
                'services': t.services,
                group: getattr(t, group),
                'resources': [{'id': r[0], 'resource': r[1], 'href': absolute_url(f'/alert/{r[0]}')} for r in t.resources]
            })
        return result

    def get_topn_flapping(self, query: Optional[Query] = None, topn: int = 100) -> List[Dict[str, Any]]:
        query = query or Query()
        group: str = 'event'
        if query and query.group:
            group = query.group[0]
        select: str = (
            "\n            WITH topn AS (SELECT * FROM alerts WHERE {where})\n            SELECT topn.{group}, COUNT(1) as count, SUM(duplicate_count) AS duplicate_count,\n                   array_agg(DISTINCT environment) AS environments, array_agg(DISTINCT svc) AS services,\n                   array_agg(DISTINCT ARRAY[topn.id, resource]) AS resources\n              FROM topn, UNNEST (service) svc, UNNEST (history) hist\n             WHERE hist.type='severity'\n          GROUP BY topn.{group}\n          ORDER BY count DESC\n        "
        ).format(where=query.where, group=group)
        rows: List[Any] = self._fetchall(select, query.vars, limit=topn)
        result: List[Dict[str, Any]] = []
        for t in rows:
            result.append({
                'count': t.count,
                'duplicateCount': t.duplicate_count,
                'environments': t.environments,
                'services': t.services,
                group: getattr(t, group),
                'resources': [{'id': r[0], 'resource': r[1], 'href': absolute_url(f'/alert/{r[0]}')} for r in t.resources]
            })
        return result

    def get_topn_standing(self, query: Optional[Query] = None, topn: int = 100) -> List[Dict[str, Any]]:
        query = query or Query()
        group: str = 'event'
        if query and query.group:
            group = query.group[0]
        select: str = (
            "\n            WITH topn AS (SELECT * FROM alerts WHERE {where})\n            SELECT topn.{group}, COUNT(1) as count, SUM(duplicate_count) AS duplicate_count,\n                   SUM(last_receive_time - create_time) as life_time,\n                   array_agg(DISTINCT environment) AS environments, array_agg(DISTINCT svc) AS services,\n                   array_agg(DISTINCT ARRAY[topn.id, resource]) AS resources\n              FROM topn, UNNEST (service) svc, UNNEST (history) hist\n             WHERE hist.type='severity'\n          GROUP BY topn.{group}\n          ORDER BY life_time DESC\n        "
        ).format(where=query.where, group=group)
        rows: List[Any] = self._fetchall(select, query.vars, limit=topn)
        result: List[Dict[str, Any]] = []
        for t in rows:
            result.append({
                'count': t.count,
                'duplicateCount': t.duplicate_count,
                'environments': t.environments,
                'services': t.services,
                group: getattr(t, group),
                'resources': [{'id': r[0], 'resource': r[1], 'href': absolute_url(f'/alert/{r[0]}')} for r in t.resources]
            })
        return result

    def get_environments(self, query: Optional[Query] = None, topn: int = 1000) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = f'\n            SELECT environment, severity, status, count(1) FROM alerts\n            WHERE {query.where}\n            GROUP BY environment, CUBE(severity, status)\n        '
        result_rows: List[Any] = self._fetchall(select, query.vars, limit=topn)
        severity_count: Dict[Any, List[Tuple[Any, Any]]] = defaultdict(list)
        status_count: Dict[Any, List[Tuple[Any, Any]]] = defaultdict(list)
        total_count: Dict[Any, int] = defaultdict(int)
        for row in result_rows:
            if row.severity and (not row.status):
                severity_count[row.environment].append((row.severity, row.count))
            if not row.severity and row.status:
                status_count[row.environment].append((row.status, row.count))
            if not row.severity and (not row.status):
                total_count[row.environment] = row.count
        select = 'SELECT DISTINCT environment FROM alerts'
        environments: List[Any] = self._fetchall(select, {})
        return [{'environment': e.environment, 'severityCounts': dict(severity_count[e.environment]), 'statusCounts': dict(status_count[e.environment]), 'count': total_count[e.environment]} for e in environments]

    def get_services(self, query: Optional[Query] = None, topn: int = 1000) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = '\n            SELECT environment, svc, severity, status, count(1) FROM alerts, UNNEST(service) svc\n            WHERE {where}\n            GROUP BY environment, svc, CUBE(severity, status)\n        '.format(where=query.where)
        result_rows: List[Any] = self._fetchall(select, query.vars, limit=topn)
        severity_count: Dict[Tuple[Any, Any], List[Tuple[Any, Any]]] = defaultdict(list)
        status_count: Dict[Tuple[Any, Any], List[Tuple[Any, Any]]] = defaultdict(list)
        total_count: Dict[Tuple[Any, Any], int] = defaultdict(int)
        for row in result_rows:
            if row.severity and (not row.status):
                severity_count[(row.environment, row.svc)].append((row.severity, row.count))
            if not row.severity and row.status:
                status_count[(row.environment, row.svc)].append((row.status, row.count))
            if not row.severity and (not row.status):
                total_count[(row.environment, row.svc)] = row.count
        select = 'SELECT DISTINCT environment, svc FROM alerts, UNNEST(service) svc'
        services: List[Any] = self._fetchall(select, {})
        return [{'environment': s.environment, 'service': s.svc, 'severityCounts': dict(severity_count[(s.environment, s.svc)]), 'statusCounts': dict(status_count[(s.environment, s.svc)]), 'count': total_count[(s.environment, s.svc)]} for s in services]

    def get_alert_groups(self, query: Optional[Query] = None, topn: int = 1000) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = f'\n            SELECT environment, "group", count(1) FROM alerts\n            WHERE {query.where}\n            GROUP BY environment, "group"\n        '
        rows: List[Any] = self._fetchall(select, query.vars, limit=topn)
        return [{'environment': g.environment, 'group': g.group, 'count': g.count} for g in rows]

    def get_alert_tags(self, query: Optional[Query] = None, topn: int = 1000) -> List[Dict[str, Any]]:
        query = query or Query()
        select: str = '\n            SELECT environment, tag, count(1) FROM alerts, UNNEST(tags) tag\n            WHERE {where}\n            GROUP BY environment, tag\n        '.format(where=query.where)
        rows: List[Any] = self._fetchall(select, query.vars, limit=topn)
        return [{'environment': t.environment, 'tag': t.tag, 'count': t.count} for t in rows]

    def create_blackout(self, blackout: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO blackouts (id, priority, environment, service, resource, event,\n                "group", tags, origin, customer, start_time, end_time,\n                duration, "user", create_time, text)\n            VALUES (%(id)s, %(priority)s, %(environment)s, %(service)s, %(resource)s, %(event)s,\n                %(group)s, %(tags)s, %(origin)s, %(customer)s, %(start_time)s, %(end_time)s,\n                %(duration)s, %(user)s, %(create_time)s, %(text)s)\n            RETURNING *, duration AS remaining\n        '
        )
        return self._insert(insert, vars(blackout))

    def get_blackout(self, id: str, customers: Optional[List[str]] = None) -> Any:
        select: str = (
            "\n            SELECT *, GREATEST(EXTRACT(EPOCH FROM (end_time - GREATEST(start_time, NOW() at time zone 'utc'))), 0) AS remaining\n            FROM blackouts\n            WHERE id=%(id)s\n              AND {customer}\n        "
        ).format(customer='customer=ANY(%(customers)s)' if customers else '1=1')
        return self._fetchone(select, {'id': id, 'customers': customers})

    def get_blackouts(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = (
            "\n            SELECT *, GREATEST(EXTRACT(EPOCH FROM (end_time - GREATEST(start_time, NOW() at time zone 'utc'))), 0) AS remaining\n              FROM blackouts\n             WHERE {where}\n          ORDER BY {order}\n        "
        ).format(where=query.where, order=query.sort)
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_blackouts_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM blackouts\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def is_blackout_period(self, alert: Any) -> bool:
        select: str = (
            '\n            SELECT *\n            FROM blackouts\n            WHERE start_time <= %(create_time)s AND end_time > %(create_time)s\n              AND environment=%(environment)s\n              AND (\n                 ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group" IS NULL AND tags=\'{}\' AND origin IS NULL )\n              OR ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group" IS NULL AND tags=\'{}\' AND origin=%(origin)s )\n              OR ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )\n              OR ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )\n              OR ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group"=%(group)s AND tags=\'{}\' AND origin IS NULL )\n              OR ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group"=%(group)s AND tags=\'{}\' AND origin=%(origin)s )\n              OR ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group"=%(group)s AND tags <@ %(tags)s AND origin IS NULL )\n              OR ( resource IS NULL AND service=\'{}\' AND event IS NULL AND "group"=%(group)s AND tags <@ %(tags)s AND origin=%(origin)s )\n              OR ( resource IS NULL AND service=\'{}\' AND event=%(event)s AND "group" IS NULL AND tags=\'{}\' AND origin IS NULL )\n              OR ( resource IS NULL AND service=\'{}\' AND event=%(event)s AND "group" IS NULL AND tags=\'{}\' AND origin=%(origin)s )\n              OR ( resource IS NULL AND service=\'{}\' AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )\n              OR ( resource IS NULL AND service=\'{}\' AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )\n              OR ( resource IS NULL AND service=\'{}\' AND event=%(event)s AND "group"=%(group)s AND tags=\'{}\' AND origin IS NULL )\n              OR ( resource IS NULL AND service=\'{}\' AND event=%(event)s AND "group"=%(group)s AND tags=\'{}\' AND origin=%(origin)s )\n              OR ( resource IS NULL AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags=\'{}\' AND origin IS NULL )\n              OR ( resource IS NULL AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags=\'{}\' AND origin=%(origin)s )\n              OR ( resource IS NULL AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )\n              OR ( resource IS NULL AND service <@ %(service)s AND event IS NULL AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags=\'{}\' AND origin IS NULL )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags=\'{}\' AND origin=%(origin)s )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin IS NULL )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group" IS NULL AND tags <@ %(tags)s AND origin=%(origin)s )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags=\'{}\' AND origin IS NULL )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags=\'{}\' AND origin=%(origin)s )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags <@ %(tags)s AND origin IS NULL )\n              OR ( resource IS NULL AND service <@ %(service)s AND event=%(event)s AND "group"=%(group)s AND tags <@ %(tags)s AND origin=%(origin)s )\n                 )\n        '
        if current_app.config['CUSTOMER_VIEWS']:
            select += ' AND (customer IS NULL OR customer=%(customer)s)'
        result: Any = self._fetchone(select, vars(alert))
        if result:
            return True
        return False

    def update_blackout(self, id: str, **kwargs: Any) -> Any:
        update: str = (
            '\n            UPDATE blackouts\n            SET\n        '
        )
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
        update += '\n            "user"=COALESCE(%(user)s, "user")\n            WHERE id=%(id)s\n            RETURNING *, GREATEST(EXTRACT(EPOCH FROM (end_time - GREATEST(start_time, NOW() at time zone \'utc\'))), 0) AS remaining\n        '
        kwargs['id'] = id
        kwargs['user'] = kwargs.get('user')
        return self._updateone(update, kwargs, returning=True)

    def delete_blackout(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM blackouts\n            WHERE id=%s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, (id,), returning=True)

    def upsert_heartbeat(self, heartbeat: Any) -> Any:
        upsert: str = (
            "\n            INSERT INTO heartbeats (id, origin, tags, attributes, type, create_time, timeout, receive_time, customer)\n            VALUES (%(id)s, %(origin)s, %(tags)s, %(attributes)s, %(event_type)s, %(create_time)s, %(timeout)s, %(receive_time)s, %(customer)s)\n            ON CONFLICT (origin, COALESCE(customer, '')) DO UPDATE\n                SET tags=%(tags)s, attributes=%(attributes)s, create_time=%(create_time)s, timeout=%(timeout)s, receive_time=%(receive_time)s\n            RETURNING *,\n                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,\n                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since\n        "
        )
        return self._upsert(upsert, vars(heartbeat))

    def get_heartbeat(self, id: str, customers: Optional[str] = None) -> Any:
        select: str = (
            '\n            SELECT *,\n                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,\n                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since\n              FROM heartbeats\n             WHERE (id=%(id)s OR id LIKE %(like_id)s)\n               AND {customer}\n        '
        ).format(customer='customer=%(customers)s' if customers else '1=1')
        return self._fetchone(select, {'id': id, 'like_id': id + '%', 'customers': customers})

    def get_heartbeats(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = (
            '\n            SELECT *,\n                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,\n                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since\n              FROM heartbeats\n             WHERE {where}\n          ORDER BY {order}\n        '
        ).format(where=query.where, order=query.sort)
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_heartbeats_by_status(self, status: Optional[List[Any]] = None, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        status = status or list()
        query = query or Query()
        swhere: str = ''
        if status:
            q: List[str] = []
            if HeartbeatStatus.OK in status:
                q.append("\n                    (EXTRACT(EPOCH FROM (NOW() at time zone 'utc' - receive_time)) <= timeout\n                    AND EXTRACT(EPOCH FROM (receive_time - create_time)) * 1000 <= {max_latency})\n                    ".format(max_latency=current_app.config['HEARTBEAT_MAX_LATENCY']))
            if HeartbeatStatus.Expired in status:
                q.append("(EXTRACT(EPOCH FROM (NOW() at time zone 'utc' - receive_time)) > timeout)")
            if HeartbeatStatus.Slow in status:
                q.append("\n                    (EXTRACT(EPOCH FROM (NOW() at time zone 'utc' - receive_time)) <= timeout\n                    AND EXTRACT(EPOCH FROM (receive_time - create_time)) * 1000 > {max_latency})\n                    ".format(max_latency=current_app.config['HEARTBEAT_MAX_LATENCY']))
            if q:
                swhere = 'AND (' + ' OR '.join(q) + ')'
        select: str = (
            '\n            SELECT *,\n                   EXTRACT(EPOCH FROM (receive_time - create_time)) AS latency,\n                   EXTRACT(EPOCH FROM (NOW() - receive_time)) AS since\n              FROM heartbeats\n             WHERE {where}\n             {swhere}\n          ORDER BY {order}\n        '
        ).format(where=query.where, swhere=swhere, order=query.sort)
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_heartbeats_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM heartbeats\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def delete_heartbeat(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM heartbeats\n            WHERE id=%(id)s OR id LIKE %(like_id)s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, {'id': id, 'like_id': id + '%'}, returning=True)

    def create_key(self, key: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO keys (id, key, "user", scopes, text, expire_time, "count", last_used_time, customer)\n            VALUES (%(id)s, %(key)s, %(user)s, %(scopes)s, %(text)s, %(expire_time)s, %(count)s, %(last_used_time)s, %(customer)s)\n            RETURNING *\n        '
        )
        return self._insert(insert, vars(key))

    def get_key(self, key: str, user: Optional[Any] = None) -> Any:
        select: str = (
            f'\n            SELECT * FROM keys\n             WHERE (id=%(key)s OR key=%(key)s)\n               AND {("\"user\"=%(user)s" if user else "1=1")}\n        '
        )
        return self._fetchone(select, {'key': key, 'user': user})

    def get_keys(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = f'\n            SELECT * FROM keys\n             WHERE {query.where}\n          ORDER BY {query.sort}\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_keys_by_user(self, user: Any) -> List[Any]:
        select: str = '\n            SELECT * FROM keys\n             WHERE "user"=%s\n        '
        return self._fetchall(select, (user,))

    def get_keys_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM keys\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def update_key(self, key: str, **kwargs: Any) -> Any:
        update: str = (
            '\n            UPDATE keys\n            SET\n        '
        )
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
        update += '\n            id=id\n            WHERE (id=%(key)s OR key=%(key)s)\n            RETURNING *\n        '
        kwargs['key'] = key
        return self._updateone(update, kwargs, returning=True)

    def update_key_last_used(self, key: str) -> Any:
        update: str = "\n            UPDATE keys\n            SET last_used_time=NOW() at time zone 'utc', count=count + 1\n            WHERE id=%s OR key=%s\n        "
        return self._updateone(update, (key, key))

    def delete_key(self, key: str) -> Any:
        delete: str = (
            '\n            DELETE FROM keys\n            WHERE id=%s OR key=%s\n            RETURNING key\n        '
        )
        return self._deleteone(delete, (key, key), returning=True)

    def create_user(self, user: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO users (id, name, login, password, email, status, roles, attributes,\n                create_time, last_login, text, update_time, email_verified)\n            VALUES (%(id)s, %(name)s, %(login)s, %(password)s, %(email)s, %(status)s, %(roles)s, %(attributes)s, %(create_time)s,\n                %(last_login)s, %(text)s, %(update_time)s, %(email_verified)s)\n            RETURNING *\n        '
        )
        return self._insert(insert, vars(user))

    def get_user(self, id: str) -> Any:
        select: str = 'SELECT * FROM users WHERE id=%s'
        return self._fetchone(select, (id,))

    def get_users(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = f'\n            SELECT * FROM users\n             WHERE {query.where}\n          ORDER BY {query.sort}\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_users_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM users\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def get_user_by_username(self, username: str) -> Any:
        select: str = 'SELECT * FROM users WHERE login=%s OR email=%s'
        return self._fetchone(select, (username, username))

    def get_user_by_email(self, email: str) -> Any:
        select: str = 'SELECT * FROM users WHERE email=%s'
        return self._fetchone(select, (email,))

    def get_user_by_hash(self, hash: str) -> Any:
        select: str = 'SELECT * FROM users WHERE hash=%s'
        return self._fetchone(select, (hash,))

    def update_last_login(self, id: str) -> Any:
        update: str = "\n            UPDATE users\n            SET last_login=NOW() at time zone 'utc'\n            WHERE id=%s\n        "
        return self._updateone(update, (id,))

    def update_user(self, id: str, **kwargs: Any) -> Any:
        update: str = '\n            UPDATE users\n            SET\n        '
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
        update += "\n            update_time=NOW() at time zone 'utc'\n            WHERE id=%(id)s\n            RETURNING *\n        "
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def update_user_attributes(self, id: str, old_attrs: Dict[str, Any], new_attrs: Dict[str, Any]) -> bool:
        from alerta.utils.collections import merge
        merge(old_attrs, new_attrs)
        attrs: Dict[str, Any] = {k: v for k, v in old_attrs.items() if v is not None}
        update: str = "\n            UPDATE users\n               SET attributes=%(attrs)s, update_time=NOW() at time zone 'utc'\n             WHERE id=%(id)s\n            RETURNING id\n        "
        return bool(self._updateone(update, {'id': id, 'attrs': attrs}, returning=True))

    def delete_user(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM users\n            WHERE id=%s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, (id,), returning=True)

    def set_email_hash(self, id: str, hash: str) -> Any:
        update: str = "\n            UPDATE users\n            SET hash=%s, update_time=NOW() at time zone 'utc'\n            WHERE id=%s\n        "
        return self._updateone(update, (hash, id))

    def create_group(self, group: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO groups (id, name, text)\n            VALUES (%(id)s, %(name)s, %(text)s)\n            RETURNING *, 0 AS count\n        '
        )
        return self._insert(insert, vars(group))

    def get_group(self, id: str) -> Any:
        select: str = 'SELECT *, COALESCE(CARDINALITY(users), 0) AS count FROM groups WHERE id=%s'
        return self._fetchone(select, (id,))

    def get_groups(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = '\n            SELECT *, COALESCE(CARDINALITY(users), 0) AS count FROM groups\n             WHERE {where}\n          ORDER BY {order}\n        '.format(where=query.where, order=query.sort)
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_groups_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM groups\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def get_group_users(self, id: str) -> List[Any]:
        select: str = (
            '\n            SELECT u.id, u.login, u.email, u.name, u.status\n              FROM (SELECT id, UNNEST(users) as uid FROM groups) g\n            INNER JOIN users u on g.uid = u.id\n            WHERE g.id = %s\n        '
        )
        return self._fetchall(select, (id,))

    def update_group(self, id: str, **kwargs: Any) -> Any:
        update: str = '\n            UPDATE groups\n            SET\n        '
        if kwargs.get('name', None) is not None:
            update += 'name=%(name)s, '
        if kwargs.get('text', None) is not None:
            update += 'text=%(text)s, '
        update += "\n            update_time=NOW() at time zone 'utc'\n            WHERE id=%(id)s\n            RETURNING *, COALESCE(CARDINALITY(users), 0) AS count\n        "
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def add_user_to_group(self, group: str, user: Any) -> Any:
        update: str = (
            '\n            UPDATE groups\n            SET users=ARRAY(SELECT DISTINCT UNNEST(users || %(users)s))\n            WHERE id=%(id)s\n            RETURNING *\n        '
        )
        return self._updateone(update, {'id': group, 'users': [user]}, returning=True)

    def remove_user_from_group(self, group: str, user: Any) -> Any:
        update: str = (
            '\n            UPDATE groups\n            SET users=(select array_agg(u) FROM unnest(users) AS u WHERE NOT u=%(user)s )\n            WHERE id=%(id)s\n            RETURNING *\n        '
        )
        return self._updateone(update, {'id': group, 'user': user}, returning=True)

    def delete_group(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM groups\n            WHERE id=%s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, (id,), returning=True)

    def get_groups_by_user(self, user: Any) -> List[Any]:
        select: str = (
            '\n            SELECT *, COALESCE(CARDINALITY(users), 0) AS count\n              FROM groups\n            WHERE %s=ANY(users)\n        '
        )
        return self._fetchall(select, (user,))

    def create_perm(self, perm: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO perms (id, match, scopes)\n            VALUES (%(id)s, %(match)s, %(scopes)s)\n            RETURNING *\n        '
        )
        return self._insert(insert, vars(perm))

    def get_perm(self, id: str) -> Any:
        select: str = 'SELECT * FROM perms WHERE id=%s'
        return self._fetchone(select, (id,))

    def get_perms(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = f'\n            SELECT * FROM perms\n             WHERE {query.where}\n          ORDER BY {query.sort}\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_perms_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM perms\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def update_perm(self, id: str, **kwargs: Any) -> Any:
        update: str = (
            '\n            UPDATE perms\n            SET\n        '
        )
        if 'match' in kwargs:
            update += 'match=%(match)s, '
        if 'scopes' in kwargs:
            update += 'scopes=%(scopes)s, '
        update += '\n            id=%(id)s\n            WHERE id=%(id)s\n            RETURNING *\n        '
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def delete_perm(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM perms\n            WHERE id=%s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, (id,), returning=True)

    def get_scopes_by_match(self, login: str, matches: List[Any]) -> List[Any]:
        if login in current_app.config['ADMIN_USERS']:
            return ADMIN_SCOPES
        scopes: List[Any] = []
        for match in matches:
            if match in current_app.config['ADMIN_ROLES']:
                return ADMIN_SCOPES
            if match in current_app.config['USER_ROLES']:
                scopes.extend(current_app.config['USER_DEFAULT_SCOPES'])
            if match in current_app.config['GUEST_ROLES']:
                scopes.extend(current_app.config['GUEST_DEFAULT_SCOPES'])
            select: str = 'SELECT scopes FROM perms WHERE match=%s'
            response = self._fetchone(select, (match,))
            if response:
                scopes.extend(response.scopes)
        return sorted(set(scopes))

    def create_customer(self, customer: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO customers (id, match, customer)\n            VALUES (%(id)s, %(match)s, %(customer)s)\n            RETURNING *\n        '
        )
        return self._insert(insert, vars(customer))

    def get_customer(self, id: str) -> Any:
        select: str = 'SELECT * FROM customers WHERE id=%s'
        return self._fetchone(select, (id,))

    def get_customers(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = f'\n            SELECT * FROM customers\n             WHERE {query.where}\n          ORDER BY {query.sort}\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_customers_count(self, query: Optional[Query] = None) -> Any:
        query = query or Query()
        select: str = f'\n            SELECT COUNT(1) FROM customers\n             WHERE {query.where}\n        '
        return self._fetchone(select, query.vars).count

    def update_customer(self, id: str, **kwargs: Any) -> Any:
        update: str = (
            '\n            UPDATE customers\n            SET\n        '
        )
        if 'match' in kwargs:
            update += 'match=%(match)s, '
        if 'customer' in kwargs:
            update += 'customer=%(customer)s, '
        update += '\n            id=%(id)s\n            WHERE id=%(id)s\n            RETURNING *\n        '
        kwargs['id'] = id
        return self._updateone(update, kwargs, returning=True)

    def delete_customer(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM customers\n            WHERE id=%s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, (id,), returning=True)

    def get_customers_by_match(self, login: str, matches: List[Any]) -> Union[str, List[Any]]:
        if login in current_app.config['ADMIN_USERS']:
            return '*'
        customers: List[Any] = []
        for match in [login] + matches:
            select: str = 'SELECT customer FROM customers WHERE match=%s'
            response = self._fetchall(select, (match,))
            if response:
                customers.extend([r.customer for r in response])
        if customers:
            if '*' in customers:
                return '*'
            return customers
        raise NoCustomerMatch(f"No customer lookup configured for user '{login}' or '{','.join(matches)}'")

    def create_note(self, note: Any) -> Any:
        insert: str = (
            '\n            INSERT INTO notes (id, text, "user", attributes, type,\n                create_time, update_time, alert, customer)\n            VALUES (%(id)s, %(text)s, %(user)s, %(attributes)s, %(note_type)s,\n                %(create_time)s, %(update_time)s, %(alert)s, %(customer)s)\n            RETURNING *\n        '
        )
        return self._insert(insert, vars(note))

    def get_note(self, id: str) -> Any:
        select: str = (
            '\n            SELECT * FROM notes\n            WHERE id=%s\n        '
        )
        return self._fetchone(select, (id,))

    def get_notes(self, query: Optional[Query] = None, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        query = query or Query()
        select: str = f'\n            SELECT * FROM notes\n             WHERE {query.where}\n          ORDER BY {query.sort or "create_time"}\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, query.vars, limit=page_size, offset=offset)

    def get_alert_notes(self, id: str, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        select: str = '\n            SELECT * FROM notes\n             WHERE alert ~* (%s)\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, (id,), limit=page_size, offset=offset)

    def get_customer_notes(self, customer: str, page: Optional[int] = None, page_size: Optional[int] = None) -> List[Any]:
        select: str = '\n            SELECT * FROM notes\n             WHERE customer=%s\n        '
        offset: int = ((page - 1) * page_size) if (page and page_size) else 0
        return self._fetchall(select, (customer,), limit=page_size, offset=offset)

    def update_note(self, id: str, **kwargs: Any) -> Any:
        update: str = '\n            UPDATE notes\n            SET\n        '
        if kwargs.get('text', None) is not None:
            update += 'text=%(text)s, '
        if kwargs.get('attributes', None) is not None:
            update += 'attributes=attributes || %(attributes)s, '
        update += '\n            "user"=COALESCE(%(user)s, "user"),\n            update_time=NOW() at time zone \'utc\'\n            WHERE id=%(id)s\n            RETURNING *\n        '
        kwargs['id'] = id
        kwargs['user'] = kwargs.get('user')
        return self._updateone(update, kwargs, returning=True)

    def delete_note(self, id: str) -> Any:
        delete: str = (
            '\n            DELETE FROM notes\n            WHERE id=%s\n            RETURNING id\n        '
        )
        return self._deleteone(delete, (id,), returning=True)

    def get_metrics(self, type: Optional[Any] = None) -> List[Any]:
        select: str = 'SELECT * FROM metrics'
        if type:
            select += ' WHERE type=%s'
            return self._fetchall(select, (type,))
        return self._fetchall(select, ())

    def set_gauge(self, gauge: Any) -> Any:
        upsert: str = (
            '\n            INSERT INTO metrics ("group", name, title, description, value, type)\n            VALUES (%(group)s, %(name)s, %(title)s, %(description)s, %(value)s, %(type)s)\n            ON CONFLICT ("group", name, type) DO UPDATE\n                SET value=%(value)s\n            RETURNING *\n        '
        )
        return self._upsert(upsert, vars(gauge))

    def inc_counter(self, counter: Any) -> Any:
        upsert: str = (
            '\n            INSERT INTO metrics ("group", name, title, description, count, type)\n            VALUES (%(group)s, %(name)s, %(title)s, %(description)s, %(count)s, %(type)s)\n            ON CONFLICT ("group", name, type) DO UPDATE\n                SET count=metrics.count + %(count)s\n            RETURNING *\n        '
        )
        return self._upsert(upsert, vars(counter))

    def update_timer(self, timer: Any) -> Any:
        upsert: str = (
            '\n            INSERT INTO metrics ("group", name, title, description, count, total_time, type)\n            VALUES (%(group)s, %(name)s, %(title)s, %(description)s, %(count)s, %(total_time)s, %(type)s)\n            ON CONFLICT ("group", name, type) DO UPDATE\n                SET count=metrics.count + %(count)s, total_time=metrics.total_time + %(total_time)s\n            RETURNING *\n        '
        )
        return self._upsert(upsert, vars(timer))

    def get_expired(self, expired_threshold: Optional[Any], info_threshold: Optional[Any]) -> List[Any]:
        if expired_threshold:
            delete: str = "\n                DELETE FROM alerts\n                 WHERE (status IN ('closed', 'expired')\n                        AND last_receive_time < (NOW() at time zone 'utc' - INTERVAL '%(expired_threshold)s seconds'))\n            "
            self._deleteall(delete, {'expired_threshold': expired_threshold})
        if info_threshold:
            delete: str = "\n                DELETE FROM alerts\n                 WHERE (severity=%(inform_severity)s\n                        AND last_receive_time < (NOW() at time zone 'utc' - INTERVAL '%(info_threshold)s seconds'))\n            "
            self._deleteall(delete, {'inform_severity': alarm_model.DEFAULT_INFORM_SEVERITY, 'info_threshold': info_threshold})
        select: str = "\n            SELECT *\n              FROM alerts\n             WHERE status NOT IN ('expired') AND COALESCE(timeout, {timeout})!=0\n               AND (last_receive_time + INTERVAL '1 second' * timeout) < NOW() at time zone 'utc'\n        ".format(timeout=current_app.config['ALERT_TIMEOUT'])
        return self._fetchall(select, {})

    def get_unshelve(self) -> List[Any]:
        select: str = "\n            SELECT DISTINCT ON (a.id) a.*\n              FROM alerts a, UNNEST(history) h\n             WHERE a.status='shelved'\n               AND h.type='shelve'\n               AND h.status='shelved'\n               AND COALESCE(h.timeout, {timeout})!=0\n               AND (a.update_time + INTERVAL '1 second' * h.timeout) < NOW() at time zone 'utc'\n          ORDER BY a.id, a.update_time DESC\n        ".format(timeout=current_app.config['SHELVE_TIMEOUT'])
        return self._fetchall(select, {})

    def get_unack(self) -> List[Any]:
        select: str = "\n            SELECT DISTINCT ON (a.id) a.*\n              FROM alerts a, UNNEST(history) h\n             WHERE a.status='ack'\n               AND h.type='ack'\n               AND h.status='ack'\n               AND COALESCE(h.timeout, {timeout})!=0\n               AND (a.update_time + INTERVAL '1 second' * h.timeout) < NOW() at time zone 'utc'\n          ORDER BY a.id, a.update_time DESC\n        ".format(timeout=current_app.config['ACK_TIMEOUT'])
        return self._fetchall(select, {})

    def _insert(self, query: str, vars: Any) -> Any:
        """
        Insert, with return.
        """
        cursor: PgCursor = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        return cursor.fetchone()

    def _fetchone(self, query: str, vars: Any) -> Any:
        """
        Return none or one row.
        """
        cursor: PgCursor = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        return cursor.fetchone()

    def _fetchall(self, query: str, vars: Any, limit: Optional[int] = None, offset: int = 0) -> List[Any]:
        """
        Return multiple rows.
        """
        if limit is None:
            limit = current_app.config['DEFAULT_PAGE_SIZE']
        query += f' LIMIT {limit} OFFSET {offset}'
        cursor: PgCursor = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        return cursor.fetchall()

    def _updateone(self, query: str, vars: Any, returning: bool = False) -> Any:
        """
        Update, with optional return.
        """
        cursor: PgCursor = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        return cursor.fetchone() if returning else None

    def _updateall(self, query: str, vars: Any, returning: bool = False) -> Any:
        """
        Update, with optional return.
        """
        cursor: PgCursor = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        return cursor.fetchall() if returning else None

    def _upsert(self, query: str, vars: Any) -> Any:
        """
        Insert or update, with return.
        """
        return self._insert(query, vars)

    def _deleteone(self, query: str, vars: Any, returning: bool = False) -> Any:
        """
        Delete, with optional return.
        """
        cursor: PgCursor = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        return cursor.fetchone() if returning else None

    def _deleteall(self, query: str, vars: Any, returning: bool = False) -> Any:
        """
        Delete multiple rows, with optional return.
        """
        cursor: PgCursor = self.get_db().cursor()
        self._log(cursor, query, vars)
        cursor.execute(query, vars)
        self.get_db().commit()
        return cursor.fetchall() if returning else None

    def _log(self, cursor: PgCursor, query: str, vars: Any) -> None:
        current_app.logger.debug('{stars}\n{query}\n{stars}'.format(stars='*' * 40, query=cursor.mogrify(query, vars).decode('utf-8')))
