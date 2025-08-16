from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import current_app
from pymongo import ASCENDING, TEXT, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure

from alerta.app import alarm_model
from alerta.database.base import Database
from alerta.exceptions import NoCustomerMatch
from alerta.models.enums import ADMIN_SCOPES
from alerta.models.heartbeat import HeartbeatStatus

from .utils import Query


class Backend(Database):

    def create_engine(self, app: Any, uri: str, dbname: Optional[str] = None, schema: Optional[str] = None, raise_on_error: bool = True) -> None:
        self.uri = uri
        self.dbname = dbname

        db = self.connect()

        try:
            self._create_indexes(db)
        except Exception as e:
            if raise_on_error:
                raise
            app.logger.warning(e)

        try:
            self._update_lookups(db)
        except Exception as e:
            if raise_on_error:
                raise
            app.logger.warning(e)

    def connect(self) -> MongoDatabase:
        self.client = MongoClient(self.uri)
        if self.dbname:
            return self.client[self.dbname]
        else:
            return self.client.get_database()

    @staticmethod
    def _create_indexes(db: MongoDatabase) -> None:
        db.alerts.create_index(
            [('environment', ASCENDING), ('customer', ASCENDING), ('resource', ASCENDING), ('event', ASCENDING)],
            unique=True
        )
        db.alerts.create_index([('$**', TEXT)])
        db.customers.drop_indexes()  # FIXME: should only drop customers index if it's unique (ie. the old one)
        db.customers.create_index([('match', ASCENDING)])
        db.heartbeats.create_index([('origin', ASCENDING), ('customer', ASCENDING)], unique=True)
        db.keys.create_index([('key', ASCENDING)], unique=True)
        db.perms.create_index([('match', ASCENDING)], unique=True)
        db.users.drop_indexes()
        db.users.create_index([('login', ASCENDING)], unique=True,
                              partialFilterExpression={'login': {'$type': 'string'}})
        db.users.create_index([('email', ASCENDING)], unique=True,
                              partialFilterExpression={'email': {'$type': 'string'}})
        db.groups.create_index([('name', ASCENDING)], unique=True)
        db.metrics.create_index([('group', ASCENDING), ('name', ASCENDING)], unique=True)

    @staticmethod
    def _update_lookups(db: MongoDatabase) -> None:
        for severity, code in alarm_model.Severity.items():
            db.codes.update_one(
                {'severity': severity},
                {'$set': {'severity': severity, 'code': code}},
                upsert=True
            )
        for status, state in alarm_model.Status.items():
            db.states.update_one(
                {'status': status},
                {'$set': {'status': status, 'state': state}},
                upsert=True
            )

    @property
    def name(self) -> str:
        return self.get_db().name

    @property
    def version(self) -> str:
        return self.get_db().client.server_info()['version']

    @property
    def is_alive(self) -> bool:
        try:
            self.get_db().client.admin.command('ismaster')
        except ConnectionFailure:
            return False
        return True

    def close(self, db: MongoDatabase) -> None:
        self.client.close()

    def destroy(self) -> None:
        db = self.connect()
        self.client.drop_database(db.name)

    # ALERTS

    def get_severity(self, alert: Any) -> Optional[str]:
        query = {
            'environment': alert.environment,
            'resource': alert.resource,
            '$or': [
                {
                    'event': alert.event,
                    'severity': {'$ne': alert.severity}
                },
                {
                    'event': {'$ne': alert.event},
                    'correlate': alert.event
                }],
            'customer': alert.customer
        }
        r = self.get_db().alerts.find_one(query, projection={'severity': 1, '_id': 0})
        return r['severity'] if r else None

    def get_status(self, alert: Any) -> Optional[str]:
        query = {
            'environment': alert.environment,
            'resource': alert.resource,
            '$or': [
                {
                    'event': alert.event
                },
                {
                    'correlate': alert.event,
                }
            ],
            'customer': alert.customer
        }
        r = self.get_db().alerts.find_one(query, projection={'status': 1, '_id': 0})
        return r['status'] if r else None

    def is_duplicate(self, alert: Any) -> Optional[Dict[str, Any]]:
        query = {
            'environment': alert.environment,
            'resource': alert.resource,
            'event': alert.event,
            'severity': alert.severity,
            'customer': alert.customer
        }
        return self.get_db().alerts.find_one(query)

    def is_correlated(self, alert: Any) -> Optional[Dict[str, Any]]:
        query = {
            'environment': alert.environment,
            'resource': alert.resource,
            '$or': [
                {
                    'event': alert.event,
                    'severity': {'$ne': alert.severity}
                },
                {
                    'event': {'$ne': alert.event},
                    'correlate': alert.event
                }],
            'customer': alert.customer
        }
        return self.get_db().alerts.find_one(query)

    def is_flapping(self, alert: Any, window: int = 1800, count: int = 2) -> bool:
        pipeline = [
            {'$match': {
                'environment': alert.environment,
                'resource': alert.resource,
                'event': alert.event,
                'customer': alert.customer
            }},
            {'$unwind': '$history'},
            {'$match': {
                'history.updateTime': {'$gt': datetime.utcnow() - timedelta(seconds=window)},
                'history.type': 'severity'
            }},
            {'$group': {'_id': '$history.type', 'count': {'$sum': 1}}}
        ]
        responses = self.get_db().alerts.aggregate(pipeline)
        for r in responses:
            if r['count'] > count:
                return True
        return False

    def dedup_alert(self, alert: Any, history: Any) -> Optional[Dict[str, Any]]:
        query = {
            'environment': alert.environment,
            'resource': alert.resource,
            'event': alert.event,
            'severity': alert.severity,
            'customer': alert.customer
        }

        now = datetime.utcnow()
        update = {
            '$set': {
                'status': alert.status,
                'service': alert.service,
                'value': alert.value,
                'text': alert.text,
                'timeout': alert.timeout,
                'rawData': alert.raw_data,
                'repeat': True,
                'lastReceiveId': alert.id,
                'lastReceiveTime': now
            },
            '$addToSet': {'tags': {'$each': alert.tags}},
            '$inc': {'duplicateCount': 1}
        }

        attributes = {'attributes.' + k: v for k, v in alert.attributes.items()}
        update['$set'].update(attributes)

        if alert.update_time:
            update['$set']['updateTime'] = alert.update_time

        if history:
            update['$push'] = {
                'history': {
                    '$each': [history.serialize],
                    '$slice': current_app.config['HISTORY_LIMIT'],
                    '$position': 0
                }
            }

        return self.get_db().alerts.find_one_and_update(
            query,
            update=update,
            return_document=ReturnDocument.AFTER
        )

    def correlate_alert(self, alert: Any, history: List[Any]) -> Optional[Dict[str, Any]]:
        query = {
            'environment': alert.environment,
            'resource': alert.resource,
            '$or': [
                {
                    'event': alert.event,
                    'severity': {'$ne': alert.severity}
                },
                {
                    'event': {'$ne': alert.event},
                    'correlate': alert.event
                }],
            'customer': alert.customer
        }

        update = {
            '$set': {
                'event': alert.event,
                'severity': alert.severity,
                'status': alert.status,
                'service': alert.service,
                'value': alert.value,
                'text': alert.text,
                'createTime': alert.create_time,
                'timeout': alert.timeout,
                'rawData': alert.raw_data,
                'duplicateCount': alert.duplicate_count,
                'repeat': alert.repeat,
                'previousSeverity': alert.previous_severity,
                'trendIndication': alert.trend_indication,
                'receiveTime': alert.receive_time,
                'lastReceiveId': alert.last_receive_id,
                'lastReceiveTime': alert.last_receive_time
            },
            '$addToSet': {'tags': {'$each': alert.tags}},
            '$push': {
                'history': {
                    '$each': [h.serialize for h in history],
                    '$slice': current_app.config['HISTORY_LIMIT'],
                    '$position': 0
                }
            }
        }

        attributes = {'attributes.' + k: v for k, v in alert.attributes.items()}
        update['$set'].update(attributes)

        if alert.update_time:
            update['$set']['updateTime'] = alert.update_time

        return self.get_db().alerts.find_one_and_update(
            query,
            update=update,
            return_document=ReturnDocument.AFTER
        )

    def create_alert(self, alert: Any) -> Optional[Dict[str, Any]]:
        data = {
            '_id': alert.id,
            'resource': alert.resource,
            'event': alert.event,
            'environment': alert.environment,
            'severity': alert.severity,
            'correlate': alert.correlate,
            'status': alert.status,
            'service': alert.service,
            'group': alert.group,
            'value': alert.value,
            'text': alert.text,
            'tags': alert.tags,
            'attributes': alert.attributes,
            'origin': alert.origin,
            'type': alert.event_type,
            'createTime': alert.create_time,
            'timeout': alert.timeout,
            'rawData': alert.raw_data,
            'customer': alert.customer,
            'duplicateCount': alert.duplicate_count,
            'repeat': alert.repeat,
            'previousSeverity': alert.previous_severity,
            'trendIndication': alert.trend_indication,
            'receiveTime': alert.receive_time,
            'lastReceiveId': alert.last_receive_id,
            'lastReceiveTime': alert.last_receive_time,
            'updateTime': alert.update_time,
            'history': [h.serialize for h in alert.history]
        }
        if self.get_db().alerts.insert_one(data).inserted_id == alert.id:
            return data
        return None

    def set_alert(self, id: str, severity: str, status: str, tags: List[str], attributes: Dict[str, Any], timeout: int, previous_severity: str, update_time: datetime, history: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        query = {'_id': {'$regex': '^' + id}}

        update = {
            '$set': {
                'severity': severity,
                'status': status,
                'attributes': attributes,
                'timeout': timeout,
                'previousSeverity': previous_severity,
                'updateTime': update_time
            },
            '$addToSet': {'tags': {'$each': tags}},
            '$push': {
                'history': {
                    '$each': [h.serialize for h in history] if history else [],
                    '$slice': current_app.config['HISTORY_LIMIT'],
                    '$position': 0
                }
            }
        }

        return self.get_db().alerts.find_one_and_update(
            query,
            update=update,
            return_document=ReturnDocument.AFTER
        )

    def get_alert(self, id: str, customers: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        if len(id) == 8:
            query = {'$or': [{'_id': {'$regex': '^' + id}}, {'lastReceiveId': {'$regex': '^' + id}}]}
        else:
            query = {'$or': [{'_id': id}, {'lastReceiveId': id}]}

        if customers:
            query['customer'] = {'$in': customers}

        return self.get_db().alerts.find_one(query)

    # STATUS, TAGS, ATTRIBUTES

    def set_status(self, id: str, status: str, timeout: int, update_time: datetime, history: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        query = {'_id': {'$regex': '^' + id}}

        update = {
            '$set': {'status': status, 'timeout': timeout, 'updateTime': update_time},
            '$push': {
                'history': {
                    '$each': [history.serialize] if history else [],
                    '$slice': current_app.config['HISTORY_LIMIT'],
                    '$position': 0
                }
            }
        }
        return self.get_db().alerts.find_one_and_update(
            query,
            update=update,
            return_document=ReturnDocument.AFTER
        )

    def tag_alert(self, id: str, tags: List[str]) -> bool:
        response = self.get_db().alerts.update_one(
            {'_id': {'$regex': '^' + id}}, {'$addToSet': {'tags': {'$each': tags}}})
        return response.matched_count > 0

    def untag_alert(self, id: str, tags: List[str]) -> bool:
        response = self.get_db().alerts.update_one({'_id': {'$regex': '^' + id}}, {'$pullAll': {'tags': tags}})
        return response.matched_count > 0

    def update_tags(self, id: str, tags: List[str]) -> bool:
        response = self.get_db().alerts.update_one({'_id': {'$regex': '^' + id}}, update={'$set': {'tags': tags}})
        return response.matched_count > 0

    def update_attributes(self, id: str, old_attrs: Dict[str, Any], new_attrs: Dict[str, Any]) -> Dict[str, Any]:
        update = dict()
        set_value = {'attributes.' + k: v for k, v in new_attrs.items() if v is not None}
        if set_value:
            update['$set'] = set_value
        unset_value = {'attributes.' + k: v for k, v in new_attrs.items() if v is None}
        if unset_value:
            update['$unset'] = unset_value

        if update:
            return self.get_db().alerts.find_one_and_update(
                {'_id': {'$regex': '^' + id}},
                update=update,
                return_document=ReturnDocument.AFTER
            )['attributes']
        return {}

    def delete_alert(self, id: str) -> bool:
        response = self.get_db().alerts.delete_one({'_id': {'$regex': '^' + id}})
        return response.deleted_count == 1

    # BULK

    def tag_alerts(self, query: Optional[Query] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        query = query or Query()
        updated = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.update(query.where, {'$addToSet': {'tags': {'$each': tags or []}}})
        return updated if response['n'] else []

    def untag_alerts(self, query: Optional[Query] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        query = query or Query()
        updated = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.update(query.where, {'$pullAll': {'tags': tags or []}})
        return updated if response['n'] else []

    def update_attributes_by_query(self, query: Optional[Query] = None, attributes: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query = query or Query()
        update = dict()
        set_value = {'attributes.' + k: v for k, v in (attributes or {}).items() if v is not None}
        if set_value:
            update['$set'] = set_value
        unset_value = {'attributes.' + k: v for k, v in (attributes or {}).items() if v is None}
        if unset_value:
            update['$unset'] = unset_value

        updated = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.update_many(query.where, update=update)
        return updated if response.matched_count > 0 else []

    def delete_alerts(self, query: Optional[Query] = None) -> List[Dict[str, Any]]:
        query = query or Query()
        deleted = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.remove(query.where)
        return deleted if response['n'] else []

    # SEARCH & HISTORY

    def add_history(self, id: str, history: Any) -> Optional[Dict[str, Any]]:
        query = {'_id': {'$regex': '^' + id}}

        update = {
            '$