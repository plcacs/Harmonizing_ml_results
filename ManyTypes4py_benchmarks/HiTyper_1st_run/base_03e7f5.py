from collections import defaultdict
from datetime import datetime, timedelta
from flask import current_app
from pymongo import ASCENDING, TEXT, MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure
from alerta.app import alarm_model
from alerta.database.base import Database
from alerta.exceptions import NoCustomerMatch
from alerta.models.enums import ADMIN_SCOPES
from alerta.models.heartbeat import HeartbeatStatus
from .utils import Query

class Backend(Database):

    def create_engine(self, app: Union[bool, str, typing.Callable], uri: Union[str, bool], dbname: Union[None, bool, list, str]=None, schema: Union[None, bool, str, typing.Callable]=None, raise_on_error: bool=True) -> None:
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

    def connect(self) -> base.Connection:
        self.client = MongoClient(self.uri)
        if self.dbname:
            return self.client[self.dbname]
        else:
            return self.client.get_database()

    @staticmethod
    def _create_indexes(db: deeplearning.ml4pl.ir.ir_database.Database) -> None:
        db.alerts.create_index([('environment', ASCENDING), ('customer', ASCENDING), ('resource', ASCENDING), ('event', ASCENDING)], unique=True)
        db.alerts.create_index([('$**', TEXT)])
        db.customers.drop_indexes()
        db.customers.create_index([('match', ASCENDING)])
        db.heartbeats.create_index([('origin', ASCENDING), ('customer', ASCENDING)], unique=True)
        db.keys.create_index([('key', ASCENDING)], unique=True)
        db.perms.create_index([('match', ASCENDING)], unique=True)
        db.users.drop_indexes()
        db.users.create_index([('login', ASCENDING)], unique=True, partialFilterExpression={'login': {'$type': 'string'}})
        db.users.create_index([('email', ASCENDING)], unique=True, partialFilterExpression={'email': {'$type': 'string'}})
        db.groups.create_index([('name', ASCENDING)], unique=True)
        db.metrics.create_index([('group', ASCENDING), ('name', ASCENDING)], unique=True)

    @staticmethod
    def _update_lookups(db: Union[pymongo.database.Database, tortoise.backends.base.clienBaseDBAsyncClient]) -> None:
        for severity, code in alarm_model.Severity.items():
            db.codes.update_one({'severity': severity}, {'$set': {'severity': severity, 'code': code}}, upsert=True)
        for status, state in alarm_model.Status.items():
            db.states.update_one({'status': status}, {'$set': {'status': status, 'state': state}}, upsert=True)

    @property
    def name(self):
        return self.get_db().name

    @property
    def version(self) -> Union[str, None, pymongo.MongoClient]:
        return self.get_db().client.server_info()['version']

    @property
    def is_alive(self) -> bool:
        try:
            self.get_db().client.admin.command('ismaster')
        except ConnectionFailure:
            return False
        return True

    def close(self, db: pymongo.database.Database) -> None:
        self.client.close()

    def destroy(self) -> None:
        db = self.connect()
        self.client.drop_database(db.name)

    def get_severity(self, alert: Union[str, server.models.bike.Bike, dict]) -> None:
        """
        Get severity of correlated alert. Used to determine previous severity.
        """
        query = {'environment': alert.environment, 'resource': alert.resource, '$or': [{'event': alert.event, 'severity': {'$ne': alert.severity}}, {'event': {'$ne': alert.event}, 'correlate': alert.event}], 'customer': alert.customer}
        r = self.get_db().alerts.find_one(query, projection={'severity': 1, '_id': 0})
        return r['severity'] if r else None

    def get_status(self, alert: Union[str, dict[str, typing.Any], alerta.models.alerAlert]) -> None:
        """
        Get status of correlated or duplicate alert. Used to determine previous status.
        """
        query = {'environment': alert.environment, 'resource': alert.resource, '$or': [{'event': alert.event}, {'correlate': alert.event}], 'customer': alert.customer}
        r = self.get_db().alerts.find_one(query, projection={'status': 1, '_id': 0})
        return r['status'] if r else None

    def is_duplicate(self, alert: Union[str, server.models.bike.Bike, alerta.models.alerAlert]) -> Union[str, None, dict[str, object]]:
        query = {'environment': alert.environment, 'resource': alert.resource, 'event': alert.event, 'severity': alert.severity, 'customer': alert.customer}
        return self.get_db().alerts.find_one(query)

    def is_correlated(self, alert: Union[str, server.models.bike.Bike, alerta.models.alerAlert]) -> Union[str, None, dict[str, object]]:
        query = {'environment': alert.environment, 'resource': alert.resource, '$or': [{'event': alert.event, 'severity': {'$ne': alert.severity}}, {'event': {'$ne': alert.event}, 'correlate': alert.event}], 'customer': alert.customer}
        return self.get_db().alerts.find_one(query)

    def is_flapping(self, alert: Union[int, tuple[typing.Union[int,list[int]]], zerver.models.Realm], window: int=1800, count: Any=2) -> bool:
        """
        Return true if alert severity has changed more than X times in Y seconds
        """
        pipeline = [{'$match': {'environment': alert.environment, 'resource': alert.resource, 'event': alert.event, 'customer': alert.customer}}, {'$unwind': '$history'}, {'$match': {'history.updateTime': {'$gt': datetime.utcnow() - timedelta(seconds=window)}, 'history.type': 'severity'}}, {'$group': {'_id': '$history.type', 'count': {'$sum': 1}}}]
        responses = self.get_db().alerts.aggregate(pipeline)
        for r in responses:
            if r['count'] > count:
                return True
        return False

    def dedup_alert(self, alert: Union[str, transfer.models.Shop, dict], history: Union[typing.Mapping, dict[str, typing.Any], zerver.models.UserProfile]) -> bool:
        """
        Update alert status, service, value, text, timeout and rawData, increment duplicate count and set
        repeat=True, and keep track of last receive id and time but don't append to history unless status changes.
        """
        query = {'environment': alert.environment, 'resource': alert.resource, 'event': alert.event, 'severity': alert.severity, 'customer': alert.customer}
        now = datetime.utcnow()
        update = {'$set': {'status': alert.status, 'service': alert.service, 'value': alert.value, 'text': alert.text, 'timeout': alert.timeout, 'rawData': alert.raw_data, 'repeat': True, 'lastReceiveId': alert.id, 'lastReceiveTime': now}, '$addToSet': {'tags': {'$each': alert.tags}}, '$inc': {'duplicateCount': 1}}
        attributes = {'attributes.' + k: v for k, v in alert.attributes.items()}
        update['$set'].update(attributes)
        if alert.update_time:
            update['$set']['updateTime'] = alert.update_time
        if history:
            update['$push'] = {'history': {'$each': [history.serialize], '$slice': current_app.config['HISTORY_LIMIT'], '$position': 0}}
        return self.get_db().alerts.find_one_and_update(query, update=update, return_document=ReturnDocument.AFTER)

    def correlate_alert(self, alert: Union[str, list[str], T], history: Union[server.models.bike.Bike, str]) -> bool:
        """
        Update alert key attributes, reset duplicate count and set repeat=False, keep track of last
        receive id and time, appending all to history. Append to history again if status changes.
        """
        query = {'environment': alert.environment, 'resource': alert.resource, '$or': [{'event': alert.event, 'severity': {'$ne': alert.severity}}, {'event': {'$ne': alert.event}, 'correlate': alert.event}], 'customer': alert.customer}
        update = {'$set': {'event': alert.event, 'severity': alert.severity, 'status': alert.status, 'service': alert.service, 'value': alert.value, 'text': alert.text, 'createTime': alert.create_time, 'timeout': alert.timeout, 'rawData': alert.raw_data, 'duplicateCount': alert.duplicate_count, 'repeat': alert.repeat, 'previousSeverity': alert.previous_severity, 'trendIndication': alert.trend_indication, 'receiveTime': alert.receive_time, 'lastReceiveId': alert.last_receive_id, 'lastReceiveTime': alert.last_receive_time}, '$addToSet': {'tags': {'$each': alert.tags}}, '$push': {'history': {'$each': [h.serialize for h in history], '$slice': current_app.config['HISTORY_LIMIT'], '$position': 0}}}
        attributes = {'attributes.' + k: v for k, v in alert.attributes.items()}
        update['$set'].update(attributes)
        if alert.update_time:
            update['$set']['updateTime'] = alert.update_time
        return self.get_db().alerts.find_one_and_update(query, update=update, return_document=ReturnDocument.AFTER)

    def create_alert(self, alert: Union[str, dict, list[str]]) -> dict[typing.Text, list]:
        data = {'_id': alert.id, 'resource': alert.resource, 'event': alert.event, 'environment': alert.environment, 'severity': alert.severity, 'correlate': alert.correlate, 'status': alert.status, 'service': alert.service, 'group': alert.group, 'value': alert.value, 'text': alert.text, 'tags': alert.tags, 'attributes': alert.attributes, 'origin': alert.origin, 'type': alert.event_type, 'createTime': alert.create_time, 'timeout': alert.timeout, 'rawData': alert.raw_data, 'customer': alert.customer, 'duplicateCount': alert.duplicate_count, 'repeat': alert.repeat, 'previousSeverity': alert.previous_severity, 'trendIndication': alert.trend_indication, 'receiveTime': alert.receive_time, 'lastReceiveId': alert.last_receive_id, 'lastReceiveTime': alert.last_receive_time, 'updateTime': alert.update_time, 'history': [h.serialize for h in alert.history]}
        if self.get_db().alerts.insert_one(data).inserted_id == alert.id:
            return data

    def set_alert(self, id: str, severity: Union[int, datetime.datetime.datetime], status: Union[int, datetime.datetime.datetime], tags: Union[int, datetime.datetime.datetime], attributes: Union[int, datetime.datetime.datetime], timeout: Union[int, datetime.datetime.datetime], previous_severity: Union[int, datetime.datetime.datetime], update_time: Union[int, datetime.datetime.datetime], history: datetime.datetime.datetime=None) -> bool:
        query = {'_id': {'$regex': '^' + id}}
        update = {'$set': {'severity': severity, 'status': status, 'attributes': attributes, 'timeout': timeout, 'previousSeverity': previous_severity, 'updateTime': update_time}, '$addToSet': {'tags': {'$each': tags}}, '$push': {'history': {'$each': [h.serialize for h in history], '$slice': current_app.config['HISTORY_LIMIT'], '$position': 0}}}
        return self.get_db().alerts.find_one_and_update(query, update=update, return_document=ReturnDocument.AFTER)

    def get_alert(self, id: str, customers: Union[None, str, int]=None) -> Union[str, lms.lmsdb.models.User, lms.lmsdb.models.Solution]:
        if len(id) == 8:
            query = {'$or': [{'_id': {'$regex': '^' + id}}, {'lastReceiveId': {'$regex': '^' + id}}]}
        else:
            query = {'$or': [{'_id': id}, {'lastReceiveId': id}]}
        if customers:
            query['customer'] = {'$in': customers}
        return self.get_db().alerts.find_one(query)

    def set_status(self, id: Union[str, datetime.datetime], status: Union[float, str, int], timeout: Union[float, str, int], update_time: Union[float, str, int], history: Union[None, float, str, int]=None) -> Union[bool, str]:
        """
        Set status and update history.
        """
        query = {'_id': {'$regex': '^' + id}}
        update = {'$set': {'status': status, 'timeout': timeout, 'updateTime': update_time}, '$push': {'history': {'$each': [history.serialize], '$slice': current_app.config['HISTORY_LIMIT'], '$position': 0}}}
        return self.get_db().alerts.find_one_and_update(query, update=update, return_document=ReturnDocument.AFTER)

    def tag_alert(self, id: str, tags: Union[str, list[str], bytes]) -> bool:
        """
        Append tags to tag list. Don't add same tag more than once.
        """
        response = self.get_db().alerts.update_one({'_id': {'$regex': '^' + id}}, {'$addToSet': {'tags': {'$each': tags}}})
        return response.matched_count > 0

    def untag_alert(self, id: str, tags: Union[str, list[str]]) -> bool:
        """
        Remove tags from tag list.
        """
        response = self.get_db().alerts.update_one({'_id': {'$regex': '^' + id}}, {'$pullAll': {'tags': tags}})
        return response.matched_count > 0

    def update_tags(self, id: str, tags: Union[str, None]) -> bool:
        response = self.get_db().alerts.update_one({'_id': {'$regex': '^' + id}}, update={'$set': {'tags': tags}})
        return response.matched_count > 0

    def update_attributes(self, id: Any, old_attrs: Union[str, bool, None], new_attrs: dict) -> Union[bool, str, None, dict]:
        update = dict()
        set_value = {'attributes.' + k: v for k, v in new_attrs.items() if v is not None}
        if set_value:
            update['$set'] = set_value
        unset_value = {'attributes.' + k: v for k, v in new_attrs.items() if v is None}
        if unset_value:
            update['$unset'] = unset_value
        if update:
            return self.get_db().alerts.find_one_and_update({'_id': {'$regex': '^' + id}}, update=update, return_document=ReturnDocument.AFTER)['attributes']
        return {}

    def delete_alert(self, id: str) -> bool:
        response = self.get_db().alerts.delete_one({'_id': {'$regex': '^' + id}})
        return True if response.deleted_count == 1 else False

    def tag_alerts(self, query: Union[None, str, users.models.CustomUser, list]=None, tags: Union[None, str, dict, users.models.CustomUser]=None) -> list:
        query = query or Query()
        updated = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.update(query.where, {'$addToSet': {'tags': {'$each': tags}}})
        return updated if response['n'] else []

    def untag_alerts(self, query: Union[None, str, users.models.CustomUser, dict]=None, tags: Union[None, str, tildes.models.topic.Topic, dict]=None) -> list:
        query = query or Query()
        updated = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.update(query.where, {'$pullAll': {'tags': tags}})
        return updated if response['n'] else []

    def update_attributes_by_query(self, query: Union[None, dict, str]=None, attributes: Union[dict, sqlalchemy.orm.query.Query]=None) -> list:
        query = query or Query()
        update = dict()
        set_value = {'attributes.' + k: v for k, v in attributes.items() if v is not None}
        if set_value:
            update['$set'] = set_value
        unset_value = {'attributes.' + k: v for k, v in attributes.items() if v is None}
        if unset_value:
            update['$unset'] = unset_value
        updated = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.update_many(query.where, update=update)
        return updated if response.matched_count > 0 else []

    def delete_alerts(self, query: Union[None, alerta.database.base.Query, str, list['Query']]=None) -> list:
        query = query or Query()
        deleted = list(self.get_db().alerts.find(query.where, projection={'_id': 1}))
        response = self.get_db().alerts.remove(query.where)
        return deleted if response['n'] else []

    def add_history(self, id: Union[str, src.dbp.models.LocationTuple], history: Union[str, int, dict]) -> bool:
        query = {'_id': {'$regex': '^' + id}}
        update = {'$push': {'history': {'$each': [history.serialize], '$slice': current_app.config['HISTORY_LIMIT'], '$position': 0}}}
        return self.get_db().alerts.find_one_and_update(query, update=update, return_document=ReturnDocument.AFTER)

    def get_alerts(self, query: Union[None, int, dict]=None, raw_data: bool=False, history: bool=False, page: Union[int, alerta.database.base.Query]=None, page_size: Union[int, alerta.database.base.Query]=None) -> Union[list[str], int]:
        query = query or Query()
        fields = dict()
        if not raw_data:
            fields['rawData'] = 0
        if not history:
            fields['history'] = 0
        pipeline = [{'$lookup': {'from': 'codes', 'localField': 'severity', 'foreignField': 'severity', 'as': 'fromCodes'}}, {'$replaceRoot': {'newRoot': {'$mergeObjects': [{'$arrayElemAt': ['$fromCodes', 0]}, '$$ROOT']}}}, {'$project': {'fromCodes': 0}}, {'$lookup': {'from': 'states', 'localField': 'status', 'foreignField': 'status', 'as': 'fromStates'}}, {'$replaceRoot': {'newRoot': {'$mergeObjects': [{'$arrayElemAt': ['$fromStates', 0]}, '$$ROOT']}}}, {'$project': {'fromStates': 0}}, {'$match': query.where}, {'$project': fields}, {'$sort': {k: v for k, v in query.sort}}, {'$skip': (page - 1) * page_size}, {'$limit': page_size}]
        return self.get_db().alerts.aggregate(pipeline)

    def get_alert_history(self, alert: Union[str, int, None], page: Union[int, alerta.database.base.Query]=None, page_size: Union[int, alerta.database.base.Query]=None) -> list[dict[typing.Text, str]]:
        query = {'environment': alert.environment, 'resource': alert.resource, '$or': [{'event': alert.event}, {'correlate': alert.event}], 'customer': alert.customer}
        fields = {'resource': 1, 'event': 1, 'environment': 1, 'customer': 1, 'service': 1, 'group': 1, 'tags': 1, 'attributes': 1, 'origin': 1, 'type': 1, 'history': 1}
        pipeline = [{'$unwind': '$history'}, {'$match': query}, {'$project': fields}, {'$sort': {'history.updateTime': -1}}, {'$skip': (page - 1) * page_size}, {'$limit': page_size}]
        responses = self.get_db().alerts.aggregate(pipeline)
        history = list()
        for response in responses:
            history.append({'id': response['history']['id'], 'resource': response['resource'], 'event': response['history'].get('event'), 'environment': response['environment'], 'severity': response['history'].get('severity'), 'service': response['service'], 'status': response['history'].get('status'), 'group': response['group'], 'value': response['history'].get('value'), 'text': response['history'].get('text'), 'tags': response['tags'], 'attributes': response['attributes'], 'origin': response['origin'], 'updateTime': response['history']['updateTime'], 'user': response['history'].get('user'), 'timeout': response['history'].get('timeout'), 'type': response['history'].get('type', 'unknown'), 'customer': response.get('customer')})
        return history

    def get_history(self, query: Union[None, int, str]=None, page: Union[int, shop.transfer.models.ShopID]=None, page_size: Union[int, str, shop.transfer.models.ShopID]=None) -> list[dict[typing.Text, str]]:
        query = query or Query()
        fields = {'resource': 1, 'event': 1, 'environment': 1, 'customer': 1, 'service': 1, 'group': 1, 'tags': 1, 'attributes': 1, 'origin': 1, 'user': 1, 'timeout': 1, 'type': 1, 'history': 1}
        pipeline = [{'$unwind': '$history'}, {'$match': query.where}, {'$project': fields}, {'$sort': {'history.updateTime': -1}}, {'$skip': (page - 1) * page_size}, {'$limit': page_size}]
        responses = self.get_db().alerts.aggregate(pipeline)
        history = list()
        for response in responses:
            history.append({'id': response['history']['id'], 'resource': response['resource'], 'event': response['history']['event'], 'environment': response['environment'], 'severity': response['history']['severity'], 'service': response['service'], 'status': response['history']['status'], 'group': response['group'], 'value': response['history']['value'], 'text': response['history']['text'], 'tags': response['tags'], 'attributes': response['attributes'], 'origin': response['origin'], 'updateTime': response['history']['updateTime'], 'user': response.get('user'), 'timeout': response.get('timeout'), 'type': response['history'].get('type', 'unknown'), 'customer': response.get('customer', None)})
        return history

    def get_count(self, query: Union[None, alerta.database.base.Query, str]=None) -> Union[int, str]:
        """
        Return total number of alerts that meet the query filter.
        """
        query = query or Query()
        return self.get_db().alerts.count_documents(query.where)

    def get_counts(self, query: Union[None, str, alerta.database.base.Query, list[str]]=None, group: Union[dict, models.PersonModel, dict[str, str]]=None):
        query = query or Query()
        if group is None:
            raise ValueError('Must define a group')
        pipeline = [{'$match': query.where}, {'$project': {group: 1}}, {'$group': {'_id': '$' + group, 'count': {'$sum': 1}}}]
        responses = self.get_db().alerts.aggregate(pipeline)
        counts = dict()
        for response in responses:
            counts[response['_id']] = response['count']
        return counts

    def get_counts_by_severity(self, query: Union[None, alerta.database.base.Query, str]=None) -> Union[int, set[int], str]:
        query = query or Query()
        return self.get_counts(query, group='severity')

    def get_counts_by_status(self, query: Union[None, alerta.database.base.Query, str]=None) -> Union[int, str, None, list[int]]:
        query = query or Query()
        return self.get_counts(query, group='status')

    def get_topn_count(self, query: Union[None, daylighdb.models.User, str, daylighdb.models.Profile]=None, group: typing.Text='event', topn: int=100) -> list[dict[typing.Text, str]]:
        query = query or Query()
        pipeline = [{'$match': query.where}, {'$unwind': '$service'}, {'$group': {'_id': f'${group}', 'count': {'$sum': 1}, 'duplicateCount': {'$sum': '$duplicateCount'}, 'environments': {'$addToSet': '$environment'}, 'services': {'$addToSet': '$service'}, 'resources': {'$addToSet': {'id': '$_id', 'resource': '$resource'}}}}, {'$sort': {'count': -1, 'duplicateCount': -1}}, {'$limit': topn}]
        responses = self.get_db().alerts.aggregate(pipeline, allowDiskUse=True)
        top = list()
        for response in responses:
            top.append({f'{group}': response['_id'], 'environments': response['environments'], 'services': response['services'], 'resources': response['resources'], 'count': response['count'], 'duplicateCount': response['duplicateCount']})
        return top

    def get_topn_flapping(self, query: Union[None, str, int]=None, group: typing.Text='event', topn: int=100) -> list[dict[typing.Text, str]]:
        query = query or Query()
        pipeline = [{'$match': query.where}, {'$unwind': '$service'}, {'$unwind': '$history'}, {'$match': {'history.type': 'severity'}}, {'$group': {'_id': f'${group}', 'count': {'$sum': 1}, 'duplicateCount': {'$max': '$duplicateCount'}, 'environments': {'$addToSet': '$environment'}, 'services': {'$addToSet': '$service'}, 'resources': {'$addToSet': {'id': '$_id', 'resource': '$resource'}}}}, {'$sort': {'count': -1, 'duplicateCount': -1}}, {'$limit': topn}]
        responses = self.get_db().alerts.aggregate(pipeline, allowDiskUse=True)
        top = list()
        for response in responses:
            top.append({f'{group}': response['_id'], 'environments': response['environments'], 'services': response['services'], 'resources': response['resources'], 'count': response['count'], 'duplicateCount': response['duplicateCount']})
        return top

    def get_topn_standing(self, query: Union[None, str, dict, zerver.models.UserProfile]=None, group: typing.Text='event', topn: int=100) -> list[dict[typing.Text, str]]:
        query = query or Query()
        pipeline = [{'$match': query.where}, {'$unwind': '$service'}, {'$group': {'_id': f'${group}', 'count': {'$sum': 1}, 'duplicateCount': {'$sum': '$duplicateCount'}, 'lifeTime': {'$sum': {'$subtract': ['$lastReceiveTime', '$createTime']}}, 'environments': {'$addToSet': '$environment'}, 'services': {'$addToSet': '$service'}, 'resources': {'$addToSet': {'id': '$_id', 'resource': '$resource'}}}}, {'$sort': {'lifeTime': -1, 'duplicateCount': -1}}, {'$limit': topn}]
        responses = self.get_db().alerts.aggregate(pipeline, allowDiskUse=True)
        top = list()
        for response in responses:
            top.append({f'{group}': response['_id'], 'environments': response['environments'], 'services': response['services'], 'resources': response['resources'], 'count': response['count'], 'duplicateCount': response['duplicateCount']})
        return top

    def get_environments(self, query: Union[None, str, list[str], list[dict]]=None, topn: int=1000) -> list[dict[typing.Text, typing.Union[str,bool]]]:
        query = query or Query()

        def pipeline(group_by: Any) -> list[typing.Union[dict[typing.Text, dict[typing.Text, int]],dict[typing.Text, dict[typing.Text, typing.Union[dict[typing.Text, typing.Text],dict[typing.Text, int]]]]]]:
            return [{'$match': query.where}, {'$project': {'environment': 1, group_by: 1}}, {'$group': {'_id': {'environment': '$environment', group_by: '$' + group_by}, 'count': {'$sum': 1}}}, {'$limit': topn}]
        response_severity = self.get_db().alerts.aggregate(pipeline('severity'))
        severity_count = defaultdict(list)
        for r in response_severity:
            severity_count[r['_id']['environment']].append((r['_id']['severity'], r['count']))
        response_status = self.get_db().alerts.aggregate(pipeline('status'))
        status_count = defaultdict(list)
        for r in response_status:
            status_count[r['_id']['environment']].append((r['_id']['status'], r['count']))
        environments = self.get_db().alerts.find().distinct('environment')
        return [{'environment': env, 'severityCounts': dict(severity_count[env]), 'statusCounts': dict(status_count[env]), 'count': sum((t[1] for t in severity_count[env]))} for env in environments]

    def get_services(self, query: Union[None, str, list[str], list[dict]]=None, topn: int=1000) -> list[dict[typing.Text, bool]]:
        query = query or Query()

        def pipeline(group_by: Any) -> list[typing.Union[dict[typing.Text, dict[typing.Text, int]],dict[typing.Text, dict[typing.Text, typing.Union[dict[typing.Text, typing.Text],dict[typing.Text, int]]]]]]:
            return [{'$unwind': '$service'}, {'$match': query.where}, {'$project': {'environment': 1, 'service': 1, group_by: 1}}, {'$group': {'_id': {'environment': '$environment', 'service': '$service', group_by: '$' + group_by}, 'count': {'$sum': 1}}}, {'$limit': topn}]
        response_severity = self.get_db().alerts.aggregate(pipeline('severity'))
        severity_count = defaultdict(list)
        for r in response_severity:
            severity_count[r['_id']['environment'], r['_id']['service']].append((r['_id']['severity'], r['count']))
        response_status = self.get_db().alerts.aggregate(pipeline('status'))
        status_count = defaultdict(list)
        for r in response_status:
            status_count[r['_id']['environment'], r['_id']['service']].append((r['_id']['status'], r['count']))
        pipeline = [{'$unwind': '$service'}, {'$group': {'_id': {'environment': '$environment', 'service': '$service'}}}, {'$limit': topn}]
        services = list(self.get_db().alerts.aggregate(pipeline))
        return [{'environment': svc['_id']['environment'], 'service': svc['_id']['service'], 'severityCounts': dict(severity_count[svc['_id']['environment'], svc['_id']['service']]), 'statusCounts': dict(status_count[svc['_id']['environment'], svc['_id']['service']]), 'count': sum((t[1] for t in severity_count[svc['_id']['environment'], svc['_id']['service']]))} for svc in services]

    def get_alert_groups(self, query: Union[None, str, alerta.database.base.Query, django.db.models.query.Q.uerySet]=None, topn: int=1000) -> list[dict[typing.Text, str]]:
        query = query or Query()
        pipeline = [{'$match': query.where}, {'$project': {'environment': 1, 'group': 1}}, {'$limit': topn}, {'$group': {'_id': {'environment': '$environment', 'group': '$group'}, 'count': {'$sum': 1}}}]
        responses = self.get_db().alerts.aggregate(pipeline)
        groups = list()
        for response in responses:
            groups.append({'environment': response['_id']['environment'], 'group': response['_id']['group'], 'count': response['count']})
        return groups

    def get_alert_tags(self, query: Union[None, str, alerta.database.base.Query, django.db.models.query.Q.uerySet]=None, topn: int=1000) -> list[dict[typing.Text, str]]:
        query = query or Query()
        pipeline = [{'$match': query.where}, {'$unwind': '$tags'}, {'$project': {'environment': 1, 'tags': 1}}, {'$limit': topn}, {'$group': {'_id': {'environment': '$environment', 'tag': '$tags'}, 'count': {'$sum': 1}}}]
        responses = self.get_db().alerts.aggregate(pipeline)
        tags = list()
        for response in responses:
            tags.append({'environment': response['_id']['environment'], 'tag': response['_id']['tag'], 'count': response['count']})
        return tags

    def create_blackout(self, blackout: Union[src.dbp.models.LocationTuple, None, int, str]) -> dict[typing.Text, ]:
        data = {'_id': blackout.id, 'priority': blackout.priority, 'environment': blackout.environment, 'startTime': blackout.start_time, 'endTime': blackout.end_time, 'duration': blackout.duration, 'user': blackout.user, 'createTime': blackout.create_time, 'text': blackout.text}
        if blackout.service:
            data['service'] = blackout.service
        if blackout.resource:
            data['resource'] = blackout.resource
        if blackout.event:
            data['event'] = blackout.event
        if blackout.group:
            data['group'] = blackout.group
        if blackout.tags:
            data['tags'] = blackout.tags
        if blackout.origin:
            data['origin'] = blackout.origin
        if blackout.customer:
            data['customer'] = blackout.customer
        if self.get_db().blackouts.insert_one(data).inserted_id == blackout.id:
            return data

    def get_blackout(self, id: Union[str, int], customers: Union[None, str]=None) -> Union[str, bool]:
        query = {'_id': id}
        if customers:
            query['customer'] = {'$in': customers}
        return self.get_db().blackouts.find_one(query)

    def get_blackouts(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, str]=None, page_size: Union[None, int, str]=None) -> Union[str, int, src.dbp.models.LocationTuple, None]:
        query = query or Query()
        return self.get_db().blackouts.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_blackouts_count(self, query: Union[None, alerta.database.base.Query, str]=None) -> Union[int, str]:
        query = query or Query()
        return self.get_db().blackouts.count_documents(query.where)

    def is_blackout_period(self, alert: Union[tuple[typing.Union[transfer.models.User,events.user.UserAccountCreated]], models.User, dict[str, typing.Any]]) -> bool:
        query = dict()
        query['startTime'] = {'$lte': alert.create_time}
        query['endTime'] = {'$gt': alert.create_time}
        query['environment'] = alert.environment
        query['$and'] = [{'$or': [{'resource': None, 'service': None, 'event': None, 'group': None, 'tags': None, 'origin': None}, {'resource': None, 'service': None, 'event': None, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': None, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': None, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': None, 'service': None, 'event': None, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': None, 'service': None, 'event': None, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': None, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': None, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': None, 'service': None, 'event': alert.event, 'group': None, 'tags': None, 'origin': None}, {'resource': None, 'service': None, 'event': alert.event, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': None, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': None, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': None, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': None, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': None, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': None, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': None, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': None, 'group': None, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': None, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': None, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': None, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': None, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': None, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': None, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': None, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': None, 'origin': alert.origin}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': None}, {'resource': alert.resource, 'service': {'$not': {'$elemMatch': {'$nin': alert.service}}}, 'event': alert.event, 'group': alert.group, 'tags': {'$not': {'$elemMatch': {'$nin': alert.tags}}}, 'origin': alert.origin}]}]
        if current_app.config['CUSTOMER_VIEWS']:
            query['$and'].append({'$or': [{'customer': None}, {'customer': alert.customer}]})
        if self.get_db().blackouts.find_one(query):
            return True
        return False

    def update_blackout(self, id: Union[int, str, list[str]], **kwargs) -> Union[bool, str, None]:
        return self.get_db().blackouts.find_one_and_update({'_id': id}, update={'$set': kwargs}, return_document=ReturnDocument.AFTER)

    def delete_blackout(self, id: Union[str, int]) -> bool:
        response = self.get_db().blackouts.delete_one({'_id': id})
        return True if response.deleted_count == 1 else False

    def upsert_heartbeat(self, heartbeat: Union[int, dict[str, int]]) -> Union[bool, list[str], None, tuple[list[typing.Any]]]:
        return self.get_db().heartbeats.find_one_and_update({'origin': heartbeat.origin, 'customer': heartbeat.customer}, {'$setOnInsert': {'_id': heartbeat.id}, '$set': {'origin': heartbeat.origin, 'tags': heartbeat.tags, 'attributes': heartbeat.attributes, 'type': heartbeat.event_type, 'createTime': heartbeat.create_time, 'timeout': heartbeat.timeout, 'receiveTime': heartbeat.receive_time, 'customer': heartbeat.customer}}, upsert=True, return_document=ReturnDocument.AFTER)

    def get_heartbeat(self, id: str, customers: Union[None, str, list[int], int]=None) -> Union[str, None, list[str]]:
        if len(id) == 8:
            query = {'_id': {'$regex': '^' + id}}
        else:
            query = {'_id': id}
        if customers:
            query['customer'] = {'$in': customers}
        return self.get_db().heartbeats.find_one(query)

    def get_heartbeats(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, str]=None, page_size: Union[None, int, str]=None) -> Union[bool, list[str]]:
        query = query or Query()
        return self.get_db().heartbeats.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_heartbeats_by_status(self, status: Union[None, int, str]=None, query: Union[None, int, alerta.database.base.Query]=None, page: int=None, page_size: Union[int, list[str]]=None) -> Union[models.GithubWebhookEvent, typing.Callable[typing.Any, None], None]:
        status = status or list()
        query = query or Query()
        max_latency = current_app.config['HEARTBEAT_MAX_LATENCY']
        pipeline = [{'$match': query.where}]
        if status:
            pipeline.extend([{'$addFields': {'timeoutInMs': {'$multiply': ['$timeout', 1000]}}}, {'$addFields': {'isExpired': {'$gt': [{'$subtract': [datetime.utcnow(), '$receiveTime']}, '$timeoutInMs']}}}, {'$addFields': {'isSlow': {'$gt': [{'$subtract': ['$receiveTime', '$createTime']}, max_latency]}}}])
            match_or = list()
            if HeartbeatStatus.OK in status:
                match_or.append({'isExpired': False, 'isSlow': False})
            if HeartbeatStatus.Expired in status:
                match_or.append({'isExpired': True})
            if HeartbeatStatus.Slow in status:
                match_or.append({'isExpired': False, 'isSlow': True})
            pipeline.append({'$match': {'$or': match_or}})
        pipeline.extend([{'$sort': {k: v for k, v in query.sort}}, {'$skip': (page - 1) * page_size}, {'$limit': page_size}])
        return self.get_db().heartbeats.aggregate(pipeline)

    def get_heartbeats_count(self, query: Union[None, alerta.database.base.Query, str]=None) -> int:
        query = query or Query()
        return self.get_db().heartbeats.count_documents(query.where)

    def delete_heartbeat(self, id: str) -> bool:
        response = self.get_db().heartbeats.delete_one({'_id': {'$regex': '^' + id}})
        return True if response.deleted_count == 1 else False

    def create_key(self, key: Union[str, KT, bytes]) -> dict[typing.Text, ]:
        data = {'_id': key.id, 'key': key.key, 'user': key.user, 'scopes': key.scopes, 'text': key.text, 'expireTime': key.expire_time, 'count': key.count, 'lastUsedTime': key.last_used_time}
        if key.customer:
            data['customer'] = key.customer
        if self.get_db().keys.insert_one(data).inserted_id == key.id:
            return data

    def get_key(self, key: Union[str, dict[str, typing.Any]], user: Union[None, core.models.UserKey, str]=None) -> Union[str, datetime.datetime, int]:
        query = {'$or': [{'key': key}, {'_id': key}]}
        if user:
            query['user'] = user
        return self.get_db().keys.find_one(query)

    def get_keys(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, str]=None, page_size: Union[None, int, str]=None) -> Union[bool, str, zerver.models.Realm]:
        query = query or Query()
        return self.get_db().keys.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_keys_by_user(self, user: Union[str, bytes, server.models.User]) -> Union[str, bool, zerver.models.DefaultStreamGroup]:
        return self.get_db().keys.find({'user': user})

    def get_keys_count(self, query: Union[None, alerta.database.base.Query, str]=None) -> Union[int, str]:
        query = query or Query()
        return self.get_db().keys.count_documents(query.where)

    def update_key(self, key: Union[str, int, KT], **kwargs) -> Union[bool, typing.Any, str, None]:
        return self.get_db().keys.find_one_and_update({'$or': [{'key': key}, {'_id': key}]}, update={'$set': kwargs}, return_document=ReturnDocument.AFTER)

    def update_key_last_used(self, key: Union[str, KT]) -> bool:
        return self.get_db().keys.update_one({'$or': [{'key': key}, {'_id': key}]}, {'$set': {'lastUsedTime': datetime.utcnow()}, '$inc': {'count': 1}}).matched_count == 1

    def delete_key(self, key: str) -> bool:
        query = {'$or': [{'key': key}, {'_id': key}]}
        response = self.get_db().keys.delete_one(query)
        return True if response.deleted_count == 1 else False

    def create_user(self, user: Union[services.user.transfer.models.User, int]) -> dict[typing.Text, ]:
        data = {'_id': user.id, 'name': user.name, 'login': user.login, 'password': user.password, 'email': user.email, 'status': user.status, 'roles': user.roles, 'attributes': user.attributes, 'createTime': user.create_time, 'lastLogin': user.last_login, 'text': user.text, 'updateTime': user.update_time, 'email_verified': user.email_verified}
        if self.get_db().users.insert_one(data).inserted_id == user.id:
            return data

    def get_user(self, id: Union[str, None, int]) -> Union[str, zerver.models.Realm, models.characters.character_base.Character]:
        query = {'_id': id}
        return self.get_db().users.find_one(query)

    def get_users(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, alerta.database.base.Query]=None, page_size: Union[None, int, alerta.database.base.Query]=None) -> Union[bool, str, datetime.datetime]:
        query = query or Query()
        return self.get_db().users.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_users_count(self, query: Union[None, alerta.database.base.Query, str, list['Query']]=None) -> Union[int, str, set[int]]:
        query = query or Query()
        return self.get_db().users.count_documents(query.where)

    def get_user_by_username(self, username: Union[str, cmk.utils.type_defs.UserId, None]) -> Union[None, str, asgard.backends.base.Orchestrator, asgard.models.user.User]:
        if not username:
            return
        query = {'$or': [{'login': username}, {'email': username}]}
        return self.get_db().users.find_one(query)

    def get_user_by_email(self, email: str) -> Union[None, str, asgard.backends.base.Orchestrator, asgard.models.user.User]:
        if not email:
            return
        query = {'email': email}
        return self.get_db().users.find_one(query)

    def get_user_by_hash(self, hash: Union[str, bytes]) -> Union[str, models.characters.character_base.Character, zerver.models.UserProfile]:
        query = {'hash': hash}
        return self.get_db().users.find_one(query)

    def update_last_login(self, id: Union[int, str, None]) -> bool:
        return self.get_db().users.update_one({'_id': id}, update={'$set': {'lastLogin': datetime.utcnow()}}).matched_count == 1

    def update_user(self, id: str, **kwargs) -> Union[bool, str]:
        update = dict()
        if 'attributes' in kwargs:
            update['$set'] = {k: v for k, v in kwargs.items() if k != 'attributes'}
            set_value = {'attributes.' + k: v for k, v in kwargs['attributes'].items() if v is not None}
            if set_value:
                update['$set'].update(set_value)
            unset_value = {'attributes.' + k: v for k, v in kwargs['attributes'].items() if v is None}
            if unset_value:
                update['$unset'] = unset_value
        else:
            update['$set'] = kwargs
        return self.get_db().users.find_one_and_update({'_id': {'$regex': '^' + id}}, update=update, return_document=ReturnDocument.AFTER)

    def update_user_attributes(self, id: Union[NameComplex, str, sqlalchemy.orm.Query], old_attrs: dict, new_attrs: Union[dict, nativecards.lib.dicts.models.DictionaryEntry, None, str]) -> bool:
        """
        Set all attributes and unset attributes by using a value of 'null'.
        """
        from alerta.utils.collections import merge
        merge(old_attrs, new_attrs)
        attrs = {k: v for k, v in old_attrs.items() if v is not None}
        update = {'$set': {'attributes': attrs}}
        response = self.get_db().users.update_one({'_id': {'$regex': '^' + id}}, update=update)
        return response.matched_count > 0

    def delete_user(self, id: Union[str, int]) -> bool:
        response = self.get_db().users.delete_one({'_id': id})
        return True if response.deleted_count == 1 else False

    def set_email_hash(self, id: Union[str, int, dict], hash: Union[str, int, dict]) -> bool:
        return self.get_db().users.update_one({'_id': id}, update={'$set': {'hash': hash, 'updateTime': datetime.utcnow()}}).matched_count == 1

    def create_group(self, group: Union[str, core.models.Grouping, psef.models.GroupSet, None]) -> dict[typing.Text, ]:
        data = {'_id': group.id, 'name': group.name, 'text': group.text}
        if self.get_db().groups.insert_one(data).inserted_id == group.id:
            return data

    def get_group(self, id: Union[str, None, int]) -> Union[str, bool]:
        query = {'_id': id}
        return self.get_db().groups.find_one(query)

    def get_groups(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, alerta.database.base.Query]=None, page_size: Union[None, int, alerta.database.base.Query]=None) -> Union[str, bool]:
        query = query or Query()
        return self.get_db().groups.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_groups_count(self, query: Union[None, alerta.database.base.Query, str, list['Query']]=None) -> Union[int, transfer.models.SiteID, str]:
        query = query or Query()
        return self.get_db().groups.count_documents(query.where)

    def get_group_users(self, id: Union[str, int]) -> list[dict[typing.Text, str]]:
        pipeline = [{'$match': {'_id': id}}, {'$unwind': '$users'}, {'$lookup': {'from': 'users', 'localField': 'users', 'foreignField': '_id', 'as': 'groupUser'}}, {'$project': {'groupUser': 1}}]
        responses = self.get_db().groups.aggregate(pipeline)
        users = list()
        for response in responses:
            users.append({'id': response['groupUser'][0]['_id'], 'login': response['groupUser'][0].get('login'), 'email': response['groupUser'][0]['email'], 'name': response['groupUser'][0]['name'], 'status': response['groupUser'][0]['status']})
        return users

    def update_group(self, id: Union[str, list, list[str]], **kwargs) -> Union[bool, None, str]:
        return self.get_db().groups.find_one_and_update({'_id': id}, update={'$set': kwargs}, return_document=ReturnDocument.AFTER)

    def add_user_to_group(self, group: Union[tracim.models.User, models.characters.character_base.Character], user: Union[tracim.models.User, models.characters.character_base.Character]) -> bool:
        response = self.get_db().groups.update_one({'_id': group}, {'$addToSet': {'users': user}})
        return response.matched_count > 0

    def remove_user_from_group(self, group: tracim.models.User, user: tracim.models.User) -> bool:
        response = self.get_db().groups.update_one({'_id': group}, {'$pullAll': {'users': [user]}})
        return response.matched_count > 0

    def delete_group(self, id: Union[str, int]) -> bool:
        response = self.get_db().groups.delete_one({'_id': id})
        return True if response.deleted_count == 1 else False

    def get_groups_by_user(self, user: Union[str, zerver.models.UserProfile, users.models.JustfixUser]) -> Union[str, models.characters.character_base.Character]:
        return self.get_db().groups.find({'users': user})

    def create_perm(self, perm: Union[int, list[str], list]) -> dict[typing.Text, ]:
        data = {'_id': perm.id, 'match': perm.match, 'scopes': perm.scopes}
        if self.get_db().perms.insert_one(data).inserted_id == perm.id:
            return data

    def get_perm(self, id: Union[str, None, int]) -> Union[str, zerver.models.UserProfile, bool]:
        query = {'_id': id}
        return self.get_db().perms.find_one(query)

    def get_perms(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, alerta.database.base.Query]=None, page_size: Union[None, int, alerta.database.base.Query]=None) -> Union[bool, str]:
        query = query or Query()
        return self.get_db().perms.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_perms_count(self, query: Union[None, alerta.database.base.Query, str]=None) -> Union[int, str]:
        query = query or Query()
        return self.get_db().perms.count_documents(query.where)

    def update_perm(self, id: Union[str, list, list[str]], **kwargs) -> Union[bool, list[str], None, str]:
        return self.get_db().perms.find_one_and_update({'_id': id}, update={'$set': kwargs}, return_document=ReturnDocument.AFTER)

    def delete_perm(self, id: str) -> bool:
        response = self.get_db().perms.delete_one({'_id': id})
        return True if response.deleted_count == 1 else False

    def get_scopes_by_match(self, login: Union[str, flask.Flask, bool], matches: django.db.models.Model) -> list[str]:
        if login in current_app.config['ADMIN_USERS']:
            return ADMIN_SCOPES
        scopes = list()
        for match in matches:
            if match in current_app.config['ADMIN_ROLES']:
                return ADMIN_SCOPES
            if match in current_app.config['USER_ROLES']:
                scopes.extend(current_app.config['USER_DEFAULT_SCOPES'])
            if match in current_app.config['GUEST_ROLES']:
                scopes.extend(current_app.config['GUEST_DEFAULT_SCOPES'])
            response = self.get_db().perms.find_one({'match': match}, projection={'scopes': 1, '_id': 0})
            if response:
                scopes.extend(response['scopes'])
        return sorted(set(scopes))

    def create_customer(self, customer: Union[salon.models.Stylist, models.tickeTicket]) -> dict[typing.Text, ]:
        data = {'_id': customer.id, 'match': customer.match, 'customer': customer.customer}
        if self.get_db().customers.insert_one(data).inserted_id == customer.id:
            return data

    def get_customer(self, id: Union[str, None, int]) -> Union[str, None, asgard.backends.base.Orchestrator]:
        query = {'_id': id}
        return self.get_db().customers.find_one(query)

    def get_customers(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, str]=None, page_size: Union[None, int, str]=None) -> Union[str, bool]:
        query = query or Query()
        return self.get_db().customers.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_customers_count(self, query: Union[None, alerta.database.base.Query, str]=None) -> Union[int, str, None]:
        query = query or Query()
        return self.get_db().customers.count_documents(query.where)

    def update_customer(self, id: Union[str, int], **kwargs) -> Union[bool, None]:
        return self.get_db().customers.find_one_and_update({'_id': id}, update={'$set': kwargs}, return_document=ReturnDocument.AFTER)

    def delete_customer(self, id: Union[str, int]) -> bool:
        response = self.get_db().customers.delete_one({'_id': id})
        return True if response.deleted_count == 1 else False

    def get_customers_by_match(self, login: Union[bool, list[str], metaswitch_tinder.database.User], matches: models.Dominion) -> Union[typing.Text, list]:
        if login in current_app.config['ADMIN_USERS']:
            return '*'
        customers = []
        for match in [login] + matches:
            for r in self.get_db().customers.find({'match': match}):
                customers.append(r['customer'])
        if customers:
            if '*' in customers:
                return '*'
            return customers
        raise NoCustomerMatch(f"No customer lookup configured for user '{login}' or '{','.join(matches)}'")

    def create_note(self, note: Union[base.OrderSpec, None, base.WhereSpec, list[pykechain.models.association.Association]]) -> dict[typing.Text, ]:
        data = {'_id': note.id, 'text': note.text, 'user': note.user, 'attributes': note.attributes, 'type': note.note_type, 'createTime': note.create_time, 'updateTime': note.update_time, 'alert': note.alert}
        if note.customer:
            data['customer'] = note.customer
        if self.get_db().notes.insert_one(data).inserted_id == note.id:
            return data

    def get_note(self, id: Union[str, None, int]) -> Union[str, None, bytes]:
        query = {'_id': id}
        return self.get_db().notes.find_one(query)

    def get_notes(self, query: Union[None, int, alerta.database.base.Query]=None, page: Union[None, int, str]=None, page_size: Union[None, int, str]=None) -> Union[str, int, typing.Iterable[str]]:
        query = query or Query()
        return self.get_db().notes.find(query.where, sort=query.sort).skip((page - 1) * page_size).limit(page_size)

    def get_alert_notes(self, id: str, page: Union[None, int, str]=None, page_size: Union[None, int, str]=None) -> Union[int, tuple[int]]:
        if len(id) == 8:
            query = {'alert': {'$regex': '^' + id}}
        else:
            query = {'alert': id}
        return self.get_db().notes.find(query).skip((page - 1) * page_size).limit(page_size)

    def get_customer_notes(self, customer: Union[int, None, alerta.database.base.Query], page: Union[None, int, alerta.database.base.Query]=None, page_size: Union[None, int, alerta.database.base.Query]=None) -> Union[bool, list[str], None, grouper.models.base.session.Session]:
        return self.get_db().notes.find({'customer': customer}).skip((page - 1) * page_size).limit(page_size)

    def update_note(self, id: Union[int, str, None], **kwargs) -> Union[bool, None]:
        kwargs['updateTime'] = datetime.utcnow()
        return self.get_db().notes.find_one_and_update({'_id': id}, update={'$set': kwargs}, return_document=ReturnDocument.AFTER)

    def delete_note(self, id: Union[int, str]) -> bool:
        response = self.get_db().notes.delete_one({'_id': id})
        return True if response.deleted_count == 1 else False

    def get_metrics(self, type: Union[None, str, typing.Type, T]=None) -> list:
        query = {'type': type} if type else {}
        return list(self.get_db().metrics.find(query, {'_id': 0}))

    def set_gauge(self, gauge: Union[str, None, dict[str, typing.Any]]) -> Union[bool, str]:
        return self.get_db().metrics.find_one_and_update({'group': gauge.group, 'name': gauge.name}, {'$set': {'group': gauge.group, 'name': gauge.name, 'title': gauge.title, 'description': gauge.description, 'value': gauge.value, 'type': 'gauge'}}, upsert=True, return_document=ReturnDocument.AFTER)['value']

    def inc_counter(self, counter: Union[typing.DefaultDict, typing.Counter]) -> Union[bool, list[str], None, str]:
        return self.get_db().metrics.find_one_and_update({'group': counter.group, 'name': counter.name}, {'$set': {'group': counter.group, 'name': counter.name, 'title': counter.title, 'description': counter.description, 'type': 'counter'}, '$inc': {'count': counter.count}}, upsert=True, return_document=ReturnDocument.AFTER)['count']

    def update_timer(self, timer: Union[float, int, property]) -> Union[bool, tuple[list[typing.Any]], str, None]:
        return self.get_db().metrics.find_one_and_update({'group': timer.group, 'name': timer.name}, {'$set': {'group': timer.group, 'name': timer.name, 'title': timer.title, 'description': timer.description, 'type': 'timer'}, '$inc': {'count': timer.count, 'totalTime': timer.total_time}}, upsert=True, return_document=ReturnDocument.AFTER)

    def get_expired(self, expired_threshold: Union[datetime.timedelta, float, int], info_threshold: Union[int, datetime.timedelta, float]) -> Union[str, list[str], None, int]:
        if expired_threshold:
            expired_seconds_ago = datetime.utcnow() - timedelta(seconds=expired_threshold)
            self.get_db().alerts.delete_many({'status': {'$in': ['closed', 'expired']}, 'lastReceiveTime': {'$lt': expired_seconds_ago}})
        if info_threshold:
            info_seconds_ago = datetime.utcnow() - timedelta(seconds=info_threshold)
            self.get_db().alerts.delete_many({'severity': alarm_model.DEFAULT_INFORM_SEVERITY, 'lastReceiveTime': {'$lt': info_seconds_ago}})
        pipeline = [{'$match': {'status': {'$nin': ['expired']}}}, {'$addFields': {'computedTimeout': {'$multiply': [{'$ifNull': ['$timeout', current_app.config['ALERT_TIMEOUT']]}, 1000]}}}, {'$addFields': {'isExpired': {'$lt': [{'$add': ['$lastReceiveTime', '$computedTimeout']}, datetime.utcnow()]}}}, {'$match': {'isExpired': True, 'computedTimeout': {'$ne': 0}}}]
        return self.get_db().alerts.aggregate(pipeline)

    def get_unshelve(self) -> Union[bool, dict[str, typing.Any], list[str]]:
        pipeline = [{'$match': {'status': 'shelved'}}, {'$unwind': '$history'}, {'$match': {'history.type': 'shelve', 'history.status': 'shelved'}}, {'$sort': {'history.updateTime': -1}}, {'$group': {'_id': '$_id', 'resource': {'$first': '$resource'}, 'event': {'$first': '$event'}, 'environment': {'$first': '$environment'}, 'severity': {'$first': '$severity'}, 'correlate': {'$first': '$correlate'}, 'status': {'$first': '$status'}, 'service': {'$first': '$service'}, 'group': {'$first': '$group'}, 'value': {'$first': '$value'}, 'text': {'$first': '$text'}, 'tags': {'$first': '$tags'}, 'attributes': {'$first': '$attributes'}, 'origin': {'$first': '$origin'}, 'type': {'$first': '$type'}, 'createTime': {'$first': '$createTime'}, 'timeout': {'$first': '$timeout'}, 'rawData': {'$first': '$rawData'}, 'customer': {'$first': '$customer'}, 'duplicateCount': {'$first': '$duplicateCount'}, 'repeat': {'$first': '$repeat'}, 'previousSeverity': {'$first': '$previousSeverity'}, 'trendIndication': {'$first': '$trendIndication'}, 'receiveTime': {'$first': '$receiveTime'}, 'lastReceiveId': {'$first': '$lastReceiveId'}, 'lastReceiveTime': {'$first': '$lastReceiveTime'}, 'updateTime': {'$first': '$updateTime'}, 'history': {'$first': '$history'}}}, {'$addFields': {'computedTimeout': {'$multiply': [{'$ifNull': ['$history.timeout', current_app.config['SHELVE_TIMEOUT']]}, 1000]}}}, {'$addFields': {'isExpired': {'$lt': [{'$add': ['$updateTime', '$computedTimeout']}, datetime.utcnow()]}}}, {'$match': {'isExpired': True, 'computedTimeout': {'$ne': 0}}}]
        return self.get_db().alerts.aggregate(pipeline)

    def get_unack(self) -> Union[bool, dict[str, typing.Any], list[str]]:
        pipeline = [{'$match': {'status': 'ack'}}, {'$unwind': '$history'}, {'$match': {'history.type': 'ack', 'history.status': 'ack'}}, {'$sort': {'history.updateTime': -1}}, {'$group': {'_id': '$_id', 'resource': {'$first': '$resource'}, 'event': {'$first': '$event'}, 'environment': {'$first': '$environment'}, 'severity': {'$first': '$severity'}, 'correlate': {'$first': '$correlate'}, 'status': {'$first': '$status'}, 'service': {'$first': '$service'}, 'group': {'$first': '$group'}, 'value': {'$first': '$value'}, 'text': {'$first': '$text'}, 'tags': {'$first': '$tags'}, 'attributes': {'$first': '$attributes'}, 'origin': {'$first': '$origin'}, 'type': {'$first': '$type'}, 'createTime': {'$first': '$createTime'}, 'timeout': {'$first': '$timeout'}, 'rawData': {'$first': '$rawData'}, 'customer': {'$first': '$customer'}, 'duplicateCount': {'$first': '$duplicateCount'}, 'repeat': {'$first': '$repeat'}, 'previousSeverity': {'$first': '$previousSeverity'}, 'trendIndication': {'$first': '$trendIndication'}, 'receiveTime': {'$first': '$receiveTime'}, 'lastReceiveId': {'$first': '$lastReceiveId'}, 'lastReceiveTime': {'$first': '$lastReceiveTime'}, 'updateTime': {'$first': '$updateTime'}, 'history': {'$first': '$history'}}}, {'$addFields': {'computedTimeout': {'$multiply': [{'$ifNull': ['$history.timeout', current_app.config['ACK_TIMEOUT']]}, 1000]}}}, {'$addFields': {'isExpired': {'$lt': [{'$add': ['$updateTime', '$computedTimeout']}, datetime.utcnow()]}}}, {'$match': {'isExpired': True, 'computedTimeout': {'$ne': 0}}}]
        return self.get_db().alerts.aggregate(pipeline)