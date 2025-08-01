import json
import os
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, cast

from alerta.app import create_app, db, plugins
from alerta.exceptions import BlackoutPeriod
from alerta.models.key import ApiKey
from alerta.plugins import PluginBase
from alerta.utils.format import DateTime

class BlackoutsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        test_config: Dict[str, Any] = {'TESTING': True, 'AUTH_REQUIRED': True, 'CUSTOMER_VIEWS': True, 'PLUGINS': []}
        self.app = create_app(test_config)
        self.client = self.app.test_client()
        self.prod_alert: Dict[str, Any] = {'resource': 'node404', 'event': 'node_down', 'environment': 'Production', 'severity': 'major', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'service': ['Core', 'Web', 'Network'], 'group': 'Network', 'tags': ['level=20', 'switch:off'], 'origin': 'foo/bar'}
        self.dev_alert: Dict[str, Any] = {'resource': 'node404', 'event': 'node_marginal', 'environment': 'Development', 'severity': 'warning', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'service': ['Core', 'Web', 'Network'], 'group': 'Network', 'tags': ['level=20', 'switch:off'], 'origin': 'foo/bar'}
        self.fatal_alert: Dict[str, Any] = {'event': 'node_down', 'resource': 'net01', 'environment': 'Production', 'service': ['Network'], 'severity': 'critical', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'tags': ['foo'], 'attributes': {'foo': 'abc def', 'bar': 1234, 'baz': False}, 'origin': 'foo/bar'}
        self.critical_alert: Dict[str, Any] = {'event': 'node_marginal', 'resource': 'net02', 'environment': 'Production', 'service': ['Network'], 'severity': 'critical', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'origin': 'foo/bar', 'timeout': 30}
        self.major_alert: Dict[str, Any] = {'event': 'node_marginal', 'resource': 'net03', 'environment': 'Production', 'service': ['Network'], 'severity': 'major', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'origin': 'foo/bar', 'timeout': 40}
        self.normal_alert: Dict[str, Any] = {'event': 'node_up', 'resource': 'net03', 'environment': 'Production', 'service': ['Network'], 'severity': 'normal', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'origin': 'foo/quux', 'timeout': 100}
        self.minor_alert: Dict[str, Any] = {'event': 'node_marginal', 'resource': 'net04', 'environment': 'Production', 'service': ['Network'], 'severity': 'minor', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'origin': 'foo/quux', 'timeout': 40}
        self.ok_alert: Dict[str, Any] = {'event': 'node_up', 'resource': 'net04', 'environment': 'Production', 'service': ['Network'], 'severity': 'ok', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'origin': 'foo/quux', 'timeout': 100}
        self.warn_alert: Dict[str, Any] = {'event': 'node_marginal', 'resource': 'net05', 'environment': 'Production', 'service': ['Network'], 'severity': 'warning', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'origin': 'foo/quux', 'timeout': 50}
        with self.app.test_request_context('/'):
            self.app.preprocess_request()
            self.admin_api_key = ApiKey(user='admin@alerta.io', scopes=['admin', 'read', 'write'], text='demo-key')
            self.customer_api_key = ApiKey(user='admin@alerta.io', scopes=['admin', 'read', 'write'], text='demo-key', customer='Foo')
            self.admin_api_key.create()
            self.customer_api_key.create()

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_suppress_blackout(self) -> None:
        os.environ['NOTIFICATION_BLACKOUT'] = 'False'
        plugins.plugins['blackout'] = Blackout()
        self.headers: Dict[str, str] = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        response = self.client.post('/blackout', data=json.dumps({'environment': 'Production'}), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        self.headers = {'Authorization': f'Key {self.customer_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)

    def test_notification_blackout(self) -> None:
        os.environ['NOTIFICATION_BLACKOUT'] = 'True'
        plugins.plugins['blackout'] = Blackout()
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        blackout: Dict[str, Any] = {'environment': 'Production', 'service': ['Core']}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        self.prod_alert['severity'] = 'major'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        self.prod_alert['severity'] = 'critical'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        self.prod_alert['severity'] = 'minor'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        self.prod_alert['severity'] = 'warning'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        self.prod_alert['severity'] = 'ok'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.prod_alert['severity'] = 'major'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        self.prod_alert['severity'] = 'minor'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.prod_alert['severity'] = 'minor'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.prod_alert['severity'] = 'ok'
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')

    def test_previous_status(self) -> None:
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'open')
        alert_id_1 = data['id']
        response = self.client.put('/alert/' + alert_id_1 + '/action', data=json.dumps({'action': 'ack'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id_1, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'ack')
        response = self.client.post('/alert', data=json.dumps(self.critical_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'open')
        alert_id_2 = data['id']
        response = self.client.put('/alert/' + alert_id_2 + '/action', data=json.dumps({'action': 'shelve'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id_2, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'shelved')
        os.environ['NOTIFICATION_BLACKOUT'] = 'yes'
        plugins.plugins['blackout'] = Blackout()
        blackout = {'environment': 'Production', 'service': ['Network']}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'blackout')
        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'major')
        self.assertEqual(data['alert']['status'], 'blackout')
        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'normal')
        self.assertEqual(data['alert']['status'], 'closed')
        response = self.client.post('/alert', data=json.dumps(self.minor_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'minor')
        self.assertEqual(data['alert']['status'], 'blackout')
        response = self.client.post('/alert', data=json.dumps(self.ok_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'ok')
        self.assertEqual(data['alert']['status'], 'closed')
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'warning')
        self.assertEqual(data['alert']['status'], 'blackout')
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'ack')
        response = self.client.post('/alert', data=json.dumps(self.critical_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'shelved')
        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'normal')
        self.assertEqual(data['alert']['status'], 'closed')
        response = self.client.post('/alert', data=json.dumps(self.minor_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'minor')
        self.assertEqual(data['alert']['status'], 'open')
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'warning')
        self.assertEqual(data['alert']['status'], 'open')

    def test_whole_environment_blackout(self) -> None:
        os.environ['NOTIFICATION_BLACKOUT'] = 'False'
        plugins.plugins['blackout'] = Blackout()
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout: Dict[str, Any] = {'environment': 'Development'}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)

    def test_combination_blackout(self) -> None:
        os.environ['NOTIFICATION_BLACKOUT'] = 'False'
        plugins.plugins['blackout'] = Blackout()
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout: Dict[str, Any] = {'environment': 'Production', 'resource': 'node404', 'service': ['Network', 'Web']}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout = {'environment': 'Production', 'group': 'Network', 'tags': ['system:web01', 'switch:off']}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.prod_alert['tags'].append('system:web01')
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout = {'environment': 'Development', 'resource': 'node404', 'tags': ['level=40']}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.dev_alert['tags'].append('level=40')
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)

    def test_origin_blackout(self) -> None:
        os.environ['NOTIFICATION_BLACKOUT'] = 'False'
        plugins.plugins['blackout'] = Blackout()
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout: Dict[str, Any] = {'environment': 'Production', 'origin': 'foo/bar'}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        response = self.client.post('/alert', data=json.dumps(self.minor_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout = {'environment': 'Production', 'tags': ['system:web01', 'switch:off'], 'origin': 'foo/bar'}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.prod_alert['tags'].append('system:web01')
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout = {'environment': 'Development', 'event': 'node_marginal', 'origin': 'foo/quux'}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.dev_alert['origin'] = 'foo/quux'
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        response = self.client.get('/blackouts', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.delete('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)

    def test_custom_notify(self) -> None:
        os.environ['NOTIFICATION_BLACKOUT'] = 'True'
        plugins.plugins['blackout'] = Blackout()
        plugins.plugins['notify'] = CustomNotify()
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        three_second_from_now = datetime.utcnow() + timedelta(seconds=3)
        blackout: Dict[str, Any] = {'environment': 'Production', 'service': ['Core'], 'endTime': three_second_from_now.strftime('%Y-%m-%dT%H:%M:%S.000Z')}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'blackout')
        alert_receive_time = data['alert']['receiveTime']
        time.sleep(5)
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['repeat'], True)
        self.assertEqual(data['alert']['receiveTime'], alert_receive_time)
        self.assertEqual(data['alert']['attributes']['is_blackout'], True)
        self.assertEqual(data['alert']['attributes']['is_suppressed'], False)
        self.assertEqual(data['alert']['attributes']['notify'], True)

    def test_edit_blackout(self) -> None:
        os.environ['NOTIFICATION_BLACKOUT'] = 'False'
        plugins.plugins['blackout'] = Blackout()
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        blackout: Dict[str, Any] = {'environment': 'Production', 'resource': 'node404', 'service': ['Network', 'Web'], 'startTime': '2019-01-01T00:00:00.000Z', 'endTime': '2049-12-31T23:59:59.999Z'}
        response = self.client.post('/blackout', data=json.dumps(blackout), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id = data['id']
        response = self.client.post('/alert', data=json.dumps(self.prod_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)
        update: Dict[str, Any] = {'environment': 'Development', 'event': None, 'tags': [], 'endTime': '2099-12-31T23:59:59.999Z'}
        response = self.client.put('/blackout/' + blackout_id, data=json.dumps(update), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'ok')
        response = self.client.get('/blackout/' + blackout_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['blackout']['environment'], 'Development')
        self.assertEqual(data['blackout']['resource'], 'node404')
        self.assertEqual(data['blackout']['service'], ['Network', 'Web'])
        self.assertEqual(data['blackout']['group'], None)
        self.assertEqual(data['blackout']['startTime'], '2019-01-01T00:00:00.000Z')
        self.assertEqual(data['blackout']['endTime'], '2099-12-31T23:59:59.999Z')
        response = self.client.post('/alert', data=json.dumps(self.dev_alert), headers=self.headers)
        self.assertEqual(response.status_code, 202)

    def test_user_info(self) -> None:
        self.headers = {'Authorization': f'Key {self.admin_api_key.key}', 'Content-type': 'application/json'}
        response = self.client.post('/blackout', data=json.dumps({'environment': 'Production', 'service': ['Network'], 'text': 'administratively down'}), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['blackout']['user'], 'admin@alerta.io')
        self.assertIsInstance(DateTime.parse(data['blackout']['createTime']), datetime)
        self.assertEqual(data['blackout']['text'], 'administratively down')

class Blackout(PluginBase):

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        NOTIFICATION_BLACKOUT: bool = self.get_config('NOTIFICATION_BLACKOUT', default=True, type=bool, **kwargs)
        if alert.is_blackout():
            if NOTIFICATION_BLACKOUT:
                alert.status = 'blackout'
            else:
                raise BlackoutPeriod('Suppressed alert during blackout period')
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def status_change(self, alert: Any, status: str, text: str, **kwargs: Any) -> None:
        return

class CustomNotify(PluginBase):

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> Any:
        is_blackout: bool = alert.is_suppressed
        do_not_notify: str = os.environ['NOTIFICATION_BLACKOUT']
        if do_not_notify and is_blackout:
            alert.attributes['notify'] = False
        elif 'shelved' in alert.status:
            alert.attributes['notify'] = False
        elif do_not_notify and (not is_blackout):
            alert.attributes['notify'] = True
        alert.attributes['is_blackout'] = alert.is_blackout()
        alert.attributes['is_suppressed'] = alert.is_suppressed
        return alert

    def status_change(self, alert: Any, status: str, text: str, **kwargs: Any) -> None:
        return
