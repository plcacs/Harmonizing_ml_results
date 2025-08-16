import json
import unittest
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from alerta.app import alarm_model, create_app, db, plugins
from alerta.models.alert import Alert
from alerta.plugins import PluginBase
from alerta.utils.api import process_alert


class AlertsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        test_config: Dict[str, Any] = {
            'TESTING': True,
            'AUTH_REQUIRED': False,
            'ALERT_TIMEOUT': 120,
            'HISTORY_LIMIT': 5
        }
        self.app = create_app(test_config)
        self.client = self.app.test_client()

        self.resource: str = str(uuid4()).upper()[:8]

        self.fatal_alert: Dict[str, Any] = {
            'event': 'node_down',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network', 'Shared'],
            'severity': 'critical',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'tags': ['foo'],
            'attributes': {'foo': 'abc def', 'bar': 1234, 'baz': False},
        }
        self.fatal_alert_no_attributes: Dict[str, Any] = {
            'event': 'node_down',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network', 'Shared'],
            'severity': 'critical',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'tags': ['foo']
        }
        self.critical_alert: Dict[str, Any] = {
            'event': 'node_marginal',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network'],
            'severity': 'critical',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'timeout': 30
        }
        self.major_alert: Dict[str, Any] = {
            'event': 'node_marginal',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network', 'Shared'],
            'severity': 'major',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'timeout': 40
        }
        self.warn_alert: Dict[str, Any] = {
            'event': 'node_marginal',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network'],
            'severity': 'warning',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'timeout': 50,
            'rawData': 'command output'
        }
        self.normal_alert: Dict[str, Any] = {
            'event': 'node_up',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network'],
            'severity': 'normal',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'timeout': 100
        }

        self.ok_alert: Dict[str, Any] = {
            'event': 'node_up',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network'],
            'severity': 'ok',
            'correlate': ['node_down', 'node_marginal', 'node_up']
        }

        self.cleared_alert: Dict[str, Any] = {
            'event': 'node_up',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network'],
            'severity': 'cleared',
            'correlate': ['node_down', 'node_marginal', 'node_up']
        }

        self.ok2_alert: Dict[str, Any] = {
            'event': 'node_up',
            'resource': self.resource + '2',
            'environment': 'Production',
            'service': ['Network'],
            'severity': 'ok',
            'correlate': ['node_down', 'node_marginal', 'node_up']
        }

        self.headers: Dict[str, str] = {
            'Content-type': 'application/json',
            'X-Forwarded-For': '10.0.0.1'
        }

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_alert(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['resource'], self.resource)
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['service'], ['Network', 'Shared'])
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        self.assertEqual(data['alert']['history'][0]['user'], None)

        alert_id: str = data['id']
        update_time: str = data['alert']['updateTime']

        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['service'], ['Network', 'Shared'])
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['previousSeverity'], alarm_model.DEFAULT_PREVIOUS_SEVERITY)
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        self.assertEqual(data['alert']['updateTime'], update_time)

        response = self.client.post('/alert', data=json.dumps(self.critical_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['service'], ['Network'])
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['previousSeverity'], self.major_alert['severity'])
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        self.assertEqual(data['alert']['updateTime'], update_time)

        response = self.client.post('/alert', data=json.dumps(self.critical_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['service'], ['Network'])
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        self.assertEqual(data['alert']['updateTime'], update_time)

        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['service'], ['Network', 'Shared'])
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['previousSeverity'], self.critical_alert['severity'])
        self.assertEqual(data['alert']['trendIndication'], 'noChange')
        self.assertEqual(data['alert']['updateTime'], update_time)

        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['service'], ['Network', 'Shared'])
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['previousSeverity'], self.fatal_alert['severity'])
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        self.assertEqual(data['alert']['updateTime'], update_time)

        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertEqual(data['alert']['service'], ['Network'])
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['previousSeverity'], self.major_alert['severity'])
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        self.assertEqual(data['alert']['updateTime'], data['alert']['receiveTime'])

        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        response = self.client.delete('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)

    def test_alert_not_found(self) -> None:
        response = self.client.get('/alert/doesnotexist')
        self.assertEqual(response.status_code, 404)

    def test_get_alerts(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id: str = data['id']

        response = self.client.get('/alerts')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertGreater(data['total'], 0)

        response = self.client.delete('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)

    def test_alert_status(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')

        alert_id: str = data['id']
        update_time: str = data['alert']['updateTime']

        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['updateTime'], update_time)

        response = self.client.put('/alert/' + alert_id + '/status',
                                 data=json.dumps({'status': 'ack'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'ack')
        self.assertNotEqual(data['alert']['updateTime'], update_time)

        update_time = data['alert']['updateTime']

        response = self.client.post('/alert', data=json.dumps(self.critical_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.assertNotEqual(data['alert']['updateTime'], update_time)

        update_time = data['alert']['updateTime']

        response = self.client.put('/alert/' + alert_id + '/status',
                                 data=json.dumps({'status': 'ack'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'ack')
        self.assertNotEqual(data['alert']['updateTime'], update_time)

        update_time = data['alert']['updateTime']

        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'ack')
        self.assertEqual(data['alert']['updateTime'], update_time)

        update_time = data['alert']['updateTime']

        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertNotEqual(data['alert']['updateTime'], update_time)

        update_time = data['alert']['updateTime']

        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.assertNotEqual(data['alert']['updateTime'], update_time)

        update_time = data['alert']['updateTime']

        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertNotEqual(data['alert']['updateTime'], update_time)

        response = self.client.delete('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)

    def test_closed_alerts(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertEqual(data['alert']['severity'], 'normal')
        self.assertEqual(data['alert']['duplicateCount'], 0)

        response = self.client.post('/alert', data=json.dumps(self.ok_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertEqual(data['alert']['severity'], 'ok')
        self.assertEqual(data['alert']['duplicateCount'], 0)

        response = self.client.post('/alert', data=json.dumps(self.cleared_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertEqual(data['alert']['severity'], 'cleared')
        self.assertEqual(data['alert']['duplicateCount'], 0)

        response = self.client.post('/alert', data=json.dumps(self.cleared_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertEqual(data['alert']['severity'], 'cleared')
        self.assertEqual(data['alert']['duplicateCount'], 1)

    def test_expired_alerts(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')

        alert_id: str = data['id']

        response = self.client.put('/alert/' + alert_id + '/status',
                                 data=json.dumps({'status': 'expired'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'expired')

        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')

        response = self.client.put('/alert/' + alert_id + '/status',
                                 data=json.dumps({'status': 'expired'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'expired')

        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')

        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')

    def test_reopen_alerts(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')

        alert_id: str = data['id']

        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'normal')
        self.assertEqual(data['alert']['status'], 'closed')

        response = self.client.post('/alert', data