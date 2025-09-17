from typing import Any, Dict, List
import json
import unittest
from datetime import datetime
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
        self.client = self.app.test_client()  # type: Any
        self.resource: str = str(uuid4()).upper()[:8]
        self.fatal_alert: Dict[str, Any] = {
            'event': 'node_down',
            'resource': self.resource,
            'environment': 'Production',
            'service': ['Network', 'Shared'],
            'severity': 'critical',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'tags': ['foo'],
            'attributes': {'foo': 'abc def', 'bar': 1234, 'baz': False}
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
        self.headers: Dict[str, str] = {'Content-type': 'application/json', 'X-Forwarded-For': '10.0.0.1'}

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_alert(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['resource'], self.resource)
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['service'], ['Network', 'Shared'])
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        self.assertEqual(data['alert']['history'][0]['user'], None)
        alert_id: str = data['id']
        update_time: Any = data['alert']['updateTime']
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
        response: Any = self.client.get('/alert/doesnotexist')
        self.assertEqual(response.status_code, 404)

    def test_get_alerts(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        response = self.client.get('/alerts')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertGreater(data['total'], 0)
        response = self.client.delete('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)

    def test_alert_status(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        alert_id: str = data['id']
        update_time: Any = data['alert']['updateTime']
        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['updateTime'], update_time)
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'ack'}), headers=self.headers)
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
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'ack'}), headers=self.headers)
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
        response: Any = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
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
        response: Any = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        alert_id: str = data['id']
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'expired'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'expired')
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'expired'}), headers=self.headers)
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
        response: Any = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        alert_id: str = data['id']
        response = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'normal')
        self.assertEqual(data['alert']['status'], 'closed')
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['severity'], 'warning')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        response = self.client.put('/alert/' + alert_id + '/action', data=json.dumps({'action': 'close', 'text': 'operator action'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'normal')
        self.assertEqual(data['alert']['status'], 'closed')
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.assertEqual(data['alert']['severity'], 'warning')
        self.assertEqual(data['alert']['duplicateCount'], 0)

    def test_duplicate_status(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['duplicateCount'], 0)
        alert_id: str = data['id']
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'closed'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')

    def test_duplicate_value(self) -> None:
        self.fatal_alert['value'] = '100'
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.fatal_alert['value'] = '101'
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'open')
        self.fatal_alert['value'] = '102'
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual([h['value'] for h in data['alert']['history']], ['100', '101', '102'])

    def test_alert_tagging(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['tags'], ['foo'])
        alert_id: str = data['id']
        response = self.client.put('/alert/' + alert_id + '/tag', data=json.dumps({'tags': ['bar']}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(sorted(data['alert']['tags']), ['bar', 'foo'])
        response = self.client.put('/alert/' + alert_id + '/tag', data=json.dumps({'tags': ['bar']}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(sorted(data['alert']['tags']), ['bar', 'foo'])
        response = self.client.put('/alert/' + alert_id + '/untag', data=json.dumps({'tags': ['foo']}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['tags'], ['bar'])

    def test_alert_no_attributes(self) -> None:
        plugins.plugins['remote_ip'] = DummyRemoteIPPlugin()
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert_no_attributes), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes'], {})
        alert_id: str = data['id']
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'ack'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'ack')
        self.assertEqual(data['alert']['attributes'], {})
        response = self.client.put('/alert/' + alert_id + '/action', data=json.dumps({'action': 'close'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertEqual(data['alert']['attributes'], {})

    def test_alert_attributes(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(sorted(data['alert']['attributes']),
                         sorted({'foo': 'abc def', 'bar': 1234, 'baz': False, 'ip': '10.0.0.1'}))
        alert_id: str = data['id']
        response = self.client.put('/alert/' + alert_id + '/attributes', data=json.dumps({'attributes': {'quux': ['q', 'u', 'u', 'x'], 'bar': None}}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(sorted(data['alert']['attributes']),
                         sorted({'foo': 'abc def', 'baz': False, 'quux': ['q', 'u', 'u', 'x'], 'ip': '10.0.0.1'}))
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(sorted(data['alert']['attributes']),
                         sorted({'foo': 'abc def', 'bar': 1234, 'baz': False, 'quux': ['q', 'u', 'u', 'x'], 'ip': '10.0.0.1'}))
        response = self.client.put('/alert/' + alert_id + '/attributes', data=json.dumps({'attributes': {'quux': [1, 'u', 'u', 4]}}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(sorted(data['alert']['attributes']),
                         sorted({'foo': 'abc def', 'bar': 1234, 'baz': False, 'quux': [1, 'u', 'u', 4], 'ip': '10.0.0.1'}))
        response = self.client.post('/alert', data=json.dumps(self.critical_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(sorted(data['alert']['attributes']),
                         sorted({'foo': 'abc def', 'bar': 1234, 'baz': False, 'quux': [1, 'u', 'u', 4], 'ip': '10.0.0.1'}))

    def test_history_limit(self) -> None:
        self.fatal_alert['value'] = '100'
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        self.fatal_alert['value'] = '101'
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'ack'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.fatal_alert['value'] = '102'
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.major_alert['value'] = '99'
        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.fatal_alert['value'] = '104'
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        self.fatal_alert['value'] = '105'
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertListEqual([h['value'] for h in data['alert']['history']], ['101', '102', '99', '104', '105'])
        self.assertListEqual([h['type'] for h in data['alert']['history']], ['status', 'value', 'severity', 'severity', 'value'])

    def test_timeout(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['timeout'], 120)
        self.fatal_alert['timeout'] = 20
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['timeout'], 20)
        self.fatal_alert['timeout'] = 0
        response = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['timeout'], 0)
        response = self.client.post('/alert', data=json.dumps(self.major_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['timeout'], 40)
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['timeout'], 50)
        self.warn_alert['timeout'] = 60
        response = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['timeout'], 60)
        response = self.client.post('/alert', data=json.dumps(self.ok_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['timeout'], 120)

    def test_filter_params(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.fatal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        response = self.client.get('/alerts?service=Network')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total'], 1)
        response = self.client.get('/alerts?event=node_down&severity=critical')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['alerts'][0]['event'], 'node_down')
        response = self.client.get('/alerts?attributes.foo=abc def')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['alerts'][0]['event'], 'node_down')
        attributes: Dict[str, Any] = {'attributes': {'acked-by': 'Big X'}}
        response = self.client.put(f'/alert/{alert_id}/attributes', data=json.dumps(attributes), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alerts?attributes.acked-by=Big X')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['alerts'][0]['event'], 'node_down')

    def test_query_param(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        response = self.client.get('/alerts?q=event:node_up')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['alerts'][0]['event'], 'node_up')

    def test_filter_and_query_params(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.normal_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        response = self.client.get('/alerts?service=Network&q=event:node_up')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['alerts'][0]['event'], 'node_up')

    def test_alerts_show_fields(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.warn_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        response = self.client.get('/alerts?show-raw-data=no')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alerts'][0]['rawData'], None)
        self.assertEqual(data['alerts'][0]['history'], [])
        response = self.client.get('/alerts?show-raw-data=yes&show-history=0')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alerts'][0]['rawData'], 'command output')
        self.assertEqual(data['alerts'][0]['history'], [])
        response = self.client.get('/alerts?show-history=yes')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['alerts'][0]['rawData'], None)
        self.assertEqual(len(data['alerts'][0]['history']), 1)

    def test_get_body(self) -> None:
        from flask import g
        with self.app.test_request_context('/'):
            g.login = 'foo'
            alert_in: Alert = Alert(resource='test1', event='event1', environment='Development', service=['svc1', 'svc2'])
            self.assertTrue(isinstance(alert_in.create_time, datetime))
            self.assertEqual(alert_in.last_receive_time, None)
            self.assertTrue(isinstance(alert_in.receive_time, datetime))
            self.assertEqual(alert_in.update_time, None)
            body: Dict[str, Any] = alert_in.get_body()
            self.assertEqual(type(body['createTime']), str)
            self.assertEqual(body['lastReceiveTime'], None)
            self.assertEqual(type(body['receiveTime']), str)
            self.assertEqual(body['updateTime'], None)
            alert_out: Alert = process_alert(alert_in)
            self.assertTrue(isinstance(alert_out.create_time, datetime))
            self.assertTrue(isinstance(alert_out.last_receive_time, datetime))
            self.assertTrue(isinstance(alert_out.receive_time, datetime))
            self.assertTrue(isinstance(alert_out.update_time, datetime))
            body = alert_out.get_body()
            self.assertEqual(type(body['createTime']), str)
            self.assertEqual(type(body['lastReceiveTime']), str)
            self.assertEqual(type(body['receiveTime']), str)
            self.assertEqual(type(body['updateTime']), str)


class DummyRemoteIPPlugin(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Any:
        return (alert, status, text)