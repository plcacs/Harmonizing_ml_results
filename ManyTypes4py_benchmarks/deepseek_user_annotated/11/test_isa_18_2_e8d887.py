import json
import unittest
from typing import Any, Dict, List, Optional

from alerta.app import alarm_model, create_app, db, plugins
from alerta.plugins import PluginBase
from alerta.models.alert import Alert


class Isa182TestCase(unittest.TestCase):

    def setUp(self) -> None:
        test_config: Dict[str, Any] = {
            'TESTING': True,
            'ALARM_MODEL': 'ISA_18_2',
            'AUTH_REQUIRED': False,
            'PLUGINS': [],
            'ALERT_TIMEOUT': 120,
            'HISTORY_LIMIT': 5
        }
        self.app = create_app(test_config, environment='development')
        self.client = self.app.test_client()

        self.fault_alarm: Dict[str, Any] = {
            'severity': 'Critical',
            'origin': 'LIC_101',
            'value': 'ERROR',
            'resource': 'LIC_101',
            'event': 'FAILED_ALM',
            'group': 'PROCESS',
            'text': 'Shutdown/Interlocked',
            'type': 'FAULT',
            'environment': 'Production',
            'service': ['REACTORS'],
            'correlate': ['FAILED_ALM', 'HI_HI_ALM', 'HI_ALM', 'LO_ALM', 'LO_LO_ALM', 'ADVISE_ALM', 'RST_ALM']
        }

        self.critical_alarm: Dict[str, Any] = {
            'severity': 'Critical',
            'origin': 'PID1',
            'value': '19',
            'resource': 'LIC_101',
            'event': 'HI_HI_ALM',
            'group': 'PROCESS',
            'text': 'High High Alarm Limit 15',
            'type': 'ALARM',
            'environment': 'Production',
            'service': ['REACTORS'],
            'correlate': ['FAILED_ALM', 'HI_HI_ALM', 'HI_ALM', 'LO_ALM', 'LO_LO_ALM', 'ADVISE_ALM', 'RST_ALM']
        }

        self.high_alarm: Dict[str, Any] = {
            'severity': 'High',
            'origin': 'PID1',
            'value': '13',
            'resource': 'LIC_101',
            'event': 'HI_ALM',
            'group': 'PROCESS',
            'text': 'High Alarm Limit 10',
            'type': 'ALARM',
            'environment': 'Production',
            'service': ['REACTORS'],
            'correlate': ['FAILED_ALM', 'HI_HI_ALM', 'HI_ALM', 'LO_ALM', 'LO_LO_ALM', 'ADVISE_ALM', 'RST_ALM']
        }

        self.medium_alarm: Dict[str, Any] = {
            'severity': 'Medium',
            'origin': 'PID1',
            'value': '6',
            'resource': 'LIC_101',
            'event': 'LO_ALM',
            'group': 'PROCESS',
            'text': 'Low Alarm Limit 5',
            'type': 'ALARM',
            'environment': 'Production',
            'service': ['REACTORS'],
            'correlate': ['FAILED_ALM', 'HI_HI_ALM', 'HI_ALM', 'LO_ALM', 'LO_LO_ALM', 'ADVISE_ALM', 'RST_ALM']
        }

        self.low_alarm: Dict[str, Any] = {
            'severity': 'Low',
            'origin': 'PID1',
            'value': '1',
            'resource': 'LIC_101',
            'event': 'LO_LO_ALM',
            'group': 'PROCESS',
            'text': 'Low Low Alarm Limit 0',
            'type': 'ALARM',
            'environment': 'Production',
            'service': ['REACTORS'],
            'correlate': ['FAILED_ALM', 'HI_HI_ALM', 'HI_ALM', 'LO_ALM', 'LO_LO_ALM', 'ADVISE_ALM', 'RST_ALM']
        }

        self.advisory_alarm: Dict[str, Any] = {
            'severity': 'Advisory',
            'origin': 'PID1',
            'value': '1',
            'resource': 'LIC_101',
            'event': 'ADVISE_ALM',
            'group': 'PROCESS',
            'text': 'Low Low Alarm Limit 0',
            'type': 'ALARM',
            'environment': 'Production',
            'service': ['REACTORS'],
            'correlate': ['FAILED_ALM', 'HI_HI_ALM', 'HI_ALM', 'LO_ALM', 'LO_LO_ALM', 'ADVISE_ALM', 'RST_ALM']
        }

        self.ok_alarm: Dict[str, Any] = {
            'severity': 'OK',
            'origin': 'PID1',
            'value': '0',
            'resource': 'LIC_101',
            'event': 'RST_ALM',
            'group': 'PROCESS',
            'text': 'OK Alarm Limit 0',
            'type': 'ALARM',
            'environment': 'Production',
            'service': ['REACTORS'],
            'correlate': ['FAILED_ALM', 'HI_HI_ALM', 'HI_ALM', 'LO_ALM', 'LO_LO_ALM', 'ADVISE_ALM', 'RST_ALM']
        }

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_ack_active_alarm(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id: str = data['id']

        response = self.client.post('/alert', data=json.dumps(self.high_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        data = {
            'action': 'ack',
            'text': 'operator ack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')

        response = self.client.post('/alert', data=json.dumps(self.critical_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        data = {
            'action': 'ack',
            'text': 'operator ack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'Critical')
        self.assertEqual(data['alert']['status'], 'ACKED')

        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')

    def test_rtn_before_ack(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id: str = data['id']

        response = self.client.post('/alert', data=json.dumps(self.high_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'RTNUN')

        response = self.client.post('/alert', data=json.dumps(self.medium_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        response = self.client.post('/alert', data=json.dumps(self.medium_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'RTNUN')

        data = {
            'action': 'ack',
            'text': 'operator ack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')

    def test_operator_unack(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id: str = data['id']

        response = self.client.post('/alert', data=json.dumps(self.high_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        data = {
            'action': 'ack',
            'text': 'operator ack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')

        data = {
            'action': 'unack',
            'text': 'operator unack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')

        data = {
            'action': 'ack',
            'text': 'operator ack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')

    def test_operator_shelve(self) -> None:
        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id: str = data['id']

        response = self.client.post('/alert', data=json.dumps(self.high_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        data = {
            'action': 'ack',
            'text': 'operator ack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')

        data = {
            'action': 'shelve',
            'text': 'operator shelved'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'SHLVD')

        data = {
            'action': 'unshelve',
            'text': 'operator unshelved'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')

        data = {
            'action': 'shelve',
            'text': 'operator shelved'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'SHLVD')

        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'SHLVD')

        data = {
            'action': 'unshelve',
            'text': 'operator unshelved'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')

    def test_out_of_service(self) -> None:
        plugins.plugins['blackout'] = NotificationBlackout()

        response = self.client.post('/alert', data=json.dumps(self.ok_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id: str = data['id']

        response = self.client.post('/alert', data=json.dumps(self.high_alarm), content_type='application/json')
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])

        data = {
            'action': 'ack',
            'text': 'operator ack'
        }
        response = self.client.put('/alert/' + alert_id + '/action',
                                   data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')

        plugins.plugins['blackout'] = NotificationBlackout()

        blackout: Dict[str, Any] = {
           