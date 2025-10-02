import json
import unittest
from typing import Any, Dict

from alerta.app import alarm_model, create_app, db, plugins
from alerta.plugins import PluginBase


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
            'correlate': [
                'FAILED_ALM',
                'HI_HI_ALM',
                'HI_ALM',
                'LO_ALM',
                'LO_LO_ALM',
                'ADVISE_ALM',
                'RST_ALM'
            ]
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
            'correlate': [
                'FAILED_ALM',
                'HI_HI_ALM',
                'HI_ALM',
                'LO_ALM',
                'LO_LO_ALM',
                'ADVISE_ALM',
                'RST_ALM'
            ]
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
            'correlate': [
                'FAILED_ALM',
                'HI_HI_ALM',
                'HI_ALM',
                'LO_ALM',
                'LO_LO_ALM',
                'ADVISE_ALM',
                'RST_ALM'
            ]
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
            'correlate': [
                'FAILED_ALM',
                'HI_HI_ALM',
                'HI_ALM',
                'LO_ALM',
                'LO_LO_ALM',
                'ADVISE_ALM',
                'RST_ALM'
            ]
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
            'correlate': [
                'FAILED_ALM',
                'HI_HI_ALM',
                'HI_ALM',
                'LO_ALM',
                'LO_LO_ALM',
                'ADVISE_ALM',
                'RST_ALM'
            ]
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
            'correlate': [
                'FAILED_ALM',
                'HI_HI_ALM',
                'HI_ALM',
                'LO_ALM',
                'LO_LO_ALM',
                'ADVISE_ALM',
                'RST_ALM'
            ]
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
            'correlate': [
                'FAILED_ALM',
                'HI_HI_ALM',
                'HI_ALM',
                'LO_ALM',
                'LO_LO_ALM',
                'ADVISE_ALM',
                'RST_ALM'
            ]
        }

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_ack_active_alarm(self) -> None:
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        response = self.client.post(
            '/alert',
            data=json.dumps(self.high_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '13')
        self.assertEqual(data['alert']['text'], 'High Alarm Limit 10')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(
            data['alert']['previousSeverity'],
            alarm_model.DEFAULT_PREVIOUS_SEVERITY
        )
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        data_ack: Dict[str, Any] = {'action': 'ack', 'text': 'operator ack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_ack),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.critical_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'Critical')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '19')
        self.assertEqual(data['alert']['text'], 'High High Alarm Limit 15')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        data_ack_critical: Dict[str, Any] = {'action': 'ack', 'text': 'operator ack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_ack_critical),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'Critical')
        self.assertEqual(data['alert']['status'], 'ACKED')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'RST_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '0')
        self.assertEqual(data['alert']['text'], 'OK Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'Critical')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')

    def test_rtn_before_ack(self) -> None:
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        response = self.client.post(
            '/alert',
            data=json.dumps(self.high_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '13')
        self.assertEqual(data['alert']['text'], 'High Alarm Limit 10')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(
            data['alert']['previousSeverity'],
            alarm_model.DEFAULT_PREVIOUS_SEVERITY
        )
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'RST_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'RTNUN')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '0')
        self.assertEqual(data['alert']['text'], 'OK Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.medium_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'LO_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'Medium')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '6')
        self.assertEqual(data['alert']['text'], 'Low Alarm Limit 5')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'OK')
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.medium_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'LO_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'Medium')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '6')
        self.assertEqual(data['alert']['text'], 'Low Alarm Limit 5')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['repeat'], True)
        self.assertEqual(data['alert']['previousSeverity'], 'OK')
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'RST_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'RTNUN')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '0')
        self.assertEqual(data['alert']['text'], 'OK Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'Medium')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        data_ack_final: Dict[str, Any] = {'action': 'ack', 'text': 'operator ack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_ack_final),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')

    def test_operator_unack(self) -> None:
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        response = self.client.post(
            '/alert',
            data=json.dumps(self.high_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '13')
        self.assertEqual(data['alert']['text'], 'High Alarm Limit 10')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(
            data['alert']['previousSeverity'],
            alarm_model.DEFAULT_PREVIOUS_SEVERITY
        )
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        data_ack: Dict[str, Any] = {'action': 'ack', 'text': 'operator ack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_ack),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')
        data_unack: Dict[str, Any] = {'action': 'unack', 'text': 'operator unack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_unack),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')
        data_ack_again: Dict[str, Any] = {'action': 'ack', 'text': 'operator ack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_ack_again),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')

    def test_operator_shelve(self) -> None:
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        response = self.client.post(
            '/alert',
            data=json.dumps(self.high_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '13')
        self.assertEqual(data['alert']['text'], 'High Alarm Limit 10')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(
            data['alert']['previousSeverity'],
            alarm_model.DEFAULT_PREVIOUS_SEVERITY
        )
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        data_ack: Dict[str, Any] = {'action': 'ack', 'text': 'operator ack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_ack),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')
        data_shelve: Dict[str, Any] = {'action': 'shelve', 'text': 'operator shelved'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_shelve),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'SHLVD')
        data_unshelve: Dict[str, Any] = {'action': 'unshelve', 'text': 'operator unshelved'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_unshelve),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')
        data_shelve_again: Dict[str, Any] = {'action': 'shelve', 'text': 'operator shelved'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_shelve_again),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'SHLVD')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'RST_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'SHLVD')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '0')
        self.assertEqual(data['alert']['text'], 'OK Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        data_unshelve_again: Dict[str, Any] = {'action': 'unshelve', 'text': 'operator unshelved'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_unshelve_again),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')
        data_shelve_third: Dict[str, Any] = {'action': 'shelve', 'text': 'operator shelved'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_shelve_third),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'RST_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '0')
        self.assertEqual(data['alert']['text'], 'OK Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['repeat'], True)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')

    def test_out_of_service(self) -> None:
        plugins.plugins['blackout'] = NotificationBlackout()
        blackout: Dict[str, Any] = {
            'environment': 'Production',
            'service': ['REACTORS']
        }
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        alert_id: str = data['id']
        response = self.client.post(
            '/alert',
            data=json.dumps(self.high_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '13')
        self.assertEqual(data['alert']['text'], 'High Alarm Limit 10')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(
            data['alert']['previousSeverity'],
            alarm_model.DEFAULT_PREVIOUS_SEVERITY
        )
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        data_ack: Dict[str, Any] = {'action': 'ack', 'text': 'operator ack'}
        response = self.client.put(
            f'/alert/{alert_id}/action',
            data=json.dumps(data_ack),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'ACKED')
        blackout_request: Dict[str, Any] = {
            'environment': 'Production',
            'service': ['REACTORS']
        }
        response = self.client.post(
            '/blackout',
            data=json.dumps(blackout_request),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id: str = data['id']
        response = self.client.post(
            '/alert',
            data=json.dumps(self.high_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'OOSRV')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '13')
        self.assertEqual(data['alert']['text'], 'High Alarm Limit 10')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['repeat'], True)
        self.assertEqual(
            data['alert']['previousSeverity'],
            alarm_model.DEFAULT_PREVIOUS_SEVERITY
        )
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.low_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'LO_LO_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'Low')
        self.assertEqual(data['alert']['status'], 'OOSRV')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '1')
        self.assertEqual(data['alert']['text'], 'Low Low Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        response = self.client.delete(
            f'/blackout/{blackout_id}',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'Low')
        self.assertEqual(data['alert']['status'], 'OOSRV')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.low_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'LO_LO_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'Low')
        self.assertEqual(data['alert']['status'], 'UNACK')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '1')
        self.assertEqual(data['alert']['text'], 'Low Low Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['repeat'], True)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        plugins.plugins['blackout'] = NotificationBlackout()
        blackout_new: Dict[str, Any] = {
            'environment': 'Production',
            'service': ['REACTORS'],
            'resource': 'LIC_101'
        }
        response = self.client.post(
            '/blackout',
            data=json.dumps(blackout_new),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        blackout_id_new: str = data['id']
        response = self.client.post(
            '/alert',
            data=json.dumps(self.high_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'HI_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'High')
        self.assertEqual(data['alert']['status'], 'OOSRV')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '13')
        self.assertEqual(data['alert']['text'], 'High Alarm Limit 10')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'Low')
        self.assertEqual(data['alert']['trendIndication'], 'moreSevere')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'RST_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'OOSRV')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '0')
        self.assertEqual(data['alert']['text'], 'OK Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 0)
        self.assertEqual(data['alert']['repeat'], False)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')
        response = self.client.delete(
            f'/blackout/{blackout_id_new}',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f'/alert/{alert_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'OOSRV')
        response = self.client.post(
            '/alert',
            data=json.dumps(self.ok_alarm),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn(alert_id, data['alert']['id'])
        self.assertEqual(data['alert']['resource'], 'LIC_101')
        self.assertEqual(data['alert']['event'], 'RST_ALM')
        self.assertEqual(data['alert']['environment'], 'Production')
        self.assertEqual(data['alert']['severity'], 'OK')
        self.assertEqual(data['alert']['status'], 'NORM')
        self.assertEqual(data['alert']['service'], ['REACTORS'])
        self.assertEqual(data['alert']['group'], 'PROCESS')
        self.assertEqual(data['alert']['value'], '0')
        self.assertEqual(data['alert']['text'], 'OK Alarm Limit 0')
        self.assertEqual(data['alert']['tags'], [])
        self.assertEqual(data['alert']['attributes'], {})
        self.assertEqual(data['alert']['origin'], 'PID1')
        self.assertEqual(data['alert']['type'], 'ALARM')
        self.assertEqual(data['alert']['duplicateCount'], 1)
        self.assertEqual(data['alert']['repeat'], True)
        self.assertEqual(data['alert']['previousSeverity'], 'High')
        self.assertEqual(data['alert']['trendIndication'], 'lessSevere')


class NotificationBlackout(PluginBase):

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        if alert.is_blackout():
            alert.status = 'OOSRV'
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def status_change(
        self,
        alert: Any,
        status: Any,
        text: Any,
        **kwargs: Any
    ) -> None:
        return
