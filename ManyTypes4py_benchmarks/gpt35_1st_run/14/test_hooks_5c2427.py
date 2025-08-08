import json
import unittest
from datetime import datetime
from uuid import uuid4
from alerta.app import create_app, db, plugins
from alerta.plugins import PluginBase
resource: str = str(uuid4()).upper()[:8]

class PluginsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        test_config: dict = {'TESTING': True, 'AUTH_REQUIRED': False, 'PLUGINS': [], 'PLUGINS_RAISE_ON_ERROR': True}
        self.app = create_app(test_config)
        self.client = self.app.test_client()
        self.reject_alert: dict = {'id': '224040c5-5fdb-4d94-b564-398c755fdd02', 'event': 'node_marginal', 'resource': resource, 'environment': 'Production', 'service': [], 'severity': 'warning', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'tags': ['one', 'two']}
        self.accept_alert: dict = {'id': '82d8379c-5ea2-45fa-92e0-51c69c3048b9', 'event': 'node_marginal', 'resource': resource, 'environment': 'Production', 'service': ['Network'], 'severity': 'warning', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'tags': ['three', 'four']}
        self.critical_alert: dict = {'id': '5e2f6e2f-01f9-4a56-b9c1-a4d8a412b055', 'event': 'node_down', 'resource': resource, 'environment': 'Production', 'service': ['Network'], 'severity': 'critical', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'value': 'UP=0', 'text': 'node is down.', 'tags': ['cisco', 'core'], 'attributes': {'region': 'EU', 'site': 'london'}, 'origin': 'test_hooks.py', 'rawData': 'raw text'}
        self.headers: dict = {'Content-type': 'application/json'}

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_run_hooks(self) -> None:
        plugins.plugins['plugin1'] = Plugin1()
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        alert_id = data['id']
        response = self.client.put('/alert/' + alert_id + '/action', data=json.dumps({'action': 'ack', 'text': 'ack text'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        note: dict = {'text': 'this is a note'}
        response = self.client.put(f'/alert/{alert_id}/note', data=json.dumps(note), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        response = self.client.get('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.delete('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)

class Plugin1(unittest.TestCase, PluginBase):

    def pre_receive(self, alert, **kwargs) -> dict:
        ...

    def post_receive(self, alert, **kwargs) -> dict:
        ...

    def take_action(self, alert, action, text, **kwargs) -> tuple:
        ...

    def status_change(self, alert, status, text, **kwargs) -> tuple:
        ...

    def take_note(self, alert, text, **kwargs) -> tuple:
        ...

    def delete(self, alert, **kwargs) -> bool:
        ...
