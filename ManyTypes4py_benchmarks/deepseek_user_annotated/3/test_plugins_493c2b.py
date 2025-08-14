import json
import os
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from alerta.app import create_app, db, plugins
from alerta.exceptions import AlertaException, InvalidAction
from alerta.models.alert import Alert
from alerta.models.enums import Status
from alerta.plugins import PluginBase


class PluginsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        test_config = {
            'TESTING': True,
            'AUTH_REQUIRED': False,
            'PLUGINS': ['remote_ip']
        }
        os.environ['ALLOWED_ENVIRONMENTS'] = 'Production,Staging,Development'

        self.app = create_app(test_config)
        self.client = self.app.test_client()

        self.resource = str(uuid4()).upper()[:8]

        self.reject_alert: Dict[str, Any] = {
            'event': 'node_marginal',
            'resource': self.resource,
            'environment': 'Staging',
            'service': [],  # alert will be rejected because service not defined
            'severity': 'warning',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'tags': ['one', 'two']
        }

        self.accept_alert: Dict[str, Any] = {
            'event': 'node_marginal',
            'resource': self.resource,
            'environment': 'Staging',
            'service': ['Network'],  # alert will be accepted because service defined
            'severity': 'warning',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'tags': ['three', 'four']
        }

        self.critical_alert: Dict[str, Any] = {
            'event': 'node_down',
            'resource': self.resource,
            'environment': 'Staging',
            'service': ['Network'],
            'severity': 'critical',
            'correlate': ['node_down', 'node_marginal', 'node_up'],
            'tags': []
        }

        self.coffee_alert: Dict[str, Any] = {
            'event': 'coffee_pot',
            'resource': self.resource,
            'environment': 'Staging',
            'service': ['Network'],
            'severity': 'critical',
            'text': 'coffee alert',
            'tags': []
        }

        self.headers: Dict[str, str] = {
            'Content-type': 'application/json'
        }

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_status_update(self) -> None:
        plugins.plugins['old1'] = OldPlugin1()
        plugins.plugins['test1'] = CustPlugin1()
        plugins.plugins['test2'] = CustPlugin2()
        plugins.plugins['test3'] = CustPlugin3()

        # create alert that will be accepted
        response = self.client.post('/alert', data=json.dumps(self.accept_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'ok')
        self.assertRegex(data['id'], '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
        self.assertEqual(data['alert']['attributes']['old'], 'post1')
        self.assertEqual(data['alert']['attributes']['aaa'], 'post1')

        alert_id = data['id']

        # ack alert
        response = self.client.put('/alert/' + alert_id + '/status',
                                   data=json.dumps({'status': 'ack', 'text': 'input'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(data['alert']['attributes']['old'], 'post1')
        self.assertEqual(data['alert']['attributes']['aaa'], 'post1')

        # alert status, tags, attributes and history text modified by plugin1 & plugin2
        self.assertEqual(data['alert']['status'], 'assign')
        self.assertEqual(sorted(data['alert']['tags']), sorted(
            ['Development', 'Production', 'Staging', 'more', 'other', 'that', 'the', 'this']))
        self.assertEqual(data['alert']['attributes']['foo'], 'bar')
        self.assertEqual(data['alert']['attributes']['baz'], 'quux')
        self.assertNotIn('abc', data['alert']['attributes'])
        self.assertEqual(data['alert']['attributes']['xyz'], 'down')
        self.assertEqual(data['alert']['history'][-1]['text'], 'input-plugin1-plugin3')

        del plugins.plugins['old1']
        del plugins.plugins['test1']
        del plugins.plugins['test2']
        del plugins.plugins['test3']

    def test_take_action(self) -> None:
        plugins.plugins['old1'] = OldPlugin1()
        plugins.plugins['test1'] = CustPlugin1()
        plugins.plugins['test2'] = CustPlugin2()
        plugins.plugins['test3'] = CustPlugin3()

        plugins.plugins['action1'] = CustActionPlugin1()

        # create alert
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['tags'], ['Production', 'Staging', 'Development'])
        self.assertEqual(data['alert']['attributes'], {'aaa': 'post1', 'ip': '127.0.0.1', 'old': 'post1'})

        alert_id = data['id']

        # create ticket for alert
        payload = {
            'action': 'createTicket',
            'text': 'ticket created by bob'
        }
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)

        # check status=assign
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'assign')
        self.assertEqual(sorted(data['alert']['tags']), sorted(
            ['Development', 'Production', 'Staging', 'more', 'other', 'that', 'the', 'this', 'aSingleTag', 'aDouble:Tag', 'a:Triple:Tag']))
        self.assertEqual(data['alert']['history'][1]['text'],
                         'ticket created by bob (ticket #12345)-plugin1-plugin3', data['alert']['history'])

        # update ticket for alert
        payload = {
            'action': 'updateTicket',
            'text': 'ticket updated by bob'
        }
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))

        # check no change in status, new alert text
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'assign')
        self.assertEqual(data['alert']['attributes']['up'], 'down')
        self.assertEqual(data['alert']['history'][2]['text'],
                         'ticket updated by bob (ticket #12345)-plugin1-plugin3', data['alert']['history'])

        # update ticket for alert
        payload = {
            'action': 'resolveTicket',
            'text': 'ticket resolved by bob'
        }
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)

        # check post_action (check if tag was added)
        response = self.client.put('/alert/' + alert_id + '/action', json={'action': 'test-post-action'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)

        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('test-post-action', data['alert']['tags'])

        # check status=closed
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertIn('true', data['alert']['tags'])
        self.assertEqual(data['alert']['history'][3]['text'],
                         'ticket resolved by bob (ticket #12345)-plugin1-plugin3', data['alert']['history'])

        del plugins.plugins['old1']
        del plugins.plugins['test1']
        del plugins.plugins['test2']
        del plugins.plugins['test3']

    def test_invalid_action(self) -> None:
        plugins.plugins['action3'] = CustActionPlugin3()

        # create alert
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id = data['id']

        # create ticket for alert
        payload = {
            'action': 'invalid',
            'text': ''
        }
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 409)

        del plugins.plugins['action3']

    def test_im_a_teapot(self) -> None:
        plugins.plugins['teapot'] = Teapot()

        # send coffee alert
        response = self.client.post('/alert', json=self.coffee_alert, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], [
            'server refuses to brew coffee because it is, permanently, a teapot',
            'See https://tools.ietf.org/html/rfc2324'
        ])

        # send non-coffee alert
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id = data['id']

        # set status to coffee
        payload = {
            'status': 'coffee',
            'text': ''
        }
        response = self.client.put('/alert/' + alert_id + '/status', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], [
            'server refuses to brew coffee because it is, permanently, a teapot',
            'See https://tools.ietf.org/html/rfc2324'
        ])

        # coffee action
        payload = {
            'action': 'coffee',
            'text': ''
        }
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], [
            'server refuses to brew coffee because it is, permanently, a teapot',
            'See https://tools.ietf.org/html/rfc2324'
        ])

        # coffee note
        payload = {
            'text': 'coffee'
        }
        response = self.client.put('/alert/' + alert_id + '/note', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], [
            'server refuses to brew coffee because it is, permanently, a teapot',
            'See https://tools.ietf.org/html/rfc2324'
        ])

        # delete non-coffee alert
        response = self.client.delete('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], [
            'server refuses to brew coffee because it is, permanently, a teapot',
            'See https://tools.ietf.org/html/rfc2324'
        ])

        del plugins.plugins['teapot']

    def test_take_note(self) -> None:
        plugins.plugins['old1'] = OldPlugin1()
        plugins.plugins['test1'] = CustPlugin1()
        plugins.plugins['test2'] = CustPlugin2()
        plugins.plugins['test3'] = CustPlugin3()

        plugins.plugins['note1'] = CustNotePlugin1()

        # create alert
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id = data['id']

        # check plugin is triggered on create
        payload = {
            'text': 'caused by: power outage'
        }
        response = self.client.put('/alert/' + alert_id + '/note', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['note']['text'], 'caused by: power outage (ticket #12345)')
        note_id = data['note']['id']

        # check if attribute got properly edited by plugin
        response = self.client.get('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['cause'], 'power outage')

        # check plugin is triggered on update
        payload = {
            'text': 'caused by: zombie invasion'
        }
        response = self.client.put('/alert/' + alert_id + '/note/' + note_id, json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['note']['text'], 'caused by: zombie invasion (ticket #23456)')

        # check if attribute got properly edited by plugin
        response = self.client.get('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['cause'], 'zombie invasion')

        del plugins.plugins['old1']
        del plugins.plugins['test1']
        del plugins.plugins['test2']
        del plugins.plugins['test3']
        del plugins.plugins['note1']

    def test_delete(self) -> None:
        plugins.plugins['old1'] = OldPlugin1()
        plugins.plugins['test1'] = CustPlugin1()
        plugins.plugins['test2'] = CustPlugin2()
        plugins.plugins['test3'] = CustPlugin3()

        plugins.plugins['delete1'] = CustDeletePlugin1()
        plugins.plugins['delete2'] = CustDeletePlugin2()

        # create alert
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id = data['id']

        # delete alert
        response = self.client.delete('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))

        # check deleted
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 404)

        del plugins.plugins['old1']
        del plugins.plugins['test1']
        del plugins.plugins['test2']
        del plugins.plugins['test3']

        del plugins.plugins['delete1']
        del plugins.plugins['delete2']

    def test_add_and_remove_tags(self) -> None:
        plugins.plugins['old1'] = OldPlugin1()
        plugins.plugins['test1'] = CustPlugin1()
        plugins.plugins['test2'] = CustPlugin2()
        plugins.plugins['test3'] = CustPlugin3()

        plugins.plugins['action1'] = CustActionPlugin1()
        plugins.plugins['action2'] = CustActionPlugin2()

        # create alert
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))

        alert_id = data['id']

        payload = {
            'action': 'ack',
            'text': 'this is a test'
        }
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)

        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('aSingleTag', data['alert']['tags'])
        self.assertIn('a:Triple:Tag', data['alert']['tags'])
        self.assertNotIn('aDouble:Tag', data['alert']['tags'])

        del plugins.plugins['old1']
        del plugins.plugins['test1']
        del plugins.plugins['test2']
        del plugins.plugins['test3']

        del plugins.plugins['action1']
        del plugins.plugins