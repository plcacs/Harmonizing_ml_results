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
        test_config: Dict[str, Any] = {'TESTING': True, 'AUTH_REQUIRED': False, 'PLUGINS': ['remote_ip']}
        os.environ['ALLOWED_ENVIRONMENTS'] = 'Production,Staging,Development'
        self.app = create_app(test_config)
        self.client = self.app.test_client()
        self.resource: str = str(uuid4()).upper()[:8]
        self.reject_alert: Dict[str, Any] = {'event': 'node_marginal', 'resource': self.resource, 'environment': 'Staging', 'service': [], 'severity': 'warning', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'tags': ['one', 'two']}
        self.accept_alert: Dict[str, Any] = {'event': 'node_marginal', 'resource': self.resource, 'environment': 'Staging', 'service': ['Network'], 'severity': 'warning', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'tags': ['three', 'four']}
        self.critical_alert: Dict[str, Any] = {'event': 'node_down', 'resource': self.resource, 'environment': 'Staging', 'service': ['Network'], 'severity': 'critical', 'correlate': ['node_down', 'node_marginal', 'node_up'], 'tags': []}
        self.coffee_alert: Dict[str, Any] = {'event': 'coffee_pot', 'resource': self.resource, 'environment': 'Staging', 'service': ['Network'], 'severity': 'critical', 'text': 'coffee alert', 'tags': []}
        self.headers: Dict[str, str] = {'Content-type': 'application/json'}

    def tearDown(self) -> None:
        plugins.plugins.clear()
        db.destroy()

    def test_status_update(self) -> None:
        plugins.plugins['old1'] = OldPlugin1()
        plugins.plugins['test1'] = CustPlugin1()
        plugins.plugins['test2'] = CustPlugin2()
        plugins.plugins['test3'] = CustPlugin3()
        response = self.client.post('/alert', data=json.dumps(self.accept_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'ok')
        self.assertRegex(data['id'], '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
        self.assertEqual(data['alert']['attributes']['old'], 'post1')
        self.assertEqual(data['alert']['attributes']['aaa'], 'post1')
        alert_id = data['id']
        response = self.client.put('/alert/' + alert_id + '/status', data=json.dumps({'status': 'ack', 'text': 'input'}), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['old'], 'post1')
        self.assertEqual(data['alert']['attributes']['aaa'], 'post1')
        self.assertEqual(data['alert']['status'], 'assign')
        self.assertEqual(sorted(data['alert']['tags']), sorted(['Development', 'Production', 'Staging', 'more', 'other', 'that', 'the', 'this']))
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
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['tags'], ['Production', 'Staging', 'Development'])
        self.assertEqual(data['alert']['attributes'], {'aaa': 'post1', 'ip': '127.0.0.1', 'old': 'post1'})
        alert_id = data['id']
        payload: Dict[str, str] = {'action': 'createTicket', 'text': 'ticket created by bob'}
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'assign')
        self.assertEqual(sorted(data['alert']['tags']), sorted(['Development', 'Production', 'Staging', 'more', 'other', 'that', 'the', 'this', 'aSingleTag', 'aDouble:Tag', 'a:Triple:Tag']))
        self.assertEqual(data['alert']['history'][1]['text'], 'ticket created by bob (ticket #12345)-plugin1-plugin3', data['alert']['history'])
        payload = {'action': 'updateTicket', 'text': 'ticket updated by bob'}
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'assign')
        self.assertEqual(data['alert']['attributes']['up'], 'down')
        self.assertEqual(data['alert']['history'][2]['text'], 'ticket updated by bob (ticket #12345)-plugin1-plugin3', data['alert']['history'])
        payload = {'action': 'resolveTicket', 'text': 'ticket resolved by bob'}
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.put('/alert/' + alert_id + '/action', json={'action': 'test-post-action'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('test-post-action', data['alert']['tags'])
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['status'], 'closed')
        self.assertIn('true', data['alert']['tags'])
        self.assertEqual(data['alert']['history'][3]['text'], 'ticket resolved by bob (ticket #12345)-plugin1-plugin3', data['alert']['history'])
        del plugins.plugins['old1']
        del plugins.plugins['test1']
        del plugins.plugins['test2']
        del plugins.plugins['test3']

    def test_invalid_action(self) -> None:
        plugins.plugins['action3'] = CustActionPlugin3()
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        alert_id = data['id']
        payload: Dict[str, str] = {'action': 'invalid', 'text': ''}
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 409)
        del plugins.plugins['action3']

    def test_im_a_teapot(self) -> None:
        plugins.plugins['teapot'] = Teapot()
        response = self.client.post('/alert', json=self.coffee_alert, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], ['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        alert_id = data['id']
        payload: Dict[str, str] = {'status': 'coffee', 'text': ''}
        response = self.client.put('/alert/' + alert_id + '/status', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], ['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
        payload = {'action': 'coffee', 'text': ''}
        response = self.client.put('/alert/' + alert_id + '/action', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], ['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
        payload = {'text': 'coffee'}
        response = self.client.put('/alert/' + alert_id + '/note', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], ['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
        response = self.client.delete('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 418)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['message'], "I'm a teapot")
        self.assertCountEqual(data['errors'], ['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
        del plugins.plugins['teapot']

    def test_take_note(self) -> None:
        plugins.plugins['old1'] = OldPlugin1()
        plugins.plugins['test1'] = CustPlugin1()
        plugins.plugins['test2'] = CustPlugin2()
        plugins.plugins['test3'] = CustPlugin3()
        plugins.plugins['note1'] = CustNotePlugin1()
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        alert_id = data['id']
        payload: Dict[str, str] = {'text': 'caused by: power outage'}
        response = self.client.put('/alert/' + alert_id + '/note', json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['note']['text'], 'caused by: power outage (ticket #12345)')
        note_id = data['note']['id']
        response = self.client.get('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['cause'], 'power outage')
        payload = {'text': 'caused by: zombie invasion'}
        response = self.client.put('/alert/' + alert_id + '/note/' + note_id, json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['note']['text'], 'caused by: zombie invasion (ticket #23456)')
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
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        alert_id = data['id']
        response = self.client.delete('/alert/' + alert_id, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
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
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        alert_id = data['id']
        payload: Dict[str, str] = {'action': 'ack', 'text': 'this is a test'}
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
        del plugins.plugins['action2']

    def test_custom_ack(self) -> None:
        plugins.plugins['ack1'] = CustAckPlugin1()
        response = self.client.post('/alert', json=self.critical_alert, headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        alert_id = data['id']
        response = self.client.put('/alert/' + alert_id + '/action', json={'action': 'shelve'}, headers=self.headers)
        response = self.client.put('/alert/' + alert_id + '/action', json={'action': 'unshelve'}, headers=self.headers)
        response = self.client.put('/alert/' + alert_id + '/action', json={'action': 'ack1'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'ack')
        response = self.client.put('/alert/' + alert_id + '/action', json={'action': 'unack'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/alert/' + alert_id)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['severity'], 'critical')
        self.assertEqual(data['alert']['status'], 'open')
        del plugins.plugins['ack1']

class OldPlugin1(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        ALLOWED_ENVIRONMENTS: List[str] = self.get_config('ALLOWED_ENVIRONMENTS', default=[], type=list, **kwargs)
        alert.attributes['old'] = 'pre1'
        alert.tags = ALLOWED_ENVIRONMENTS
        return alert

    def post_receive(self, alert: Alert) -> Alert:
        alert.attributes['old'] = 'post1'
        return alert

    def status_change(self, alert: Alert, status: str, text: str) -> Tuple[Alert, str, str]:
        return (alert, status, text)

class CustPlugin1(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        alert.attributes['aaa'] = 'pre1'
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        alert.attributes['aaa'] = 'post1'
        return alert

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        alert.tags.extend(['this', 'that', 'the', 'other'])
        alert.attributes['foo'] = 'bar'
        alert.attributes['abc'] = 123
        alert.attributes['xyz'] = 'up'
        text = text + '-plugin1'
        return (alert, status, text)

class CustPlugin2(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> None:
        return

class CustPlugin3(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        alert.tags.extend(['this', 'that', 'more'])
        alert.attributes['baz'] = 'quux'
        if alert.attributes['abc'] == 123:
            alert.attributes['abc'] = None
        alert.attributes['xyz'] = 'down'
        if status == 'ack':
            status = 'assign'
        text = text + '-plugin3'
        return (alert, status, text)

class CustActionPlugin1(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        if action == 'createTicket':
            alert.status = 'ack'
            text = text + ' (ticket #12345)'
        if action == 'updateTicket':
            alert.attributes['up'] = 'down'
            text = text + ' (ticket #12345)'
        if action == 'resolveTicket':
            alert.status = 'closed'
            alert.tags.append('true')
            text = text + ' (ticket #12345)'
        alert.tags.append('aSingleTag')
        alert.tags.extend(['aDouble:Tag', 'a:Triple:Tag'])
        return (alert, action, text)

    def post_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Optional[Alert]:
        if action == 'test-post-action':
            alert.tags.append('test-post-action')
            return alert
        return None

class CustActionPlugin2(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        alert.tags.remove('aDouble:Tag')
        if action == 'invalid':
            raise InvalidAction(f'{action} is not a valid action for this status')
        return (alert, action, text)

    def post_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> None:
        raise NotImplementedError

class CustActionPlugin3(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        if action == 'invalid':
            raise InvalidAction(f'{action} is not a valid action for this status')
        return (alert, action, text)

    def post_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> None:
        pass

class CustNotePlugin1(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_note(self, alert: Alert, text: str, **kwargs: Any) -> Tuple[Alert, str]:
        if text == 'caused by: power outage':
            alert.attributes['cause'] = 'power outage'
            text = text + ' (ticket #12345)'
        if text == 'caused by: zombie invasion':
            alert.attributes['cause'] = 'zombie invasion'
            text = text + ' (ticket #23456)'
        return (alert, text)

class CustDeletePlugin1(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, action, text)

    def delete(self, alert: Alert, **kwargs: Any) -> bool:
        return True

class CustDeletePlugin2(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, action, text)

    def delete(self, alert: Alert, **kwargs: Any) -> bool:
        return True

class CustAckPlugin1(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Optional[Alert]:
        if not action == 'ack1':
            return None
        alert.status = Status.Ack
        return alert

    def delete(self, alert: Alert, **kwargs: Any) -> bool:
        return True

class Teapot(PluginBase):

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        if 'coffee' in alert.text:
            raise AlertaException("I'm a teapot", code=418, errors=['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        if 'coffee' in alert.text:
            raise AlertaException("I'm a teapot", code=418, errors=['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> None:
        if status != 'tea':
            raise AlertaException("I'm a teapot", code=418, errors=['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        if action != 'tea':
            raise AlertaException("I'm a teapot", code=418, errors=['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
        return (alert, action, text)

    def take_note(self, alert: Alert, text: str, **kwargs: Any) -> None:
        if 'coffee' in text:
            raise AlertaException("I'm a teapot", code=418, errors=['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])

    def delete(self, alert: Alert, **kwargs: Any) -> None:
        raise AlertaException("I'm a teapot", code=418, errors=['server refuses to brew coffee because it is, permanently, a teapot', 'See https://tools.ietf.org/html/rfc2324'])
