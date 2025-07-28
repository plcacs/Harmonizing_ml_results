import json
import os
import unittest
import pkg_resources
from typing import Any, Dict, List, Tuple
from alerta.app import create_app, plugins
from alerta.models.enums import Scope
from alerta.models.key import ApiKey
from alerta.plugins import PluginBase

class RoutingTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.dist: pkg_resources.Distribution = pkg_resources.Distribution(__file__, project_name='alerta-routing', version='0.1')
        s: str = 'rules = tests.test_zrouting:rules'
        self.entry_point: pkg_resources.EntryPoint = pkg_resources.EntryPoint.parse(s, dist=self.dist)
        self.dist._ep_map = {'alerta.routing': {'rules': self.entry_point}}
        pkg_resources.working_set.add(self.dist)
        test_config: Dict[str, Any] = {
            'TESTING': True,
            'AUTH_REQUIRED': True,
            'CUSTOMER_VIEWS': True,
            'ADMIN_USERS': ['admin@alerta.io'],
            'ALLOWED_EMAIL_DOMAINS': ['alerta.io', 'foo.com', 'bar.com'],
            'PD_API_KEYS': {'Tyrell Corporation': 'tc-key', 'Cyberdyne Systems': 'cs-key', 'Weyland-Yutani': 'wy-key', 'Zorin Enterprises': 'ze-key'},
            'SLACK_API_KEYS': {'Soylent Corporation': 'sc-key', 'Omni Consumer Products': 'ocp-key', 'Dolmansaxlil Shoe Corporation': 'dsc-key'},
            'API_KEY': 'default-key',
            'HOOK': 'not-set'
        }
        self.app: Any = create_app(test_config)
        self.client: Any = self.app.test_client()
        self.tier1_tc_alert: Dict[str, Any] = {'event': 'foo1', 'resource': 'foo1', 'environment': 'Production', 'service': ['Web'], 'customer': 'Tyrell Corporation'}
        self.tier1_wy_alert: Dict[str, Any] = {'event': 'foo1', 'resource': 'foo1', 'environment': 'Production', 'service': ['Web'], 'customer': 'Weyland-Yutani'}
        self.tier2_sc_alert: Dict[str, Any] = {'event': 'bar1', 'resource': 'bar1', 'environment': 'Production', 'service': ['Web'], 'customer': 'Soylent Corporation'}
        self.tier2_ocp_alert: Dict[str, Any] = {'event': 'bar1', 'resource': 'bar1', 'environment': 'Production', 'service': ['Web'], 'customer': 'Omni Consumer Products'}
        self.tier2_dsc_alert: Dict[str, Any] = {'event': 'bar1', 'resource': 'bar1', 'environment': 'Production', 'service': ['Web'], 'customer': 'Dolmansaxlil Shoe Corporation'}
        self.tier3_it_alert: Dict[str, Any] = {'event': 'bar1', 'resource': 'bar1', 'environment': 'Production', 'service': ['Web'], 'customer': 'Initech'}
        with self.app.test_request_context('/'):
            self.app.preprocess_request()
            self.api_key: ApiKey = ApiKey(user='admin@alerta.io', scopes=[Scope.admin, Scope.read, Scope.write], text='admin-key')
            self.api_key.create()
        self.headers: Dict[str, str] = {'Authorization': f'Key {self.api_key.key}', 'Content-type': 'application/json'}
        plugins.plugins['pagerduty'] = DummyPagerDutyPlugin()
        plugins.plugins['slack'] = DummySlackPlugin()
        plugins.plugins['config'] = DummyConfigPlugin()

    def tearDown(self) -> None:
        plugins.plugins.clear()
        self.dist._ep_map.clear()

    def test_config(self) -> None:
        os.environ['BOOL_ENVVAR'] = 'yes'
        os.environ['INT_ENVVAR'] = '2020'
        os.environ['FLOAT_ENVVAR'] = '0.99'
        os.environ['LIST_ENVVAR'] = 'up,down,left,right'
        os.environ['STR_ENVVAR'] = 'a string with spaces'
        os.environ['DICT_ENVVAR'] = '{"key":"value", "key2": "value2" }'
        self.app.config.update({
            'BOOL_SETTING': True,
            'INT_SETTING': 1001,
            'FLOAT_SETTING': 7.07,
            'LIST_SETTING': ['a', 2, 'z'],
            'STR_SETTING': ' long string with spaces',
            'DICT_SETTING': {'a': 'dict', 'with': 'three', 'keys': 3}
        })
        response: Any = self.client.post('/alert', data=json.dumps(self.tier1_tc_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['env']['bool'], True)
        self.assertEqual(data['alert']['attributes']['env']['int'], 2020)
        self.assertEqual(data['alert']['attributes']['env']['float'], 0.99)
        self.assertEqual(data['alert']['attributes']['env']['list'], ['up', 'down', 'left', 'right'])
        self.assertEqual(data['alert']['attributes']['env']['str'], 'a string with spaces')
        self.assertEqual(data['alert']['attributes']['env']['dict'], dict(key='value', key2='value2'))
        self.assertEqual(data['alert']['attributes']['setting']['bool'], True)
        self.assertEqual(data['alert']['attributes']['setting']['int'], 1001)
        self.assertEqual(data['alert']['attributes']['setting']['float'], 7.07)
        self.assertEqual(data['alert']['attributes']['setting']['list'], ['a', 2, 'z'])
        self.assertEqual(data['alert']['attributes']['setting']['str'], ' long string with spaces')
        self.assertEqual(data['alert']['attributes']['setting']['dict'], {'a': 'dict', 'with': 'three', 'keys': 3})
        self.assertEqual(data['alert']['attributes']['default']['bool'], False)
        self.assertEqual(data['alert']['attributes']['default']['int'], 999)
        self.assertEqual(data['alert']['attributes']['default']['float'], 5.55)
        self.assertEqual(data['alert']['attributes']['default']['list'], ['j', 'k'])
        self.assertEqual(data['alert']['attributes']['default']['str'], 'setting')
        self.assertEqual(data['alert']['attributes']['default']['dict'], dict(baz='quux'))

    def test_config_precedence(self) -> None:
        os.environ['var1'] = 'env1'
        self.app.config.update({'var1': 'setting1', 'var2': 'setting2'})
        response: Any = self.client.post('/alert', data=json.dumps(self.tier1_tc_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['precedence']['var1'], 'env1')
        self.assertEqual(data['alert']['attributes']['precedence']['var2'], 'setting2')
        self.assertEqual(data['alert']['attributes']['precedence']['var3'], 'default3')

    def test_routing(self) -> None:
        response: Any = self.client.post('/alert', data=json.dumps(self.tier1_tc_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data: Dict[str, Any] = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['NOTIFY'], 'pagerduty')
        self.assertEqual(data['alert']['attributes']['API_KEY'], 'tc-key')
        response = self.client.post('/alert', data=json.dumps(self.tier1_wy_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['NOTIFY'], 'pagerduty')
        self.assertEqual(data['alert']['attributes']['API_KEY'], 'wy-key')
        response = self.client.post('/alert', data=json.dumps(self.tier2_sc_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['NOTIFY'], 'slack')
        self.assertEqual(data['alert']['attributes']['API_KEY'], 'sc-key')
        response = self.client.post('/alert', data=json.dumps(self.tier2_ocp_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['NOTIFY'], 'slack')
        self.assertEqual(data['alert']['attributes']['API_KEY'], 'ocp-key')
        response = self.client.post('/alert', data=json.dumps(self.tier2_dsc_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['alert']['attributes']['NOTIFY'], 'slack')
        self.assertEqual(data['alert']['attributes']['API_KEY'], 'dsc-key')
        response = self.client.post('/alert', data=json.dumps(self.tier3_it_alert), headers=self.headers)
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertNotIn('NOTIFY', data['alert']['attributes'])
        self.assertNotIn('API_KEY', data['alert']['attributes'])


class DummyConfigPlugin(unittest.TestCase, PluginBase):

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        alert.attributes['env'] = {
            'bool': self.get_config('BOOL_ENVVAR', default=False, type=bool, **kwargs),
            'int': self.get_config('INT_ENVVAR', default=100, type=int, **kwargs),
            'float': self.get_config('FLOAT_ENVVAR', default=1.001, type=float, **kwargs),
            'list': self.get_config('LIST_ENVVAR', default=['a', 'b', 'c'], type=list, **kwargs),
            'str': self.get_config('STR_ENVVAR', default='environment', type=str, **kwargs),
            'dict': self.get_config('DICT_ENVVAR', default={'foo': 'bar'}, type=json.loads, **kwargs)
        }
        alert.attributes['setting'] = {
            'bool': self.get_config('BOOL_SETTING', default=False, type=bool, **kwargs),
            'int': self.get_config('INT_SETTING', default=999, type=int, **kwargs),
            'float': self.get_config('FLOAT_SETTING', default=5.55, type=float, **kwargs),
            'list': self.get_config('LIST_SETTING', default=['j', 'k'], type=list, **kwargs),
            'str': self.get_config('STR_SETTING', default='setting', type=str, **kwargs),
            'dict': self.get_config('DICT_SETTING', default={'baz': 'quux'}, type=json.loads, **kwargs)
        }
        alert.attributes['default'] = {
            'bool': self.get_config('BOOL_DEFAULT', default=False, type=bool, **kwargs),
            'int': self.get_config('INT_DEFAULT', default=999, type=int, **kwargs),
            'float': self.get_config('FLOAT_DEFAULT', default=5.55, type=float, **kwargs),
            'list': self.get_config('LIST_DEFAULT', default=['j', 'k'], type=list, **kwargs),
            'str': self.get_config('STR_DEFAULT', default='setting', type=str, **kwargs),
            'dict': self.get_config('DICT_DEFAULT', default={'baz': 'quux'}, type=json.loads, **kwargs)
        }
        alert.attributes['precedence'] = {
            'var1': self.get_config('var1', default='default1', **kwargs),
            'var2': self.get_config('var2', default='default2', **kwargs),
            'var3': self.get_config('var3', default='default3', **kwargs)
        }
        self.assertEqual(self.get_config('HOOK', default='', type=str, **kwargs), 'pre-receive')
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> Any:
        self.assertEqual(self.get_config('HOOK', default='', type=str, **kwargs), 'post-receive')
        return alert

    def status_change(self, alert: Any, status: Any, text: Any, **kwargs: Any) -> Tuple[Any, Any, Any]:
        return (alert, status, text)

class DummyPagerDutyPlugin(PluginBase):

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> Any:
        alert.attributes['NOTIFY'] = 'pagerduty'
        alert.attributes['API_KEY'] = self.get_config('API_KEY', **kwargs)
        return alert

    def status_change(self, alert: Any, status: Any, text: Any, **kwargs: Any) -> Tuple[Any, Any, Any]:
        return (alert, status, text)

class DummySlackPlugin(PluginBase):

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> Any:
        alert.attributes['NOTIFY'] = 'slack'
        alert.attributes['API_KEY'] = self.get_config('API_KEY', **kwargs)
        return alert

    def status_change(self, alert: Any, status: Any, text: Any, **kwargs: Any) -> Tuple[Any, Any, Any]:
        return (alert, status, text)

def rules(alert: Any, plugins: Dict[str, PluginBase], **kwargs: Any) -> Tuple[List[PluginBase], Dict[str, Any]]:
    if alert.repeat is None:
        hook: str = 'pre-receive'
    else:
        hook = 'post-receive'
    TIER_ONE_CUSTOMERS: List[str] = ['Tyrell Corporation', 'Cyberdyne Systems', 'Weyland-Yutani', 'Zorin Enterprises']
    TIER_TWO_CUSTOMERS: List[str] = ['Soylent Corporation', 'Omni Consumer Products', 'Dolmansaxlil Shoe Corporation']
    config: Dict[str, Any] = kwargs['config']
    if alert.customer in TIER_ONE_CUSTOMERS:
        return (
            [plugins['remote_ip'], plugins['reject'], plugins['blackout'], plugins['pagerduty'], plugins['config']],
            dict(HOOK=hook, API_KEY=config['PD_API_KEYS'][alert.customer])
        )
    elif alert.customer in TIER_TWO_CUSTOMERS:
        return (
            [plugins['remote_ip'], plugins['reject'], plugins['blackout'], plugins['slack'], plugins['config']],
            dict(HOOK=hook, API_KEY=config['SLACK_API_KEYS'][alert.customer])
        )
    else:
        return (
            [plugins['remote_ip'], plugins['reject'], plugins['blackout'], plugins['config']],
            dict(HOOK=hook)
        )