import json
import os
import unittest
import pkg_resources
from alerta.app import create_app, plugins
from alerta.models.enums import Scope
from alerta.models.key import ApiKey
from alerta.plugins import PluginBase

class RoutingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # ...

    def tearDown(self) -> None:
        # ...

    def test_config(self) -> None:
        # ...

    def test_config_precedence(self) -> None:
        # ...

    def test_routing(self) -> None:
        # ...

class DummyConfigPlugin(unittest.TestCase, PluginBase):
    def pre_receive(self, alert: dict, **kwargs: dict) -> dict:
        # ...

    def post_receive(self, alert: dict, **kwargs: dict) -> dict:
        # ...

    def status_change(self, alert: dict, status: str, text: str, **kwargs: dict) -> tuple:
        # ...

class DummyPagerDutyPlugin(PluginBase):
    def pre_receive(self, alert: dict, **kwargs: dict) -> dict:
        # ...

    def post_receive(self, alert: dict, **kwargs: dict) -> dict:
        # ...

    def status_change(self, alert: dict, status: str, text: str, **kwargs: dict) -> tuple:
        # ...

class DummySlackPlugin(PluginBase):
    def pre_receive(self, alert: dict, **kwargs: dict) -> dict:
        # ...

    def post_receive(self, alert: dict, **kwargs: dict) -> dict:
        # ...

    def status_change(self, alert: dict, status: str, text: str, **kwargs: dict) -> tuple:
        # ...

def rules(alert: dict, plugins: dict, **kwargs: dict) -> tuple:
    # ...
