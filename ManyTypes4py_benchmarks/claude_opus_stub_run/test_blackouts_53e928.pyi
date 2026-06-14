import json
import os
import time
import unittest
from datetime import datetime, timedelta
from typing import Any, Optional

from flask import Flask
from flask.testing import FlaskClient

from alerta.app import create_app, db, plugins
from alerta.exceptions import BlackoutPeriod
from alerta.models.alert import Alert
from alerta.models.key import ApiKey
from alerta.plugins import PluginBase
from alerta.utils.format import DateTime


class BlackoutsTestCase(unittest.TestCase):
    app: Flask
    client: FlaskClient
    prod_alert: dict[str, Any]
    dev_alert: dict[str, Any]
    fatal_alert: dict[str, Any]
    critical_alert: dict[str, Any]
    major_alert: dict[str, Any]
    normal_alert: dict[str, Any]
    minor_alert: dict[str, Any]
    ok_alert: dict[str, Any]
    warn_alert: dict[str, Any]
    admin_api_key: ApiKey
    customer_api_key: ApiKey
    headers: dict[str, str]

    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def test_suppress_blackout(self) -> None: ...
    def test_notification_blackout(self) -> None: ...
    def test_previous_status(self) -> None: ...
    def test_whole_environment_blackout(self) -> None: ...
    def test_combination_blackout(self) -> None: ...
    def test_origin_blackout(self) -> None: ...
    def test_custom_notify(self) -> None: ...
    def test_edit_blackout(self) -> None: ...
    def test_user_info(self) -> None: ...


class Blackout(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert: ...
    def post_receive(self, alert: Alert, **kwargs: Any) -> Alert: ...
    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> None: ...


class CustomNotify(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert: ...
    def post_receive(self, alert: Alert, **kwargs: Any) -> Alert: ...
    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> None: ...