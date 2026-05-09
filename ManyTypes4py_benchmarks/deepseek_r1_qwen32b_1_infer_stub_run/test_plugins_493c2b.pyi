import json
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
from flask.typing import Response
from alerta.app import App
from alerta.models.alert import Alert
from alerta.models.enums import Status
from alerta.plugins import PluginBase

class PluginsTestCase(unittest.TestCase):
    app: App
    client: Any
    resource: str
    reject_alert: Dict[str, Any]
    accept_alert: Dict[str, Any]
    critical_alert: Dict[str, Any]
    coffee_alert: Dict[str, Any]
    headers: Dict[str, str]
    
    def setUp(self) -> None:
        ...
    
    def tearDown(self) -> None:
        ...
    
    def test_status_update(self) -> None:
        ...
    
    def test_take_action(self) -> None:
        ...
    
    def test_invalid_action(self) -> None:
        ...
    
    def test_im_a_teapot(self) -> None:
        ...
    
    def test_take_note(self) -> None:
        ...
    
    def test_delete(self) -> None:
        ...
    
    def test_add_and_remove_tags(self) -> None:
        ...
    
    def test_custom_ack(self) -> None:
        ...

class OldPlugin1(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert) -> Alert:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str]) -> Tuple[Alert, Status, Optional[str]]:
        ...

class CustPlugin1(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...

class CustPlugin2(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> None:
        ...

class CustPlugin3(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...

class CustActionPlugin1(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...
    
    def take_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, str, Optional[str]]:
        ...
    
    def post_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Alert:
        ...

class CustActionPlugin2(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...
    
    def take_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, str, Optional[str]]:
        ...
    
    def post_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> None:
        ...

class CustActionPlugin3(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...
    
    def take_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, str, Optional[str]]:
        ...
    
    def post_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> None:
        ...

class CustNotePlugin1(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...
    
    def take_note(self, alert: Alert, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Optional[str]]:
        ...

class CustDeletePlugin1(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...
    
    def take_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, str, Optional[str]]:
        ...
    
    def delete(self, alert: Alert, **kwargs: Dict[str, Any]) -> bool:
        ...

class CustDeletePlugin2(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...
    
    def take_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, str, Optional[str]]:
        ...
    
    def delete(self, alert: Alert, **kwargs: Dict[str, Any]) -> bool:
        ...

class CustAckPlugin1(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, Status, Optional[str]]:
        ...
    
    def take_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Optional[Alert]:
        ...
    
    def delete(self, alert: Alert, **kwargs: Dict[str, Any]) -> bool:
        ...

class Teapot(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> Alert:
        ...
    
    def post_receive(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...
    
    def status_change(self, alert: Alert, status: Status, text: Optional[str], **kwargs: Dict[str, Any]) -> None:
        ...
    
    def take_action(self, alert: Alert, action: str, text: Optional[str], **kwargs: Dict[str, Any]) -> Tuple[Alert, str, Optional[str]]:
        ...
    
    def take_note(self, alert: Alert, text: Optional[str], **kwargs: Dict[str, Any]) -> None:
        ...
    
    def delete(self, alert: Alert, **kwargs: Dict[str, Any]) -> None:
        ...