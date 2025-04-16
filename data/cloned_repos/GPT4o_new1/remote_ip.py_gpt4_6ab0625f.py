import logging
from flask import request
from alerta.plugins import PluginBase
from typing import Any, Dict

LOG = logging.getLogger('alerta.plugins')

class RemoteIpAddr(PluginBase):
    """
    Add originating IP address of HTTP client as an alert attribute. This information
    can be used for debugging, access control, or generating geolocation data.
    """

    def pre_receive(self, alert: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        remote_addr: str = next(iter(request.access_route), request.remote_addr)
        alert['attributes'].update(ip=remote_addr)
        return alert

    def post_receive(self, alert: Dict[str, Any], **kwargs: Any) -> None:
        return

    def status_change(self, alert: Dict[str, Any], status: str, text: str, **kwargs: Any) -> None:
        return

    def take_action(self, alert: Dict[str, Any], action: str, text: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def delete(self, alert: Dict[str, Any], **kwargs: Any) -> bool:
        raise NotImplementedError
