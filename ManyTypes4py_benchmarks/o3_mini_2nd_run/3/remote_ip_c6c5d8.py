import logging
from flask import request
from alerta.plugins import PluginBase
from typing import Any

LOG: logging.Logger = logging.getLogger('alerta.plugins')

class RemoteIpAddr(PluginBase):
    """
    Add originating IP address of HTTP client as an alert attribute. This information
    can be used for debugging, access control, or generating geolocation data.
    """

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        remote_addr: str = next(iter(request.access_route), request.remote_addr)
        alert.attributes.update(ip=remote_addr)
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Any, status: Any, text: Any, **kwargs: Any) -> None:
        return

    def take_action(self, alert: Any, action: Any, text: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def delete(self, alert: Any, **kwargs: Any) -> None:
        raise NotImplementedError