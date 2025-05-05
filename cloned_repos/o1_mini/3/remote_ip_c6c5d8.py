import logging
from typing import Any, Dict

from flask import request
from alerta.models.alert import Alert
from alerta.plugins import PluginBase

LOG = logging.getLogger('alerta.plugins')


class RemoteIpAddr(PluginBase):
    """
    Add originating IP address of HTTP client as an alert attribute. This information
    can be used for debugging, access control, or generating geolocation data.
    """

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        remote_addr = next(iter(request.access_route), request.remote_addr)
        alert.attributes.update(ip=remote_addr)
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(
        self, alert: Alert, status: str, text: str, **kwargs: Any
    ) -> None:
        return

    def take_action(
        self, alert: Alert, action: str, text: str, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    def delete(self, alert: Alert, **kwargs: Any) -> Any:
        raise NotImplementedError
