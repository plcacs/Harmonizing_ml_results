import logging
from flask import request
from alerta.plugins import PluginBase

LOG = logging.getLogger('alerta.plugins')

class RemoteIpAddr(PluginBase):
    """
    Add originating IP address of HTTP client as an alert attribute. This information
    can be used for debugging, access control, or generating geolocation data.
    """

    def pre_receive(self, alert: object, **kwargs: dict) -> object:
        remote_addr = next(iter(request.access_route), request.remote_addr)
        alert.attributes.update(ip=remote_addr)
        return alert

    def post_receive(self, alert: object, **kwargs: dict) -> None:
        return

    def status_change(self, alert: object, status: str, text: str, **kwargs: dict) -> None:
        return

    def take_action(self, alert: object, action: str, text: str, **kwargs: dict) -> None:
        raise NotImplementedError

    def delete(self, alert: object, **kwargs: dict) -> None:
        raise NotImplementedError
