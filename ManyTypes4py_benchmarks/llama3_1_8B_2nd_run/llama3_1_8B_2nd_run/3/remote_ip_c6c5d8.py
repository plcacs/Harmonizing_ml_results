import logging
from flask import Request
from alerta.plugins import PluginBase

LOG = logging.getLogger('alerta.plugins')

class RemoteIpAddr(PluginBase):
    """
    Add originating IP address of HTTP client as an alert attribute. This information
    can be used for debugging, access control, or generating geolocation data.
    """

    def pre_receive(self, alert: dict, **kwargs) -> dict:
        remote_addr = next(iter(request.access_route), request.remote_addr)
        alert['attributes']['ip'] = remote_addr
        return alert

    def post_receive(self, alert: dict, **kwargs) -> None:
        return

    def status_change(self, alert: dict, status: str, text: str, **kwargs) -> None:
        return

    def take_action(self, alert: dict, action: str, text: str, **kwargs) -> None:
        raise NotImplementedError

    def delete(self, alert: dict, **kwargs) -> None:
        raise NotImplementedError
