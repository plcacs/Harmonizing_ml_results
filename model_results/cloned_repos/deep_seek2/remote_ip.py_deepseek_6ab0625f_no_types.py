import logging
from typing import Any, Dict, Optional
from flask import request
from alerta.models.alert import Alert
from alerta.plugins import PluginBase
LOG = logging.getLogger('alerta.plugins')

class RemoteIpAddr(PluginBase):
    """
    Add originating IP address of HTTP client as an alert attribute. This information
    can be used for debugging, access control, or generating geolocation data.
    """

    def pre_receive(self, alert, **kwargs: Any):
        remote_addr = next(iter(request.access_route), request.remote_addr)
        alert.attributes.update(ip=remote_addr)
        return alert

    def post_receive(self, alert, **kwargs: Any):
        return None

    def status_change(self, alert, status, text, **kwargs: Any):
        return None

    def take_action(self, alert, action, text, **kwargs: Any):
        raise NotImplementedError

    def delete(self, alert, **kwargs: Any):
        raise NotImplementedError