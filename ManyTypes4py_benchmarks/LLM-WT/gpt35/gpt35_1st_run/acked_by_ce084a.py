import logging
from flask import g
from alerta.plugins import PluginBase
from alerta.models.alert import Alert
from typing import Optional

LOG: logging.Logger = logging.getLogger('alerta.plugins')

class AckedBy(PluginBase):
    def pre_receive(self, alert: Alert, **kwargs) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs) -> Alert:
        if status == 'open':
            alert.attributes['acked-by'] = None
        return alert

    def take_action(self, alert: Alert, action: str, text: str, **kwargs) -> Alert:
        if action == 'ack' and g.login:
            watch: str = 'watch:' + g.login
            alert.tags.append(watch)
            alert.attributes['acked-by'] = g.login
        if action == 'unack':
            alert.attributes['acked-by'] = None
        return alert

    def delete(self, alert: Alert, **kwargs) -> None:
        raise NotImplementedError
