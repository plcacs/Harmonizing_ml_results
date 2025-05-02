import logging
from typing import Any, Dict, Optional
from flask import g
from alerta.plugins import PluginBase
from alerta.models.alert import Alert

LOG = logging.getLogger('alerta.plugins')

class AckedBy(PluginBase):
    """
    Add "acked-by" attribute to alerts with login id of the operator when
    an alert is acked and automatically watch the alert. Unset the attribute
    when alert is un-acked. Un-watching requires manual intervention.

    To display the "acked-by" attribute in the alert summary add "acked-by"
    to the list of alert attributes in the COLUMNS server setting.
    """

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Alert:
        if status == 'open':
            alert.attributes['acked-by'] = None
        return alert

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Alert:
        if action == 'ack' and g.login:
            watch = 'watch:' + g.login
            alert.tags.append(watch)
            alert.attributes['acked-by'] = g.login
        if action == 'unack':
            alert.attributes['acked-by'] = None
        return alert

    def delete(self, alert: Alert, **kwargs: Any) -> None:
        raise NotImplementedError
