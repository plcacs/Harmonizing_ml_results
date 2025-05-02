import logging
from flask import g
from alerta.plugins import PluginBase
from typing import Any, Dict, Optional

LOG = logging.getLogger('alerta.plugins')

class AckedBy(PluginBase):
    """
    Add "acked-by" attribute to alerts with login id of the operator when
    an alert is acked and automatically watch the alert. Unset the attribute
    when alert is un-acked. Un-watching requires manual intervention.

    To display the "acked-by" attribute in the alert summary add "acked-by"
    to the list of alert attributes in the COLUMNS server setting.
    """

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Any, status: str, text: str, **kwargs: Any) -> Any:
        if status == 'open':
            alert.attributes['acked-by'] = None
        return alert

    def take_action(self, alert: Any, action: str, text: str, **kwargs: Any) -> Any:
        if action == 'ack' and g.login:
            watch = 'watch:' + g.login
            alert.tags.append(watch)
            alert.attributes['acked-by'] = g.login
        if action == 'unack':
            alert.attributes['acked-by'] = None
        return alert

    def delete(self, alert: Any, **kwargs: Any) -> None:
        raise NotImplementedError
