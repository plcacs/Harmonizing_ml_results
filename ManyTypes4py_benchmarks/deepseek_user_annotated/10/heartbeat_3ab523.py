import logging
from typing import Any, List, Optional

from alerta.exceptions import HeartbeatReceived
from alerta.models.heartbeat import Heartbeat
from alerta.plugins import PluginBase

LOG = logging.getLogger('alerta.plugins')


class HeartbeatReceiver(PluginBase):
    """
    Default heartbeat receiver intercepts alerts with event='Heartbeat', converts
    them into heartbeats and will return a 202 Accept HTTP status code.
    """

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        HEARTBEAT_EVENTS: List[str] = self.get_config('HEARTBEAT_EVENTS', default=['Heartbeat'], type=list, **kwargs)

        if alert.event in HEARTBEAT_EVENTS:
            hb = Heartbeat(
                origin=alert.origin,
                tags=alert.tags,
                attributes={
                    'environment': alert.environment,
                    'severity': alert.severity,
                    'service': alert.service,
                    'group': alert.group
                },
                timeout=alert.timeout,
                customer=alert.customer
            )
            r = hb.create()
            raise HeartbeatReceived(r.id)

        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> None:
        return None

    def status_change(self, alert: Any, status: str, text: str, **kwargs: Any) -> None:
        return None

    def take_action(self, alert: Any, action: str, text: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def delete(self, alert: Any, **kwargs: Any) -> bool:
        raise NotImplementedError
