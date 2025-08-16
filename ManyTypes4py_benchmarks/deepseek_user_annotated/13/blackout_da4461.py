import logging
from typing import Any, Optional

from alerta.exceptions import BlackoutPeriod
from alerta.plugins import PluginBase
from alerta.models.alert import Alert

LOG = logging.getLogger('alerta.plugins')


class BlackoutHandler(PluginBase):
    """
    Default suppression blackout handler will drop alerts that match a blackout
    period and will return a 202 Accept HTTP status code.

    If "NOTIFICATION_BLACKOUT" is set to ``True`` then the alert is processed
    but alert status is set to "blackout" and the alert will not be passed to
    any plugins for further notification.
    """

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        NOTIFICATION_BLACKOUT: bool = self.get_config('NOTIFICATION_BLACKOUT', default=True, type=bool, **kwargs)

        status: str = 'blackout' if self.get_config('ALARM_MODEL', **kwargs) == 'ALERTA' else 'OOSRV'  # ISA_18_2

        if alert.is_blackout():
            if NOTIFICATION_BLACKOUT:
                LOG.debug(f'Set status to "{status}" during blackout period (id={alert.id})')
                alert.status = status
            else:
                LOG.debug(f'Suppressed alert during blackout period (id={alert.id})')
                raise BlackoutPeriod('Suppressed alert during blackout period')
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return None

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> None:
        return None

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def delete(self, alert: Alert, **kwargs: Any) -> bool:
        raise NotImplementedError
