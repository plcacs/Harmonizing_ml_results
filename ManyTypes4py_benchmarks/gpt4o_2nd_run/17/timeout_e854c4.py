import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple
from alerta.models.enums import ChangeType
from alerta.plugins import PluginBase

if TYPE_CHECKING:
    from alerta.models.alert import Alert

LOG = logging.getLogger('alerta.plugins')

class TimeoutPolicy(PluginBase):
    """
    Override user-defined ack and shelve timeout values with server defaults.
    """

    def pre_receive(self, alert: 'Alert', **kwargs: Any) -> 'Alert':
        return alert

    def post_receive(self, alert: 'Alert', **kwargs: Any) -> Optional[None]:
        return None

    def status_change(self, alert: 'Alert', status: str, text: str, **kwargs: Any) -> Optional[None]:
        return

    def take_action(self, alert: 'Alert', action: str, text: str, **kwargs: Any) -> Tuple['Alert', str, str, int]:
        timeout: int = kwargs['timeout']
        if action == ChangeType.ack:
            ack_timeout: int = self.get_config('ACK_TIMEOUT')
            if timeout != ack_timeout:
                LOG.warning('Override user-defined ack timeout of {} seconds to {} seconds.'.format(timeout, ack_timeout))
                timeout = ack_timeout
                text += ' (using server timeout value)'
        if action == ChangeType.shelve:
            shelve_timeout: int = self.get_config('SHELVE_TIMEOUT')
            if timeout != shelve_timeout:
                LOG.warning('Override user-defined shelve timeout of {} seconds to {} seconds.'.format(timeout, shelve_timeout))
                timeout = shelve_timeout
                text += ' (using server timeout value)'
        return (alert, action, text, timeout)

    def take_note(self, alert: 'Alert', text: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def delete(self, alert: 'Alert', **kwargs: Any) -> None:
        raise NotImplementedError
