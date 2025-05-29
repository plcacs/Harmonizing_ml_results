import logging
from typing import TYPE_CHECKING, Any, Optional, List, Tuple, Dict
from flask import request
from alerta.exceptions import ForwardingLoop
from alerta.plugins import PluginBase
from alerta.utils.client import Client
from alerta.utils.response import base_url
if TYPE_CHECKING:
    from alerta.models.alert import Alert
LOG: logging.Logger = logging.getLogger('alerta.plugins.forwarder')
X_LOOP_HEADER: str = 'X-Alerta-Loop'

def append_to_header(origin):
    x_loop: Optional[str] = request.headers.get(X_LOOP_HEADER)
    return origin if not x_loop else f'{x_loop},{origin}'

def is_in_xloop(server):
    x_loop: Optional[str] = request.headers.get(X_LOOP_HEADER)
    return server in x_loop if server and x_loop else False

class Forwarder(PluginBase):
    """
    Alert and action forwarder for federated Alerta deployments
    See https://docs.alerta.io/en/latest/federated.html
    """

    def pre_receive(self, alert, **kwargs: Any):
        if is_in_xloop(base_url()):
            http_origin: str = request.origin or '(unknown)'
            raise ForwardingLoop(f'Alert forwarded by {http_origin} already processed by {base_url()}')
        return alert

    def post_receive(self, alert, **kwargs: Any):
        destinations: List[Tuple[str, Dict[str, Any], List[str]]] = self.get_config('FWD_DESTINATIONS', default=[], type=list, **kwargs)
        for remote, auth, actions in destinations:
            if is_in_xloop(remote):
                LOG.debug(f'Forward [action=alerts]: {alert.id} ; Remote {remote} already processed alert. Skip.')
                continue
            if not ('*' in actions or 'alerts' in actions):
                LOG.debug(f'Forward [action=alerts]: {alert.id} ; Remote {remote} not configured for alerts. Skip.')
                continue
            headers: Dict[str, str] = {X_LOOP_HEADER: append_to_header(base_url())}
            client: Client = Client(endpoint=remote, headers=headers, **auth)
            LOG.info(f'Forward [action=alerts]: {alert.id} ; {base_url()} -> {remote}')
            try:
                body: Dict[str, Any] = alert.get_body(history=False)
                body['id'] = alert.last_receive_id
                r = client.send_alert(**body)
            except Exception as e:
                LOG.warning(f'Forward [action=alerts]: {alert.id} ; Failed to forward alert to {remote} - {str(e)}')
                continue
            LOG.debug(f'Forward [action=alerts]: {alert.id} ; [{r.status_code}] {r.text}')
        return alert

    def status_change(self, alert, status, text, **kwargs: Any):
        return

    def take_action(self, alert, action, text, **kwargs: Any):
        if is_in_xloop(base_url()):
            http_origin: str = request.origin or '(unknown)'
            raise ForwardingLoop('Action {} forwarded by {} already processed by {}'.format(action, http_origin, base_url()))
        destinations: List[Tuple[str, Dict[str, Any], List[str]]] = self.get_config('FWD_DESTINATIONS', default=[], type=list, **kwargs)
        for remote, auth, actions in destinations:
            if is_in_xloop(remote):
                LOG.debug(f'Forward [action={action}]: {alert.id} ; Remote {remote} already processed action. Skip.')
                continue
            if not ('*' in actions or 'actions' in actions or action in actions):
                LOG.debug(f'Forward [action={action}]: {alert.id} ; Remote {remote} not configured for action. Skip.')
                continue
            headers: Dict[str, str] = {X_LOOP_HEADER: append_to_header(base_url())}
            client: Client = Client(endpoint=remote, headers=headers, **auth)
            LOG.info(f'Forward [action={action}]: {alert.id} ; {base_url()} -> {remote}')
            try:
                r = client.action(alert.id, action, text)
            except Exception as e:
                LOG.warning(f'Forward [action={action}]: {alert.id} ; Failed to action alert on {remote} - {str(e)}')
                continue
            LOG.debug(f'Forward [action={action}]: {alert.id} ; [{r.status_code}] {r.text}')
        return alert

    def delete(self, alert, **kwargs: Any):
        if is_in_xloop(base_url()):
            http_origin: str = request.origin or '(unknown)'
            raise ForwardingLoop(f'Delete forwarded by {http_origin} already processed by {base_url()}')
        destinations: List[Tuple[str, Dict[str, Any], List[str]]] = self.get_config('FWD_DESTINATIONS', default=[], type=list, **kwargs)
        for remote, auth, actions in destinations:
            if is_in_xloop(remote):
                LOG.debug(f'Forward [action=delete]: {alert.id} ; Remote {remote} already processed delete. Skip.')
                continue
            if not ('*' in actions or 'delete' in actions):
                LOG.debug(f'Forward [action=delete]: {alert.id} ; Remote {remote} not configured for deletes. Skip.')
                continue
            headers: Dict[str, str] = {X_LOOP_HEADER: append_to_header(base_url())}
            client: Client = Client(endpoint=remote, headers=headers, **auth)
            LOG.info(f'Forward [action=delete]: {alert.id} ; {base_url()} -> {remote}')
            try:
                r = client.delete_alert(alert.id)
            except Exception as e:
                LOG.warning(f'Forward [action=delete]: {alert.id} ; Failed to delete alert on {remote} - {str(e)}')
                continue
            LOG.debug(f'Forward [action=delete]: {alert.id} ; [{r.status_code}] {r.text}')
        return True