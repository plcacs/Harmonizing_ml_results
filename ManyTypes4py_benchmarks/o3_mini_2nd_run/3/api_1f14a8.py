import logging
from typing import Any, Dict, List, Optional, Tuple
from flask import current_app, g
from alerta.app import plugins
from alerta.exceptions import (AlertaException, ApiError, BlackoutPeriod, ForwardingLoop, 
                               HeartbeatReceived, InvalidAction, RateLimit, RejectException)
from alerta.models.alert import Alert
from alerta.models.enums import Scope


def assign_customer(wanted: Optional[str] = None, permission: Scope = Scope.admin_alerts) -> Optional[str]:
    customers: List[str] = g.get('customers', [])
    if wanted:
        if Scope.admin in g.scopes or permission in g.scopes:
            return wanted
        if wanted not in customers:
            raise ApiError(f"not allowed to set customer to '{wanted}'", 400)
        else:
            return wanted
    if customers:
        if len(customers) > 1:
            raise ApiError('must define customer as more than one possibility', 400)
        else:
            return customers[0]
    return None


def process_alert(alert: Alert) -> Alert:
    wanted_plugins, wanted_config = plugins.routing(alert)  # type: Tuple[List[Any], Dict[str, Any]]
    skip_plugins: bool = False
    for plugin in wanted_plugins:
        if alert.is_suppressed:
            skip_plugins = True
            break
        try:
            alert = plugin.pre_receive(alert, config=wanted_config)
        except TypeError:
            alert = plugin.pre_receive(alert)
        except (RejectException, HeartbeatReceived, BlackoutPeriod, RateLimit, ForwardingLoop, AlertaException):
            raise
        except Exception as e:
            if current_app.config['PLUGINS_RAISE_ON_ERROR']:
                raise RuntimeError(f"Error while running pre-receive plugin '{plugin.name}': {str(e)}")
            else:
                logging.error(f"Error while running pre-receive plugin '{plugin.name}': {str(e)}")
        if not alert:
            raise SyntaxError(f"Plugin '{plugin.name}' pre-receive hook did not return modified alert")
    try:
        is_duplicate = alert.is_duplicate()
        if is_duplicate:
            alert = alert.deduplicate(is_duplicate)
        else:
            is_correlated = alert.is_correlated()
            if is_correlated:
                alert = alert.update(is_correlated)
            else:
                alert = alert.create()
    except Exception as e:
        raise ApiError(str(e))
    wanted_plugins, wanted_config = plugins.routing(alert)  # type: Tuple[List[Any], Dict[str, Any]]
    alert_was_updated: bool = False
    for plugin in wanted_plugins:
        if skip_plugins:
            break
        try:
            updated = plugin.post_receive(alert, config=wanted_config)
        except TypeError:
            updated = plugin.post_receive(alert)
        except AlertaException:
            raise
        except Exception as e:
            if current_app.config['PLUGINS_RAISE_ON_ERROR']:
                raise ApiError(f"Error while running post-receive plugin '{plugin.name}': {str(e)}")
            else:
                logging.error(f"Error while running post-receive plugin '{plugin.name}': {str(e)}")
        if updated:
            alert = updated
            alert_was_updated = True
    if alert_was_updated:
        alert.update_tags(alert.tags)
        alert.attributes = alert.update_attributes(alert.attributes)
    return alert


def process_action(alert: Alert, action: str, text: str, timeout: Optional[int] = None, 
                   post_action: bool = False) -> Tuple[Alert, str, str, Optional[int]]:
    wanted_plugins, wanted_config = plugins.routing(alert)  # type: Tuple[List[Any], Dict[str, Any]]
    updated: Any = None
    alert_was_updated: bool = False
    for plugin in wanted_plugins:
        if alert.is_suppressed:
            break
        try:
            if post_action:
                updated = plugin.post_action(alert, action, text, timeout=timeout, config=wanted_config)
            else:
                updated = plugin.take_action(alert, action, text, timeout=timeout, config=wanted_config)
        except NotImplementedError:
            pass
        except (RejectException, ForwardingLoop, InvalidAction, AlertaException):
            raise
        except Exception as e:
            if current_app.config['PLUGINS_RAISE_ON_ERROR']:
                raise ApiError(f"Error while running action plugin '{plugin.name}': {str(e)}")
            else:
                logging.error(f"Error while running action plugin '{plugin.name}': {str(e)}")
        if isinstance(updated, Alert):
            updated = (updated, action, text, timeout)
        if isinstance(updated, tuple):
            if len(updated) == 4:
                alert, action, text, timeout = updated
            elif len(updated) == 3:
                alert, action, text = updated
        if updated:
            alert_was_updated = True
    if alert_was_updated:
        alert.update_tags(alert.tags)
        alert.attributes = alert.update_attributes(alert.attributes)
    return (alert, action, text, timeout)


def process_note(alert: Alert, text: str) -> Tuple[Alert, str]:
    wanted_plugins, wanted_config = plugins.routing(alert)  # type: Tuple[List[Any], Dict[str, Any]]
    updated: Any = None
    alert_was_updated: bool = False
    for plugin in wanted_plugins:
        try:
            updated = plugin.take_note(alert, text, config=wanted_config)
        except NotImplementedError:
            pass
        except (RejectException, ForwardingLoop, AlertaException):
            raise
        except Exception as e:
            if current_app.config['PLUGINS_RAISE_ON_ERROR']:
                raise ApiError(f"Error while running note plugin '{plugin.name}': {str(e)}")
            else:
                logging.error(f"Error while running note plugin '{plugin.name}': {str(e)}")
        if isinstance(updated, Alert):
            updated = (updated, text)
        if isinstance(updated, tuple) and len(updated) == 2:
            alert, text = updated
        if updated:
            alert_was_updated = True
    if alert_was_updated:
        alert.update_tags(alert.tags)
        alert.update_attributes(alert.attributes)
    return (alert, text)


def process_status(alert: Alert, status: str, text: str) -> Tuple[Alert, str, str]:
    wanted_plugins, wanted_config = plugins.routing(alert)  # type: Tuple[List[Any], Dict[str, Any]]
    updated: Any = None
    alert_was_updated: bool = False
    for plugin in wanted_plugins:
        if alert.is_suppressed:
            break
        try:
            updated = plugin.status_change(alert, status, text, config=wanted_config)
        except TypeError:
            updated = plugin.status_change(alert, status, text)
        except (RejectException, AlertaException):
            raise
        except Exception as e:
            if current_app.config['PLUGINS_RAISE_ON_ERROR']:
                raise ApiError(f"Error while running status plugin '{plugin.name}': {str(e)}")
            else:
                logging.error(f"Error while running status plugin '{plugin.name}': {str(e)}")
        if updated:
            alert_was_updated = True
            try:
                alert, status, text = updated
            except Exception:
                alert = updated
    if alert_was_updated:
        alert.update_tags(alert.tags)
        alert.attributes = alert.update_attributes(alert.attributes)
    return (alert, status, text)


def process_delete(alert: Alert) -> bool:
    wanted_plugins, wanted_config = plugins.routing(alert)  # type: Tuple[List[Any], Dict[str, Any]]
    delete: bool = True
    for plugin in wanted_plugins:
        try:
            delete = delete and plugin.delete(alert, config=wanted_config)
        except NotImplementedError:
            pass
        except (RejectException, AlertaException):
            raise
        except Exception as e:
            if current_app.config['PLUGINS_RAISE_ON_ERROR']:
                raise ApiError(f"Error while running delete plugin '{plugin.name}': {str(e)}")
            else:
                logging.error(f"Error while running delete plugin '{plugin.name}': {str(e)}")
    return delete and alert.delete()