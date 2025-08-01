import abc
import logging
import os
from typing import Any, Optional, Type, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from alerta.models.alert import Alert

LOG = logging.getLogger('alerta.plugins')

class PluginBase(metaclass=abc.ABCMeta):

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name or self.__module__
        if self.__doc__:
            LOG.info(f'\n{self.__doc__}\n')

    @abc.abstractmethod
    def pre_receive(self, alert: "Alert", **kwargs: Any) -> Any:
        """
        Pre-process an alert based on alert properties or reject it
        by raising RejectException or BlackoutPeriod.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def post_receive(self, alert: "Alert", **kwargs: Any) -> Any:
        """Send an alert to another service or notify users."""
        raise NotImplementedError

    @abc.abstractmethod
    def status_change(self, alert: "Alert", status: str, text: str, **kwargs: Any) -> Any:
        """Trigger integrations based on status changes."""
        raise NotImplementedError

    def take_action(self, alert: "Alert", action: str, text: str, **kwargs: Any) -> Any:
        """
        Trigger integrations based on external actions. (optional)
        Pre-trigger, eg. this triggers before the status are updated.
        """
        raise NotImplementedError

    def post_action(self, alert: "Alert", action: str, text: str, **kwargs: Any) -> Any:
        """
        Trigger integrations based on external actions. (optional)
        Post-trigger, eg. after the status is updated
        """
        raise NotImplementedError

    def take_note(self, alert: "Alert", text: str, **kwargs: Any) -> Any:
        """Trigger integrations based on notes. (optional)"""
        raise NotImplementedError

    def delete(self, alert: "Alert", **kwargs: Any) -> Any:
        """Trigger integrations when an alert is deleted. (optional)"""
        raise NotImplementedError

    @staticmethod
    def get_config(key: str, default: Optional[Any] = None, type: Optional[Type] = None, **kwargs: Any) -> Any:
        if key in os.environ:
            rv: Any = os.environ[key]
            if type == bool:
                return rv.lower() in ['yes', 'on', 'true', 't', '1']
            elif type == list:
                return rv.split(',')
            elif type is not None:
                try:
                    rv = type(rv)
                except ValueError:
                    rv = default
            return rv
        try:
            rv = kwargs['config'].get(key, default)
        except KeyError:
            rv = default
        return rv

class FakeApp:

    def init_app(self) -> None:
        from alerta.app import config
        self.config: Dict[str, Any] = config.get_user_config()

app: FakeApp = FakeApp()