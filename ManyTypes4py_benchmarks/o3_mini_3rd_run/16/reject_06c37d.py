import logging
import re
from typing import Any, List, Pattern
from alerta.exceptions import RejectException
from alerta.plugins import PluginBase

LOG = logging.getLogger('alerta.plugins')


class RejectPolicy(PluginBase):
    """
    Default reject policy will block alerts that do not have the following
    required attributes:
    1) environment - must match an allowed environment. By default it should
       be either "Production" or "Development". Config setting is `ALLOWED_ENVIRONMENTS`.
    2) service - must supply a value for service. Any value is acceptable.
    """

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        ORIGIN_BLACKLIST: List[str] = self.get_config('ORIGIN_BLACKLIST', default=[], type=list, **kwargs)
        ALLOWED_ENVIRONMENTS: List[str] = self.get_config('ALLOWED_ENVIRONMENTS', default=[], type=list, **kwargs)
        ORIGIN_BLACKLIST_REGEX: List[Pattern] = [re.compile(x) for x in ORIGIN_BLACKLIST]
        ALLOWED_ENVIRONMENT_REGEX: List[Pattern] = [re.compile(x) for x in ALLOWED_ENVIRONMENTS]

        if any((regex.match(alert.origin) for regex in ORIGIN_BLACKLIST_REGEX)):
            LOG.warning("[POLICY] Alert origin '%s' has been blacklisted", alert.origin)
            raise RejectException(f"[POLICY] Alert origin '{alert.origin}' has been blacklisted")

        if not any((regex.fullmatch(alert.environment) for regex in ALLOWED_ENVIRONMENT_REGEX)):
            LOG.warning('[POLICY] Alert environment does not match one of %s', ', '.join(ALLOWED_ENVIRONMENTS))
            raise RejectException('[POLICY] Alert environment does not match one of %s' % ', '.join(ALLOWED_ENVIRONMENTS))

        if not alert.service:
            LOG.warning('[POLICY] Alert must define a service')
            raise RejectException('[POLICY] Alert must define a service')

        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Any, status: str, text: str, **kwargs: Any) -> None:
        return

    def take_action(self, alert: Any, action: str, text: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def delete(self, alert: Any, **kwargs: Any) -> None:
        raise NotImplementedError