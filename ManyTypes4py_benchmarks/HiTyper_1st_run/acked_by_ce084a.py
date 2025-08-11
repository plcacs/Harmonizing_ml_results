import logging
from flask import g
from alerta.plugins import PluginBase
LOG = logging.getLogger('alerta.plugins')

class AckedBy(PluginBase):
    """
    Add "acked-by" attribute to alerts with login id of the operator when
    an alert is acked and automatically watch the alert. Unset the attribute
    when alert is un-acked. Un-watching requires manual intervention.

    To display the "acked-by" attribute in the alert summary add "acked-by"
    to the list of alert attributes in the COLUMNS server setting.
    """

    def pre_receive(self, alert: list[tuple[str]], **kwargs) -> list[tuple[str]]:
        return alert

    def post_receive(self, alert: list[tuple[str]], **kwargs) -> None:
        return

    def status_change(self, alert: Union[str, list[str]], status: Union[str, set[str], typing.Sequence[str]], text: Union[str, set[str], typing.Sequence[str]], **kwargs) -> Union[str, list[str]]:
        if status == 'open':
            alert.attributes['acked-by'] = None
        return alert

    def take_action(self, alert: Union[str, dict[str, str], list, None], action: str, text: Union[str, list[dict]], **kwargs) -> Union[str, dict[str, str], list, None]:
        if action == 'ack' and g.login:
            watch = 'watch:' + g.login
            alert.tags.append(watch)
            alert.attributes['acked-by'] = g.login
        if action == 'unack':
            alert.attributes['acked-by'] = None
        return alert

    def delete(self, alert: Union[list[tuple[str]], list[dict[str, typing.Any]], str], **kwargs) -> None:
        raise NotImplementedError