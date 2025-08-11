import logging
from flask import request
from alerta.plugins import PluginBase
LOG = logging.getLogger('alerta.plugins')

class RemoteIpAddr(PluginBase):
    """
    Add originating IP address of HTTP client as an alert attribute. This information
    can be used for debugging, access control, or generating geolocation data.
    """

    def pre_receive(self, alert: Union[dict[str, typing.Any], dict, alerta.models.alerAlert], **kwargs) -> Union[dict[str, typing.Any], dict, alerta.models.alerAlert]:
        remote_addr = next(iter(request.access_route), request.remote_addr)
        alert.attributes.update(ip=remote_addr)
        return alert

    def post_receive(self, alert: list[tuple[str]], **kwargs) -> None:
        return

    def status_change(self, alert: Union[str, set[str], typing.Sequence[str]], status: Union[str, set[str], typing.Sequence[str]], text: Union[str, set[str], typing.Sequence[str]], **kwargs) -> None:
        return

    def take_action(self, alert: Union[str, list[dict]], action: Union[str, list[dict]], text: Union[str, list[dict]], **kwargs) -> None:
        raise NotImplementedError

    def delete(self, alert: Union[list[tuple[str]], list[dict[str, typing.Any]], str], **kwargs) -> None:
        raise NotImplementedError