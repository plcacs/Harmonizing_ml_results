import logging
from ihcsdk.ihccontroller import IHCController
from homeassistant.helpers.entity import Entity
from .const import CONF_INFO, DOMAIN
_LOGGER = logging.getLogger(__name__)

class IHCEntity(Entity):
    _attr_should_poll: bool = False

    def __init__(self, ihc_controller: IHCController, controller_id: str, name: str, ihc_id: str, product: dict = None) -> None:
        self.ihc_controller: IHCController = ihc_controller
        self._name: str = name
        self.ihc_id: str = ihc_id
        self.controller_id: str = controller_id
        self.device_id: str = None
        self.suggested_area: str = None
        if product:
            self.ihc_name: str = product['name']
            self.ihc_note: str = product['note']
            self.ihc_position: str = product['position']
            self.suggested_area: str = product.get('group')
            if 'id' in product:
                product_id: str = product['id']
                self.device_id: str = f'{controller_id}_{product_id}'
                self.device_name: str = product['name']
                if self.ihc_position:
                    self.device_name += f' ({self.ihc_position})'
                self.device_model: str = product['model']
        else:
            self.ihc_name: str = ''
            self.ihc_note: str = ''
            self.ihc_position: str = ''

    async def async_added_to_hass(self) -> None:
        _LOGGER.debug('Adding IHC entity notify event: %s', self.ihc_id)
        self.ihc_controller.add_notify_event(self.ihc_id, self.on_ihc_change, True)

    @property
    def name(self) -> str:
        return self._name

    @property
    def unique_id(self) -> str:
        return f'{self.controller_id}-{self.ihc_id}'

    @property
    def extra_state_attributes(self) -> dict:
        if not self.hass.data[DOMAIN][self.controller_id][CONF_INFO]:
            return {}
        attributes: dict = {'ihc_id': self.ihc_id, 'ihc_name': self.ihc_name, 'ihc_note': self.ihc_note, 'ihc_position': self.ihc_position}
        if len(self.hass.data[DOMAIN]) > 1:
            attributes['ihc_controller'] = self.controller_id
        return attributes

    def on_ihc_change(self, ihc_id: str, value: str) -> None:
        raise NotImplementedError
