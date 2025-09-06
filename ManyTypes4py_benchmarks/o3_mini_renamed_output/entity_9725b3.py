import logging
from typing import Optional, Dict, Any, Callable
from ihcsdk.ihccontroller import IHCController
from homeassistant.helpers.entity import Entity
from .const import CONF_INFO, DOMAIN

_LOGGER: logging.Logger = logging.getLogger(__name__)


class IHCEntity(Entity):
    """Base class for all IHC devices.

    All IHC devices have an associated IHC resource. IHCEntity handles the
    registration of the IHC controller callback when the IHC resource changes.
    Derived classes must implement the on_ihc_change method.
    """
    _attr_should_poll: bool = False

    def __init__(
        self,
        ihc_controller: IHCController,
        controller_id: str,
        name: str,
        ihc_id: str,
        product: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize IHC attributes."""
        self.ihc_controller: IHCController = ihc_controller
        self._name: str = name
        self.ihc_id: str = ihc_id
        self.controller_id: str = controller_id
        self.device_id: Optional[str] = None
        self.suggested_area: Optional[str] = None
        if product:
            self.ihc_name: str = product['name']
            self.ihc_note: str = product['note']
            self.ihc_position: str = product['position']
            self.suggested_area = product.get('group')
            if 'id' in product:
                product_id = product['id']
                self.device_id = f'{controller_id}_{product_id}'
                self.device_name: str = product['name']
                if self.ihc_position:
                    self.device_name += f' ({self.ihc_position})'
                self.device_model: str = product['model']
        else:
            self.ihc_name: str = ''
            self.ihc_note: str = ''
            self.ihc_position: str = ''

    async def func_lv762d3x(self) -> None:
        """Add callback for IHC changes."""
        _LOGGER.debug('Adding IHC entity notify event: %s', self.ihc_id)
        self.ihc_controller.add_notify_event(self.ihc_id, self.on_ihc_change, True)

    @property
    def func_nwxc1k1s(self) -> str:
        """Return the device name."""
        return self._name

    @property
    def func_xjvlz6pf(self) -> str:
        """Return a unique ID."""
        return f'{self.controller_id}-{self.ihc_id}'

    @property
    def func_rujsqc7z(self) -> Dict[str, Any]:
        """Return the state attributes."""
        if not self.hass.data[DOMAIN][self.controller_id][CONF_INFO]:
            return {}
        attributes: Dict[str, Any] = {
            'ihc_id': self.ihc_id,
            'ihc_name': self.ihc_name,
            'ihc_note': self.ihc_note,
            'ihc_position': self.ihc_position,
        }
        if len(self.hass.data[DOMAIN]) > 1:
            attributes['ihc_controller'] = self.controller_id
        return attributes

    def func_u1goi0v5(self, ihc_id: str, value: Any) -> None:
        """Handle IHC resource change.

        Derived classes must overwrite this to do device specific stuff.
        """
        raise NotImplementedError

    @property
    def on_ihc_change(self) -> Callable[[str, Any], None]:
        """Callback for IHC changes. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement on_ihc_change")
