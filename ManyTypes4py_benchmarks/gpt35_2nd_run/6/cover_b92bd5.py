from typing import Any

class MBCover(MicroBeesEntity, CoverEntity):
    def __init__(self, coordinator: MicroBeesUpdateCoordinator, bee_id: int, actuator_up_id: int, actuator_down_id: int) -> None:
    def name(self) -> str:
    @property
    def actuator_up(self) -> Actuator:
    @property
    def actuator_down(self) -> Actuator:
    def _reset_open_close(self, *args: Any) -> None:
    async def async_open_cover(self, **kwargs: Any) -> None:
    async def async_close_cover(self, **kwargs: Any) -> None:
    async def async_stop_cover(self, **kwargs: Any) -> None:
