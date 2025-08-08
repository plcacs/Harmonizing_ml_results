def async_response(func: Callable) -> Callable:
def require_admin(func: Callable) -> Callable:
def ws_require_user(only_owner: bool = False, only_system_user: bool = False, allow_system_user: bool = True, only_active_user: bool = True, only_inactive_user: bool = False, only_supervisor: bool = False) -> Callable:
def websocket_command(schema: VolDictType) -> Callable:
