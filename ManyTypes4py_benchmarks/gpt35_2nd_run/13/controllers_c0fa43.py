    def get_app(id_or_name: str, cast_type: Optional[str] = None, strict: bool = False, show_warning: bool = False) -> App:

    def get_controller(cast: Any, app: App, action: Optional[str] = None, prep: Optional[str] = None) -> Any:

    def setup_cast(device_desc: Any, video_url: Optional[str] = None, controller: Optional[str] = None, ytdl_options: Any = None, action: Optional[str] = None, prep: Optional[str] = None) -> Any:

    class CattStore:

    class CastState(CattStore):

    class CastStatusListener:

    class MediaStatusListener:

    class SimpleListener:

    class CastController:

    class MediaControllerMixin:

    class PlaybackBaseMixin:

    class DefaultCastController(CastController, MediaControllerMixin, PlaybackBaseMixin):

    class DashCastController(CastController):

    class YoutubeCastController(CastController, MediaControllerMixin, PlaybackBaseMixin):
