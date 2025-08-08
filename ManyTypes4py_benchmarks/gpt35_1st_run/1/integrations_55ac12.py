from typing import Any, Callable, Sequence

OptionValidator = Callable[[str, str], Any]
META_CATEGORY: dict[str, str] = {'meta-integration': gettext_lazy('Integration frameworks'), 'bots': gettext_lazy('Interactive bots')}
CATEGORIES: dict[str, str] = {**META_CATEGORY, 'continuous-integration': gettext_lazy('Continuous integration'), 'customer-support': gettext_lazy('Customer support'), 'deployment': gettext_lazy('Deployment'), 'entertainment': gettext_lazy('Entertainment'), 'communication': gettext_lazy('Communication'), 'financial': gettext_lazy('Financial'), 'hr': gettext_lazy('Human resources'), 'marketing': gettext_lazy('Marketing'), 'misc': gettext_lazy('Miscellaneous'), 'monitoring': gettext_lazy('Monitoring'), 'project-management': gettext_lazy('Project management'), 'productivity': gettext_lazy('Productivity'), 'version-control': gettext_lazy('Version control')}

class Integration:
    DEFAULT_LOGO_STATIC_PATH_PNG: str = 'images/integrations/logos/{name}.png'
    DEFAULT_LOGO_STATIC_PATH_SVG: str = 'images/integrations/logos/{name}.svg'
    DEFAULT_BOT_AVATAR_PATH: str = 'images/integrations/bot_avatars/{name}.png'

    def __init__(self, name: str, categories: Sequence[str], client_name: str = None, logo: str = None, secondary_line_text: str = None, display_name: str = None, doc: str = None, stream_name: str = None, legacy: bool = False, config_options: Sequence[Any] = []):
        self.name: str = name
        self.client_name: str = client_name if client_name is not None else name
        self.secondary_line_text: str = secondary_line_text
        self.legacy: bool = legacy
        self.doc: str = doc
        self.config_options: Sequence[Any] = config_options
        self.categories: Sequence[str] = [CATEGORIES[c] for c in categories]
        self.logo_path: str = logo if logo is not None else self.get_logo_path()
        self.logo_url: str = self.get_logo_url()
        self.display_name: str = display_name if display_name is not None else name.title()
        self.stream_name: str = stream_name if stream_name is not None else self.name

    def is_enabled(self) -> bool:
        return True

    def get_logo_path(self) -> str:
        ...

    def get_bot_avatar_path(self) -> str:
        ...

    def get_logo_url(self) -> str:
        ...

    def get_translated_categories(self) -> Sequence[str]:
        ...

class BotIntegration(Integration):
    ...

class WebhookIntegration(Integration):
    ...

def split_fixture_path(path: str) -> tuple[str, str]:
    ...

@dataclass
class BaseScreenshotConfig:
    image_name: str = '001.png'
    image_dir: str = None
    bot_name: str = None

@dataclass
class ScreenshotConfig(BaseScreenshotConfig):
    payload_as_query_param: bool = False
    payload_param_name: str = 'payload'
    extra_params: dict = field(default_factory=dict)
    use_basic_auth: bool = False
    custom_headers: dict = field(default_factory=dict)

def get_fixture_and_image_paths(integration: Integration, screenshot_config: ScreenshotConfig) -> tuple[str, str]:
    ...

class HubotIntegration(Integration):
    ...

class EmbeddedBotIntegration(Integration):
    ...

EMBEDDED_BOTS: list[EmbeddedBotIntegration] = [EmbeddedBotIntegration('converter', []), EmbeddedBotIntegration('encrypt', []), EmbeddedBotIntegration('helloworld', []), EmbeddedBotIntegration('virtual_fs', []), EmbeddedBotIntegration('giphy', []), EmbeddedBotIntegration('followup', [])]
WEBHOOK_INTEGRATIONS: list[WebhookIntegration] = [...]
INTEGRATIONS: dict[str, Integration] = {...}
BOT_INTEGRATIONS: list[BotIntegration] = [...]
HUBOT_INTEGRATIONS: list[HubotIntegration] = [...]
NO_SCREENSHOT_WEBHOOKS: set[str] = {'beeminder', 'ifttt', 'slack_incoming', 'zapier'}
DOC_SCREENSHOT_CONFIG: dict[str, list[ScreenshotConfig]] = {...}

def get_all_event_types_for_integration(integration: Integration) -> Any:
    ...
