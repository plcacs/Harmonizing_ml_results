import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, Optional, Union, Dict, List, Tuple

from django.contrib.staticfiles.storage import staticfiles_storage
from django.http import HttpRequest, HttpResponseBase
from django.urls import URLPattern, path
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy
from django.views.decorators.csrf import csrf_exempt
from django_stubs_ext import StrPromise

from zerver.lib.storage import static_path
from zerver.lib.validator import check_bool, check_string
from zerver.lib.webhooks.common import WebhookConfigOption

OptionValidator: TypeAlias = Callable[[str, str], Union[str, bool, None]]

META_CATEGORY: Dict[str, StrPromise] = {
    "meta-integration": gettext_lazy("Integration frameworks"),
    "bots": gettext_lazy("Interactive bots"),
}

CATEGORIES: Dict[str, StrPromise] = {
    **META_CATEGORY,
    "continuous-integration": gettext_lazy("Continuous integration"),
    "customer-support": gettext_lazy("Customer support"),
    "deployment": gettext_lazy("Deployment"),
    "entertainment": gettext_lazy("Entertainment"),
    "communication": gettext_lazy("Communication"),
    "financial": gettext_lazy("Financial"),
    "hr": gettext_lazy("Human resources"),
    "marketing": gettext_lazy("Marketing"),
    "misc": gettext_lazy("Miscellaneous"),
    "monitoring": gettext_lazy("Monitoring"),
    "project-management": gettext_lazy("Project management"),
    "productivity": gettext_lazy("Productivity"),
    "version-control": gettext_lazy("Version control"),
}

@dataclass
class BaseScreenshotConfig:
    fixture_name: str
    image_name: str = "001.png"
    image_dir: Optional[str] = None
    bot_name: Optional[str] = None

@dataclass
class ScreenshotConfig(BaseScreenshotConfig):
    payload_as_query_param: bool = False
    payload_param_name: str = "payload"
    extra_params: Dict[str, str] = field(default_factory=dict)
    use_basic_auth: bool = False
    custom_headers: Dict[str, str] = field(default_factory=dict)

class Integration:
    DEFAULT_LOGO_STATIC_PATH_PNG: str = "images/integrations/logos/{name}.png"
    DEFAULT_LOGO_STATIC_PATH_SVG: str = "images/integrations/logos/{name}.svg"
    DEFAULT_BOT_AVATAR_PATH: str = "images/integrations/bot_avatars/{name}.png"

    def __init__(
        self,
        name: str,
        categories: List[str],
        client_name: Optional[str] = None,
        logo: Optional[str] = None,
        secondary_line_text: Optional[str] = None,
        display_name: Optional[str] = None,
        doc: Optional[str] = None,
        stream_name: Optional[str] = None,
        legacy: bool = False,
        config_options: Sequence[WebhookConfigOption] = [],
    ) -> None:
        self.name = name
        self.client_name = client_name if client_name is not None else name
        self.secondary_line_text = secondary_line_text
        self.legacy = legacy
        self.doc = doc
        self.config_options = config_options
        self.categories = [CATEGORIES[c] for c in categories]
        self.logo_path = logo if logo is not None else self.get_logo_path()
        self.logo_url = self.get_logo_url()
        self.display_name = display_name if display_name is not None else name.title()
        self.stream_name = stream_name if stream_name is not None else self.name

    def is_enabled(self) -> bool:
        return True

    def get_logo_path(self) -> Optional[str]:
        logo_file_path_svg = self.DEFAULT_LOGO_STATIC_PATH_SVG.format(name=self.name)
        logo_file_path_png = self.DEFAULT_LOGO_STATIC_PATH_PNG.format(name=self.name)
        if os.path.isfile(static_path(logo_file_path_svg)):
            return logo_file_path_svg
        elif os.path.isfile(static_path(logo_file_path_png)):
            return logo_file_path_png
        return None

    def get_bot_avatar_path(self) -> Optional[str]:
        if self.logo_path is not None:
            name = os.path.splitext(os.path.basename(self.logo_path))[0]
            return self.DEFAULT_BOT_AVATAR_PATH.format(name=name)
        return None

    def get_logo_url(self) -> Optional[str]:
        if self.logo_path is not None:
            return staticfiles_storage.url(self.logo_path)
        return None

    def get_translated_categories(self) -> List[str]:
        return [str(category) for category in self.categories]

class BotIntegration(Integration):
    DEFAULT_LOGO_STATIC_PATH_PNG: str = "generated/bots/{name}/logo.png"
    DEFAULT_LOGO_STATIC_PATH_SVG: str = "generated/bots/{name}/logo.svg"
    ZULIP_LOGO_STATIC_PATH_PNG: str = "images/logo/zulip-icon-128x128.png"
    DEFAULT_DOC_PATH: str = "{name}/doc.md"

    def __init__(
        self,
        name: str,
        categories: List[str],
        logo: Optional[str] = None,
        secondary_line_text: Optional[str] = None,
        display_name: Optional[str] = None,
        doc: Optional[str] = None,
    ) -> None:
        super().__init__(
            name,
            client_name=name,
            categories=categories,
            secondary_line_text=secondary_line_text,
        )
        self.logo_url = staticfiles_storage.url(logo) if logo is not None else self.get_logo_url()
        if self.logo_url is None and logo is None:
            self.logo_url = staticfiles_storage.url(self.ZULIP_LOGO_STATIC_PATH_PNG)
        self.display_name = f"{display_name if display_name is not None else name.title()} Bot"
        self.doc = doc if doc is not None else self.DEFAULT_DOC_PATH.format(name=name)

class WebhookIntegration(Integration):
    DEFAULT_FUNCTION_PATH: str = "zerver.webhooks.{name}.view.api_{name}_webhook"
    DEFAULT_URL: str = "api/v1/external/{name}"
    DEFAULT_CLIENT_NAME: str = "Zulip{name}Webhook"
    DEFAULT_DOC_PATH: str = "{name}/doc.{ext}"

    def __init__(
        self,
        name: str,
        categories: List[str],
        client_name: Optional[str] = None,
        logo: Optional[str] = None,
        secondary_line_text: Optional[str] = None,
        function: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        doc: Optional[str] = None,
        stream_name: Optional[str] = None,
        legacy: bool = False,
        config_options: Sequence[WebhookConfigOption] = [],
        dir_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name,
            categories,
            client_name=client_name if client_name is not None else self.DEFAULT_CLIENT_NAME.format(name=name.title()),
            logo=logo,
            secondary_line_text=secondary_line_text,
            display_name=display_name,
            stream_name=stream_name,
            legacy=legacy,
            config_options=config_options,
        )
        self.function_name = function if function is not None else self.DEFAULT_FUNCTION_PATH.format(name=name)
        self.url = url if url is not None else self.DEFAULT_URL.format(name=name)
        self.doc = doc if doc is not None else self.DEFAULT_DOC_PATH.format(name=name, ext="md")
        self.dir_name = dir_name if dir_name is not None else self.name

    def get_function(self) -> Callable[[HttpRequest], HttpResponseBase]:
        return import_string(self.function_name)

    @csrf_exempt
    def view(self, request: HttpRequest) -> HttpResponseBase:
        function = self.get_function()
        assert function.csrf_exempt
        return function(request)

    @property
    def url_object(self) -> URLPattern:
        return path(self.url, self.view)

def split_fixture_path(path: str) -> Tuple[str, str]:
    path, fixture_name = os.path.split(path)
    fixture_name, _ = os.path.splitext(fixture_name)
    integration_name = os.path.split(os.path.dirname(path))[-1]
    return integration_name, fixture_name

class HubotIntegration(Integration):
    GIT_URL_TEMPLATE: str = "https://github.com/hubot-archive/hubot-{}"
    SECONDARY_LINE_TEXT: str = "(Hubot script)"
    DOC_PATH: str = "zerver/integrations/hubot_common.md"

    def __init__(
        self,
        name: str,
        categories: List[str],
        display_name: Optional[str] = None,
        logo: Optional[str] = None,
        git_url: Optional[str] = None,
        legacy: bool = False,
    ) -> None:
        self.hubot_docs_url = git_url if git_url is not None else self.GIT_URL_TEMPLATE.format(name)
        super().__init__(
            name,
            categories,
            logo=logo,
            secondary_line_text=self.SECONDARY_LINE_TEXT,
            display_name=display_name,
            doc=self.DOC_PATH,
            legacy=legacy,
        )

class EmbeddedBotIntegration(Integration):
    DEFAULT_CLIENT_NAME: str = "Zulip{name}EmbeddedBot"

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        assert kwargs.get("client_name") is None
        kwargs["client_name"] = self.DEFAULT_CLIENT_NAME.format(name=name.title())
        super().__init__(name, *args, **kwargs)

EMBEDDED_BOTS: List[EmbeddedBotIntegration] = [
    EmbeddedBotIntegration("converter", []),
    EmbeddedBotIntegration("encrypt", []),
    EmbeddedBotIntegration("helloworld", []),
    EmbeddedBotIntegration("virtual_fs", []),
    EmbeddedBotIntegration("giphy", []),
    EmbeddedBotIntegration("followup", []),
]

WEBHOOK_INTEGRATIONS: List[WebhookIntegration] = [
    # ... (rest of the WEBHOOK_INTEGRATIONS list remains the same)
]

INTEGRATIONS: Dict[str, Integration] = {
    # ... (rest of the INTEGRATIONS dict remains the same)
}

BOT_INTEGRATIONS: List[BotIntegration] = [
    # ... (rest of the BOT_INTEGRATIONS list remains the same)
]

HUBOT_INTEGRATIONS: List[HubotIntegration] = [
    # ... (rest of the HUBOT_INTEGRATIONS list remains the same)
]

for hubot_integration in HUBOT_INTEGRATIONS:
    INTEGRATIONS[hubot_integration.name] = hubot_integration

for webhook_integration in WEBHOOK_INTEGRATIONS:
    INTEGRATIONS[webhook_integration.name] = webhook_integration

for bot_integration in BOT_INTEGRATIONS:
    INTEGRATIONS[bot_integration.name] = bot_integration

NO_SCREENSHOT_WEBHOOKS: Set[str] = {
    "beeminder",
    "ifttt",
    "slack_incoming",
    "zapier",
}

DOC_SCREENSHOT_CONFIG: Dict[str, List[BaseScreenshotConfig]] = {
    # ... (rest of the DOC_SCREENSHOT_CONFIG dict remains the same)
}

def get_fixture_and_image_paths(
    integration: Integration, screenshot_config: BaseScreenshotConfig
) -> Tuple[str, str]:
    if isinstance(integration, WebhookIntegration):
        fixture_dir = os.path.join("zerver", "webhooks", integration.dir_name, "fixtures")
    else:
        fixture_dir = os.path.join("zerver", "integration_fixtures", integration.name)
    fixture_path = os.path.join(fixture_dir, screenshot_config.fixture_name)
    image_dir = screenshot_config.image_dir or integration.name
    image_name = screenshot_config.image_name
    image_path = os.path.join("static/images/integrations", image_dir, image_name)
    return fixture_path, image_path

def get_all_event_types_for_integration(integration: Integration) -> Optional[List[str]]:
    integration = INTEGRATIONS[integration.name]
    if isinstance(integration, WebhookIntegration):
        if integration.name == "githubsponsors":
            return import_string("zerver.webhooks.github.view.SPONSORS_EVENT_TYPES")
        function = integration.get_function()
        if hasattr(function, "_all_event_types"):
            return function._all_event_types
    return None
