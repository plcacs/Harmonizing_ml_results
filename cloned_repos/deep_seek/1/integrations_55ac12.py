import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, Optional, Union, List, Dict, Tuple, Set, TypeVar, Type, cast
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

OptionValidator = Callable[[str, str], Union[str, bool, None]]
META_CATEGORY: Dict[str, StrPromise] = {
    'meta-integration': gettext_lazy('Integration frameworks'), 
    'bots': gettext_lazy('Interactive bots')
}
CATEGORIES: Dict[str, StrPromise] = {
    **META_CATEGORY, 
    'continuous-integration': gettext_lazy('Continuous integration'), 
    'customer-support': gettext_lazy('Customer support'), 
    'deployment': gettext_lazy('Deployment'), 
    'entertainment': gettext_lazy('Entertainment'), 
    'communication': gettext_lazy('Communication'), 
    'financial': gettext_lazy('Financial'), 
    'hr': gettext_lazy('Human resources'), 
    'marketing': gettext_lazy('Marketing'), 
    'misc': gettext_lazy('Miscellaneous'), 
    'monitoring': gettext_lazy('Monitoring'), 
    'project-management': gettext_lazy('Project management'), 
    'productivity': gettext_lazy('Productivity'), 
    'version-control': gettext_lazy('Version control')
}

T = TypeVar('T', bound='Integration')

class Integration:
    DEFAULT_LOGO_STATIC_PATH_PNG: str = 'images/integrations/logos/{name}.png'
    DEFAULT_LOGO_STATIC_PATH_SVG: str = 'images/integrations/logos/{name}.svg'
    DEFAULT_BOT_AVATAR_PATH: str = 'images/integrations/bot_avatars/{name}.png'

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
        config_options: List[Any] = []
    ) -> None:
        self.name: str = name
        self.client_name: str = client_name if client_name is not None else name
        self.secondary_line_text: Optional[str] = secondary_line_text
        self.legacy: bool = legacy
        self.doc: Optional[str] = doc
        self.config_options: List[Any] = config_options
        for category in categories:
            if category not in CATEGORIES:
                raise KeyError('INTEGRATIONS: ' + name + " - category '" + category + "' is not a key in CATEGORIES.")
        self.categories: List[StrPromise] = [CATEGORIES[c] for c in categories]
        self.logo_path: Optional[str] = logo if logo is not None else self.get_logo_path()
        self.logo_url: Optional[str] = self.get_logo_url()
        if display_name is None:
            display_name = name.title()
        self.display_name: str = display_name
        if stream_name is None:
            stream_name = self.name
        self.stream_name: str = stream_name

    def is_enabled(self) -> bool:
        return True

    def get_logo_path(self) -> Optional[str]:
        logo_file_path_svg: str = self.DEFAULT_LOGO_STATIC_PATH_SVG.format(name=self.name)
        logo_file_path_png: str = self.DEFAULT_LOGO_STATIC_PATH_PNG.format(name=self.name)
        if os.path.isfile(static_path(logo_file_path_svg)):
            return logo_file_path_svg
        elif os.path.isfile(static_path(logo_file_path_png)):
            return logo_file_path_png
        return None

    def get_bot_avatar_path(self) -> Optional[str]:
        if self.logo_path is not None:
            name: str = os.path.splitext(os.path.basename(self.logo_path))[0]
            return self.DEFAULT_BOT_AVATAR_PATH.format(name=name)
        return None

    def get_logo_url(self) -> Optional[str]:
        if self.logo_path is not None:
            return staticfiles_storage.url(self.logo_path)
        return None

    def get_translated_categories(self) -> List[str]:
        return [str(category) for category in self.categories]

class BotIntegration(Integration):
    DEFAULT_LOGO_STATIC_PATH_PNG: str = 'generated/bots/{name}/logo.png'
    DEFAULT_LOGO_STATIC_PATH_SVG: str = 'generated/bots/{name}/logo.svg'
    ZULIP_LOGO_STATIC_PATH_PNG: str = 'images/logo/zulip-icon-128x128.png'
    DEFAULT_DOC_PATH: str = '{name}/doc.md'

    def __init__(
        self, 
        name: str, 
        categories: List[str], 
        logo: Optional[str] = None, 
        secondary_line_text: Optional[str] = None, 
        display_name: Optional[str] = None, 
        doc: Optional[str] = None
    ) -> None:
        super().__init__(name, client_name=name, categories=categories, secondary_line_text=secondary_line_text)
        if logo is None:
            self.logo_url: Optional[str] = self.get_logo_url()
            if self.logo_url is None:
                logo = staticfiles_storage.url(self.ZULIP_LOGO_STATIC_PATH_PNG)
        else:
            self.logo_url = staticfiles_storage.url(logo)
        if display_name is None:
            display_name = f'{name.title()} Bot'
        else:
            display_name = f'{display_name} Bot'
        self.display_name: str = display_name
        if doc is None:
            doc = self.DEFAULT_DOC_PATH.format(name=name)
        self.doc: str = doc

class WebhookIntegration(Integration):
    DEFAULT_FUNCTION_PATH: str = 'zerver.webhooks.{name}.view.api_{name}_webhook'
    DEFAULT_URL: str = 'api/v1/external/{name}'
    DEFAULT_CLIENT_NAME: str = 'Zulip{name}Webhook'
    DEFAULT_DOC_PATH: str = '{name}/doc.{ext}'

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
        config_options: List[Any] = [], 
        dir_name: Optional[str] = None
    ) -> None:
        if client_name is None:
            client_name = self.DEFAULT_CLIENT_NAME.format(name=name.title())
        super().__init__(
            name, 
            categories, 
            client_name=client_name, 
            logo=logo, 
            secondary_line_text=secondary_line_text, 
            display_name=display_name, 
            stream_name=stream_name, 
            legacy=legacy, 
            config_options=config_options
        )
        if function is None:
            function = self.DEFAULT_FUNCTION_PATH.format(name=name)
        self.function_name: str = function
        if url is None:
            url = self.DEFAULT_URL.format(name=name)
        self.url: str = url
        if doc is None:
            doc = self.DEFAULT_DOC_PATH.format(name=name, ext='md')
        self.doc: str = doc
        if dir_name is None:
            dir_name = self.name
        self.dir_name: str = dir_name

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
    return (integration_name, fixture_name)

@dataclass
class BaseScreenshotConfig:
    image_name: str = '001.png'
    image_dir: Optional[str] = None
    bot_name: Optional[str] = None

@dataclass
class ScreenshotConfig(BaseScreenshotConfig):
    payload_as_query_param: bool = False
    payload_param_name: str = 'payload'
    extra_params: Dict[str, str] = field(default_factory=dict)
    use_basic_auth: bool = False
    custom_headers: Dict[str, str] = field(default_factory=dict)

def get_fixture_and_image_paths(
    integration: Integration, 
    screenshot_config: ScreenshotConfig
) -> Tuple[str, str]:
    if isinstance(integration, WebhookIntegration):
        fixture_dir = os.path.join('zerver', 'webhooks', integration.dir_name, 'fixtures')
    else:
        fixture_dir = os.path.join('zerver', 'integration_fixtures', integration.name)
    fixture_path = os.path.join(fixture_dir, screenshot_config.fixture_name)
    image_dir = screenshot_config.image_dir or integration.name
    image_name = screenshot_config.image_name
    image_path = os.path.join('static/images/integrations', image_dir, image_name)
    return (fixture_path, image_path)

class HubotIntegration(Integration):
    GIT_URL_TEMPLATE: str = 'https://github.com/hubot-archive/hubot-{}'
    SECONDARY_LINE_TEXT: str = '(Hubot script)'
    DOC_PATH: str = 'zerver/integrations/hubot_common.md'

    def __init__(
        self, 
        name: str, 
        categories: List[str], 
        display_name: Optional[str] = None, 
        logo: Optional[str] = None, 
        git_url: Optional[str] = None, 
        legacy: bool = False
    ) -> None:
        if git_url is None:
            git_url = self.GIT_URL_TEMPLATE.format(name)
        self.hubot_docs_url: str = git_url
        super().__init__(
            name, 
            categories, 
            logo=logo, 
            secondary_line_text=self.SECONDARY_LINE_TEXT, 
            display_name=display_name, 
            doc=self.DOC_PATH, 
            legacy=legacy
        )

class EmbeddedBotIntegration(Integration):
    DEFAULT_CLIENT_NAME: str = 'Zulip{name}EmbeddedBot'

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        assert kwargs.get('client_name') is None
        kwargs['client_name'] = self.DEFAULT_CLIENT_NAME.format(name=name.title())
        super().__init__(name, *args, **kwargs)

EMBEDDED_BOTS: List[EmbeddedBotIntegration] = [
    EmbeddedBotIntegration('converter', []), 
    EmbeddedBotIntegration('encrypt', []), 
    EmbeddedBotIntegration('helloworld', []), 
    EmbeddedBotIntegration('virtual_fs', []), 
    EmbeddedBotIntegration('giphy', []), 
    EmbeddedBotIntegration('followup', [])
]

WEBHOOK_INTEGRATIONS: List[WebhookIntegration] = [
    WebhookIntegration('airbrake', ['monitoring']), 
    WebhookIntegration('airbyte', ['monitoring']), 
    WebhookIntegration('alertmanager', ['monitoring'], display_name='Prometheus Alertmanager', logo='images/integrations/logos/prometheus.svg'), 
    WebhookIntegration('ansibletower', ['deployment'], display_name='Ansible Tower'), 
    WebhookIntegration('appfollow', ['customer-support'], display_name='AppFollow'), 
    WebhookIntegration('appveyor', ['continuous-integration'], display_name='AppVeyor'), 
    WebhookIntegration('azuredevops', ['version-control'], display_name='AzureDevOps'), 
    WebhookIntegration('beanstalk', ['version-control'], stream_name='commits'), 
    WebhookIntegration('basecamp', ['project-management']), 
    WebhookIntegration('beeminder', ['misc'], display_name='Beeminder'), 
    WebhookIntegration('bitbucket3', ['version-control'], logo='images/integrations/logos/bitbucket.svg', display_name='Bitbucket Server', stream_name='bitbucket'), 
    WebhookIntegration('bitbucket2', ['version-control'], logo='images/integrations/logos/bitbucket.svg', display_name='Bitbucket', stream_name='bitbucket'), 
    WebhookIntegration('bitbucket', ['version-control'], display_name='Bitbucket', secondary_line_text='(Enterprise)', stream_name='commits', legacy=True), 
    WebhookIntegration('buildbot', ['continuous-integration']), 
    WebhookIntegration('canarytoken', ['monitoring'], display_name='Thinkst Canarytokens'), 
    WebhookIntegration('circleci', ['continuous-integration'], display_name='CircleCI'), 
    WebhookIntegration('clubhouse', ['project-management']), 
    WebhookIntegration('codeship', ['continuous-integration', 'deployment']), 
    WebhookIntegration('crashlytics', ['monitoring']), 
    WebhookIntegration('dialogflow', ['customer-support']), 
    WebhookIntegration('delighted', ['customer-support', 'marketing']), 
    WebhookIntegration('dropbox', ['productivity']), 
    WebhookIntegration('errbit', ['monitoring']), 
    WebhookIntegration('flock', ['customer-support']), 
    WebhookIntegration('freshdesk', ['customer-support']), 
    WebhookIntegration('freshping', ['monitoring']), 
    WebhookIntegration('freshstatus', ['monitoring', 'customer-support']), 
    WebhookIntegration('front', ['customer-support']), 
    WebhookIntegration('gitea', ['version-control'], stream_name='commits'), 
    WebhookIntegration('github', ['version-control'], display_name='GitHub', function='zerver.webhooks.github.view.api_github_webhook', stream_name='github', config_options=[
        WebhookConfigOption(name='branches', description='Filter by branches (comma-separated list)', validator=check_string), 
        WebhookConfigOption(name='ignore_private_repositories', description='Exclude notifications from private repositories', validator=check_bool)
    ]), 
    WebhookIntegration('githubsponsors', ['financial'], display_name='GitHub Sponsors', logo='images/integrations/logos/github.svg', dir_name='github', function='zerver.webhooks.github.view.api_github_webhook', doc='github/githubsponsors.md', stream_name='github'), 
    WebhookIntegration('gitlab', ['version-control'], display_name='GitLab'), 
    WebhookIntegration('gocd', ['continuous-integration'], display_name='GoCD'), 
    WebhookIntegration('gogs', ['version-control'], stream_name='commits'), 
    WebhookIntegration('gosquared', ['marketing'], display_name='GoSquared'), 
    WebhookIntegration('grafana', ['monitoring']), 
    WebhookIntegration('greenhouse', ['hr']), 
    WebhookIntegration('groove', ['customer-support']), 
    WebhookIntegration('harbor', ['deployment', 'productivity']), 
    WebhookIntegration('hellosign', ['productivity', 'hr'], display_name='HelloSign'), 
    WebhookIntegration('helloworld', ['misc'], display_name='Hello World'), 
    WebhookIntegration('heroku', ['deployment']), 
    WebhookIntegration('homeassistant', ['misc'], display_name='Home Assistant'), 
    WebhookIntegration('ifttt', ['meta-integration'], function='zerver.webhooks.ifttt.view.api_iftt_app_webhook', display_name='IFTTT'), 
    WebhookIntegration('insping', ['monitoring']), 
    WebhookIntegration('intercom', ['customer-support']), 
    WebhookIntegration('jira', ['project-management']), 
    WebhookIntegration('jotform', ['misc']), 
    WebhookIntegration('json', ['misc'], display_name='JSON formatter'), 
    WebhookIntegration('librato', ['monitoring']), 
    WebhookIntegration('lidarr', ['entertainment']), 
    WebhookIntegration('linear', ['project-management']), 
    WebhookIntegration('mention', ['marketing']), 
    WebhookIntegration('netlify', ['continuous-integration', 'deployment']), 
    WebhookIntegration('newrelic', ['monitoring'], display_name='New Relic'), 
    WebhookIntegration('opencollective', ['financial'], display_name='Open Collective'), 
    WebhookIntegration('opsgenie', ['meta-integration', 'monitoring']), 
    WebhookIntegration('pagerduty', ['monitoring'], display_name='PagerDuty'), 
    WebhookIntegration('papertrail', ['monitoring']), 
    WebhookIntegration('patreon', ['financial']), 
    WebhookIntegration('pingdom', ['monitoring']), 
    WebhookIntegration('pivotal', ['project-management'], display_name='Pivotal Tracker'), 
    WebhookIntegration('radarr', ['entertainment']), 
    WebhookIntegration('raygun', ['monitoring']), 
    WebhookIntegration('reviewboard', ['version