import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TypeAlias, Union
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

OptionValidator: TypeAlias = Callable[[str, str], Optional[Union[str, bool]]]
META_CATEGORY: Dict[str, StrPromise] = {'meta-integration': gettext_lazy('Integration frameworks'), 'bots': gettext_lazy('Interactive bots')}
CATEGORIES: Dict[str, StrPromise] = {**META_CATEGORY, 'continuous-integration': gettext_lazy('Continuous integration'), 'customer-support': gettext_lazy('Customer support'), 'deployment': gettext_lazy('Deployment'), 'entertainment': gettext_lazy('Entertainment'), 'communication': gettext_lazy('Communication'), 'financial': gettext_lazy('Financial'), 'hr': gettext_lazy('Human resources'), 'marketing': gettext_lazy('Marketing'), 'misc': gettext_lazy('Miscellaneous'), 'monitoring': gettext_lazy('Monitoring'), 'project-management': gettext_lazy('Project management'), 'productivity': gettext_lazy('Productivity'), 'version-control': gettext_lazy('Version control')}

class Integration:
    DEFAULT_LOGO_STATIC_PATH_PNG = 'images/integrations/logos/{name}.png'
    DEFAULT_LOGO_STATIC_PATH_SVG = 'images/integrations/logos/{name}.svg'
    DEFAULT_BOT_AVATAR_PATH = 'images/integrations/bot_avatars/{name}.png'

    def __init__(self, name: str, categories: List[str], client_name: Optional[str]=None, logo: Optional[str]=None, secondary_line_text: Optional[str]=None, display_name: Optional[str]=None, doc: Optional[str]=None, stream_name: Optional[str]=None, legacy: bool=False, config_options: List[WebhookConfigOption]=[]) -> None:
        self.name = name
        self.client_name = client_name if client_name is not None else name
        self.secondary_line_text = secondary_line_text
        self.legacy = legacy
        self.doc = doc
        self.config_options = config_options
        for category in categories:
            if category not in CATEGORIES:
                raise KeyError('INTEGRATIONS: ' + name + " - category '" + category + "' is not a key in CATEGORIES.")
        self.categories = [CATEGORIES[c] for c in categories]
        self.logo_path = logo if logo is not None else self.get_logo_path()
        self.logo_url = self.get_logo_url()
        if display_name is None:
            display_name = name.title()
        self.display_name = display_name
        if stream_name is None:
            stream_name = self.name
        self.stream_name = stream_name

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
    DEFAULT_LOGO_STATIC_PATH_PNG = 'generated/bots/{name}/logo.png'
    DEFAULT_LOGO_STATIC_PATH_SVG = 'generated/bots/{name}/logo.svg'
    ZULIP_LOGO_STATIC_PATH_PNG = 'images/logo/zulip-icon-128x128.png'
    DEFAULT_DOC_PATH = '{name}/doc.md'

    def __init__(self, name: str, categories: List[str], logo: Optional[str]=None, secondary_line_text: Optional[str]=None, display_name: Optional[str]=None, doc: Optional[str]=None) -> None:
        super().__init__(name, client_name=name, categories=categories, secondary_line_text=secondary_line_text)
        if logo is None:
            self.logo_url = self.get_logo_url()
            if self.logo_url is None:
                logo = staticfiles_storage.url(self.ZULIP_LOGO_STATIC_PATH_PNG)
        else:
            self.logo_url = staticfiles_storage.url(logo)
        if display_name is None:
            display_name = f'{name.title()} Bot'
        else:
            display_name = f'{display_name} Bot'
        self.display_name = display_name
        if doc is None:
            doc = self.DEFAULT_DOC_PATH.format(name=name)
        self.doc = doc

class WebhookIntegration(Integration):
    DEFAULT_FUNCTION_PATH = 'zerver.webhooks.{name}.view.api_{name}_webhook'
    DEFAULT_URL = 'api/v1/external/{name}'
    DEFAULT_CLIENT_NAME = 'Zulip{name}Webhook'
    DEFAULT_DOC_PATH = '{name}/doc.{ext}'

    def __init__(self, name: str, categories: List[str], client_name: Optional[str]=None, logo: Optional[str]=None, secondary_line_text: Optional[str]=None, function: Optional[str]=None, url: Optional[str]=None, display_name: Optional[str]=None, doc: Optional[str]=None, stream_name: Optional[str]=None, legacy: bool=False, config_options: List[WebhookConfigOption]=[], dir_name: Optional[str]=None) -> None:
        if client_name is None:
            client_name = self.DEFAULT_CLIENT_NAME.format(name=name.title())
        super().__init__(name, categories, client_name=client_name, logo=logo, secondary_line_text=secondary_line_text, display_name=display_name, stream_name=stream_name, legacy=legacy, config_options=config_options)
        if function is None:
            function = self.DEFAULT_FUNCTION_PATH.format(name=name)
        self.function_name = function
        if url is None:
            url = self.DEFAULT_URL.format(name=name)
        self.url = url
        if doc is None:
            doc = self.DEFAULT_DOC_PATH.format(name=name, ext='md')
        self.doc = doc
        if dir_name is None:
            dir_name = self.name
        self.dir_name = dir_name

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

def get_fixture_and_image_paths(integration: Integration, screenshot_config: BaseScreenshotConfig) -> Tuple[str, str]:
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
    GIT_URL_TEMPLATE = 'https://github.com/hubot-archive/hubot-{}'
    SECONDARY_LINE_TEXT = '(Hubot script)'
    DOC_PATH = 'zerver/integrations/hubot_common.md'

    def __init__(self, name: str, categories: List[str], display_name: Optional[str]=None, logo: Optional[str]=None, git_url: Optional[str]=None, legacy: bool=False) -> None:
        if git_url is None:
            git_url = self.GIT_URL_TEMPLATE.format(name)
        self.hubot_docs_url = git_url
        super().__init__(name, categories, logo=logo, secondary_line_text=self.SECONDARY_LINE_TEXT, display_name=display_name, doc=self.DOC_PATH, legacy=legacy)

class EmbeddedBotIntegration(Integration):
    """
    This class acts as a registry for bots verified as safe
    and valid such that these are capable of being deployed on the server.
    """
    DEFAULT_CLIENT_NAME = 'Zulip{name}EmbeddedBot'

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        assert kwargs.get('client_name') is None
        kwargs['client_name'] = self.DEFAULT_CLIENT_NAME.format(name=name.title())
        super().__init__(name, *args, **kwargs)

EMBEDDED_BOTS: List[EmbeddedBotIntegration] = [EmbeddedBotIntegration('converter', []), EmbeddedBotIntegration('encrypt', []), EmbeddedBotIntegration('helloworld', []), EmbeddedBotIntegration('virtual_fs', []), EmbeddedBotIntegration('giphy', []), EmbeddedBotIntegration('followup', [])]
WEBHOOK_INTEGRATIONS: List[WebhookIntegration] = [WebhookIntegration('airbrake', ['monitoring']), WebhookIntegration('airbyte', ['monitoring']), WebhookIntegration('alertmanager', ['monitoring'], display_name='Prometheus Alertmanager', logo='images/integrations/logos/prometheus.svg'), WebhookIntegration('ansibletower', ['deployment'], display_name='Ansible Tower'), WebhookIntegration('appfollow', ['customer-support'], display_name='AppFollow'), WebhookIntegration('appveyor', ['continuous-integration'], display_name='AppVeyor'), WebhookIntegration('azuredevops', ['version-control'], display_name='AzureDevOps'), WebhookIntegration('beanstalk', ['version-control'], stream_name='commits'), WebhookIntegration('basecamp', ['project-management']), WebhookIntegration('beeminder', ['misc'], display_name='Beeminder'), WebhookIntegration('bitbucket3', ['version-control'], logo='images/integrations/logos/bitbucket.svg', display_name='Bitbucket Server', stream_name='bitbucket'), WebhookIntegration('bitbucket2', ['version-control'], logo='images/integrations/logos/bitbucket.svg', display_name='Bitbucket', stream_name='bitbucket'), WebhookIntegration('bitbucket', ['version-control'], display_name='Bitbucket', secondary_line_text='(Enterprise)', stream_name='commits', legacy=True), WebhookIntegration('buildbot', ['continuous-integration']), WebhookIntegration('canarytoken', ['monitoring'], display_name='Thinkst Canarytokens'), WebhookIntegration('circleci', ['continuous-integration'], display_name='CircleCI'), WebhookIntegration('clubhouse', ['project-management']), WebhookIntegration('codeship', ['continuous-integration', 'deployment']), WebhookIntegration('crashlytics', ['monitoring']), WebhookIntegration('dialogflow', ['customer-support']), WebhookIntegration('delighted', ['customer-support', 'marketing']), WebhookIntegration('dropbox', ['productivity']), WebhookIntegration('errbit', ['monitoring']), WebhookIntegration('flock', ['customer-support']), WebhookIntegration('freshdesk', ['customer-support']), WebhookIntegration('freshping', ['monitoring']), WebhookIntegration('freshstatus', ['monitoring', 'customer-support']), WebhookIntegration('front', ['customer-support']), WebhookIntegration('gitea', ['version-control'], stream_name='commits'), WebhookIntegration('github', ['version-control'], display_name='GitHub', function='zerver.webhooks.github.view.api_github_webhook', stream_name='github', config_options=[WebhookConfigOption(name='branches', description='Filter by branches (comma-separated list)', validator=check_string), WebhookConfigOption(name='ignore_private_repositories', description='Exclude notifications from private repositories', validator=check_bool)]), WebhookIntegration('githubsponsors', ['financial'], display_name='GitHub Sponsors', logo='images/integrations/logos/github.svg', dir_name='github', function='zerver.webhooks.github.view.api_github_webhook', doc='github/githubsponsors.md', stream_name='github'), WebhookIntegration('gitlab', ['version-control'], display_name='GitLab'), WebhookIntegration('gocd', ['continuous-integration'], display_name='GoCD'), WebhookIntegration('gogs', ['version-control'], stream_name='commits'), WebhookIntegration('gosquared', ['marketing'], display_name='GoSquared'), WebhookIntegration('grafana', ['monitoring']), WebhookIntegration('greenhouse', ['hr']), WebhookIntegration('groove', ['customer-support']), WebhookIntegration('harbor', ['deployment', 'productivity']), WebhookIntegration('hellosign', ['productivity', 'hr'], display_name='HelloSign'), WebhookIntegration('helloworld', ['misc'], display_name='Hello World'), WebhookIntegration('heroku', ['deployment']), WebhookIntegration('homeassistant', ['misc'], display_name='Home Assistant'), WebhookIntegration('ifttt', ['meta-integration'], function='zerver.webhooks.ifttt.view.api_iftt_app_webhook', display_name='IFTTT'), WebhookIntegration('insping', ['monitoring']), WebhookIntegration('intercom', ['customer-support']), WebhookIntegration('jira', ['project-management']), WebhookIntegration('jotform', ['misc']), WebhookIntegration('json', ['misc'], display_name='JSON formatter'), WebhookIntegration('librato', ['monitoring']), WebhookIntegration('lidarr', ['entertainment']), WebhookIntegration('linear', ['project-management']), WebhookIntegration('mention', ['marketing']), WebhookIntegration('netlify', ['continuous-integration', 'deployment']), WebhookIntegration('newrelic', ['monitoring'], display_name='New Relic'), WebhookIntegration('opencollective', ['financial'], display_name='Open Collective'), WebhookIntegration('opsgenie', ['meta-integration', 'monitoring']), WebhookIntegration('pagerduty', ['monitoring'], display_name='PagerDuty'), WebhookIntegration('papertrail', ['monitoring']), WebhookIntegration('patreon', ['financial']), WebhookIntegration('pingdom', ['monitoring']), WebhookIntegration('pivotal', ['project-management'], display_name='Pivotal Tracker'), WebhookIntegration('radarr', ['entertainment']), WebhookIntegration('raygun', ['monitoring']), WebhookIntegration('reviewboard', ['version-control'], display_name='Review Board'), WebhookIntegration('rhodecode', ['version-control'], display_name='RhodeCode'), WebhookIntegration('rundeck', ['deployment']), WebhookIntegration('semaphore', ['continuous-integration', 'deployment']), WebhookIntegration('sentry', ['monitoring']), WebhookIntegration('slack_incoming', ['communication', 'meta-integration'], display_name='Slack-compatible webhook', logo='images/integrations/logos/slack.svg'), WebhookIntegration('slack', ['communication']), WebhookIntegration('sonarqube', ['continuous-integration'], display_name='SonarQube'), WebhookIntegration('sonarr', ['entertainment']), WebhookIntegration('splunk', ['monitoring']), WebhookIntegration('statuspage', ['customer-support']), WebhookIntegration('stripe', ['financial']), WebhookIntegration('taiga', ['project-management']), WebhookIntegration('teamcity', ['continuous-integration']), WebhookIntegration('thinkst', ['monitoring']), WebhookIntegration('transifex', ['misc']), WebhookIntegration('travis', ['continuous-integration'], display_name='Travis CI'), WebhookIntegration('trello', ['project-management']), WebhookIntegration('updown', ['monitoring']), WebhookIntegration('uptimerobot', ['monitoring'], display_name='UptimeRobot'), WebhookIntegration('wekan', ['productivity']), WebhookIntegration('wordpress', ['marketing'], display_name='WordPress'), WebhookIntegration('zapier', ['meta-integration']), WebhookIntegration('zendesk', ['customer-support']), WebhookIntegration('zabbix', ['monitoring'])]
INTEGRATIONS: Dict[str, Integration] = {'asana': Integration('asana', ['project-management'], doc='zerver/integrations/asana.md'), 'big-blue-button': Integration('big-blue-button', ['communication'], logo='images/integrations/logos/bigbluebutton.svg', display_name='BigBlueButton', doc='zerver/integrations/big-blue-button.md'), 'capistrano': Integration('capistrano', ['deployment'], display_name='Capistrano', doc='zerver/integrations/capistrano.md'), 'codebase': Integration('codebase', ['version-control'], doc='zerver/integrations/codebase.md'), 'discourse': Integration('discourse', ['communication'], doc='zerver/integrations/discourse.md'), 'email': Integration('email', ['communication'], doc='zerver/integrations/email.md'), 'errbot': Integration('errbot', ['meta-integration', 'bots'], doc='zerver/integrations/errbot.md'), 'giphy': Integration('giphy', display_name='GIPHY', categories=['misc'], doc='zerver/integrations/giphy.md'), 'git': Integration('git', ['version-control'], stream_name='commits', doc='zerver/integrations/git.md'), 'github-actions': Integration('github-actions', ['continuous-integration'], display_name='GitHub Actions', doc='zerver/integrations/github-actions.md'), 'google-calendar': Integration('google-calendar', ['productivity'], display_name='Google Calendar', doc='zerver/integrations/google-calendar.md'), 'hubot': Integration('hubot', ['meta-integration', 'bots'], doc='zerver/integrations/hubot.md'), 'irc': Integration('irc', ['communication'], display_name='IRC', doc='zerver/integrations/irc.md'), 'jenkins': Integration('jenkins', ['continuous-integration'], doc='zerver/integrations/jenkins.md'), 'jira-plugin': Integration('jira-plugin', ['project-management'], logo='images/integrations/logos/jira.svg', secondary_line_text='(locally installed)', display_name='Jira', doc='zerver/integrations/jira-plugin.md', stream_name='jira', legacy=True), 'jitsi': Integration('jitsi', ['communication'], display_name='Jitsi Meet', doc='zerver/integrations/jitsi.md'), 'mastodon': Integration('mastodon', ['communication'], doc='zerver/integrations/mastodon.md'), 'matrix': Integration('matrix', ['communication'], doc='zerver/integrations/matrix.md'), 'mercurial': Integration('mercurial', ['version-control'], display_name='Mercurial (hg)', doc='zerver/integrations/mercurial.md', stream_name='commits'), 'nagios': Integration('nagios', ['monitoring'], doc='zerver/integrations/nagios.md'), 'notion': Integration('notion', ['productivity'], doc='zerver/integrations/notion.md'), 'openshift': Integration('openshift', ['deployment'], display_name='OpenShift', doc='zerver/integrations/openshift.md', stream_name='deployments'), 'onyx': Integration('onyx', ['productivity'], logo='images/integrations/logos/onyx.png', doc='zerver/integrations/onyx.md'), 'perforce': Integration('perforce', ['version-control'], doc='zerver/integrations/perforce.md'), 'phabricator': Integration('phabricator', ['version-control'], doc='zerver/integrations/phabricator.md'), 'puppet': Integration('puppet', ['deployment'], doc='zerver/integrations/puppet.md'), 'redmine': Integration('redmine', ['project-management'], doc='zerver/integrations/redmine.md'), 'rss': Integration('rss', ['communication'], display_name='RSS', doc='zerver/integrations/rss.md'), 'svn': Integration('svn', ['version-control'], display_name='Subversion', doc='zerver/integrations/svn.md'), 'trac': Integration('trac', ['project-management'], doc='zerver/integrations/trac.md'), 'twitter': Integration('twitter', ['customer-support', 'marketing'], logo='images/integrations/logos/twitte_r.svg', doc='zerver/integrations/twitter.md'), 'zoom': Integration('zoom', ['communication'], doc='zerver/integrations/zoom.md')}
BOT_INTEGRATIONS: List[BotIntegration] = [BotIntegration('github_detail', ['version-control', 'bots'], display_name='GitHub Detail'), BotIntegration('xkcd', ['bots', 'misc'], display_name='xkcd', logo='images/integrations/logos/xkcd.png')]
HUBOT_INTEGRATIONS: List[HubotIntegration] = [HubotIntegration('assembla', ['version-control', 'project-management']), HubotIntegration('bonusly', ['hr']), HubotIntegration('chartbeat', ['marketing']), HubotIntegration('darksky', ['misc'], display_name='Dark Sky'), HubotIntegration('instagram', ['misc'], logo='images/integrations/logos/instagra_m.svg'), HubotIntegration('mailchimp', ['communication', 'marketing']), HubotIntegration('google-translate', ['misc'], display_name='Google Translate'), HubotIntegration('youtube', ['misc'], display_name='YouTube', logo='images/integrations/logos/youtub_e.svg')]
for hubot_integration in HUBOT_INTEGRATIONS:
    INTEGRATIONS[hubot_integration.name] = hubot_integration
for webhook_integration in WEBHOOK_INTEGRATIONS:
    INTEGRATIONS[webhook_integration.name] = webhook_integration
for bot_integration in BOT_INTEGRATIONS:
    INTEGRATIONS[bot_integration.name] = bot_integration
NO_SCREENSHOT_WEBHOOKS: Set[str] = {'beeminder', 'ifttt', 'slack_incoming', 'zapier'}
DOC_SCREENSHOT_CONFIG: Dict[str, List[ScreenshotConfig]] = {'airbrake': [ScreenshotConfig('error_message.json')], 'airbyte': [ScreenshotConfig('airbyte_job_payload_success.json')], 'alertmanager': [ScreenshotConfig('alert.json', extra_params={'name': 'topic', 'desc': 'description'})], 'ansibletower': [ScreenshotConfig('job_successful_multiple_hosts.json')], 'appfollow': [ScreenshotConfig('review.json')], 'appveyor': [ScreenshotConfig('appveyor_build_success.json')], 'azuredevops': [ScreenshotConfig('code_push.json')], 'basecamp': [ScreenshotConfig('doc_active.json')], 'beanstalk': [ScreenshotConfig('git_multiple.json', use_basic_auth=True, payload_as_query_param=True)], 'bitbucket': [ScreenshotConfig('push.json', '002.png', use_basic_auth=True, payload_as_query_param=True)], 'bitbucket2': [ScreenshotConfig('issue_created.json', '003.png', 'bitbucket', bot_name='Bitbucket Bot')], 'bitbucket3': [ScreenshotConfig('repo_push_update_single_branch.json', '004.png', 'bitbucket', bot_name='Bitbucket Server Bot')], 'buildbot': [ScreenshotConfig('started.json')], 'canarytoken': [ScreenshotConfig('canarytoken_real.json')], 'circleci': [ScreenshotConfig('github_job_completed.json')], 'clubhouse': [ScreenshotConfig('story_create.json')], 'codeship': [ScreenshotConfig('error_build.json')], 'crashlytics': [ScreenshotConfig('issue_message.json')], 'delighted': [ScreenshotConfig('survey_response_updated_promoter.json')], 'dialogflow': [ScreenshotConfig('weather_app.json', extra_params={'email': 'iago@zulip.com'})], 'dropbox': [ScreenshotConfig('file_updated.json')], 'errbit': [ScreenshotConfig('error_message.json')], 'flock': [ScreenshotConfig('messages.json')], 'freshdesk': [ScreenshotConfig('ticket_created.json', image_name='004.png', use_basic_auth=True)], 'freshping': [ScreenshotConfig('freshping_check_unreachable.json')], 'freshstatus': [ScreenshotConfig('freshstatus_incident_open.json')], 'front': [ScreenshotConfig('inbound_message.json')], 'gitea': [ScreenshotConfig('pull_request__merged.json')], 'github': [ScreenshotConfig('push__1_commit.json')], 'githubsponsors': [ScreenshotConfig('created.json')], 'gitlab': [ScreenshotConfig('push_hook__push_local_branch_without_commits.json')], 'gocd': [ScreenshotConfig('pipeline_with_mixed_job_result.json')], 'gogs': [ScreenshotConfig('pull_request__opened.json')], 'gosquared': [ScreenshotConfig('traffic_spike.json')], 'grafana': [ScreenshotConfig('alert_values_v11.json')], 'greenhouse': [ScreenshotConfig('candidate_stage_change.json')], 'groove': [ScreenshotConfig('ticket_started.json')], 'harbor': [ScreenshotConfig('scanning_completed.json')], 'hellosign': [ScreenshotConfig('signatures_signed_by_one_signatory.json', payload_as_query_param=True, payload_param_name='json')], 'helloworld': [ScreenshotConfig('hello.json')], 'heroku': [ScreenshotConfig('deploy.txt')], 'homeassistant': [ScreenshotConfig('reqwithtitle.json')], 'insping': [ScreenshotConfig('website_state_available.json')], 'intercom': [ScreenshotConfig('conversation_admin_replied.json')], 'jira': [ScreenshotConfig('created_v1.json')], 'jotform': [ScreenshotConfig('response.multipart')], 'json': [ScreenshotConfig('json_github_push__1_commit.json')], 'librato': [ScreenshotConfig('three_conditions_alert.json', payload_as_query_param=True)], 'lidarr': [ScreenshotConfig('lidarr_album_grabbed.json')], 'linear': [ScreenshotConfig('issue_create_complex.json')], 'mention': [ScreenshotConfig('webfeeds.json')], 'nagios': [BaseScreenshotConfig('service_notify.json')], 'netlify': [ScreenshotConfig('deploy_building.json')], 'newrelic': [ScreenshotConfig('incident_activated_new_default_payload.json')], 'opencollective': [ScreenshotConfig('one_time_donation.json')], 'opsgenie': [ScreenshotConfig('addrecipient.json')], 'pagerduty': [ScreenshotConfig('trigger_v2.json')], 'papertrail': [ScreenshotConfig('short_post.json', payload_as_query_param=True)], 'patreon': [ScreenshotConfig('members_pledge_create.json')], 'pingdom': [ScreenshotConfig('http_up_to_down.json')], 'pivotal': [ScreenshotConfig('v5_type_changed.json')], 'radarr': [ScreenshotConfig('radarr_movie_grabbed.json')], 'raygun': [ScreenshotConfig('new_error.json')], 'reviewboard': [ScreenshotConfig('review_request_published.json')], 'rhodecode': [ScreenshotConfig('push.json')], 'rundeck': [ScreenshotConfig('start.json')], 'semaphore': [ScreenshotConfig('pull_request.json')], 'sentry': [ScreenshotConfig('event_for_exception_python.json')], 'slack': [ScreenshotConfig('message_with_normal_text.json')], 'sonarqube': [ScreenshotConfig('error.json')], 'sonarr': [ScreenshotConfig('sonarr_episode_grabbed.json')], 'splunk': [ScreenshotConfig('search_one_result.json')], 'statuspage': [ScreenshotConfig('incident_created.json')], 'stripe': [ScreenshotConfig('charge_succeeded__card.json')], 'taiga': [ScreenshotConfig('userstory_changed_status.json')], 'teamcity': [ScreenshotConfig('success.json')], 'thinkst': [ScreenshotConfig('canary_consolidated_port_scan.json')], 'transifex': [ScreenshotConfig('', extra_params={'project': 'Zulip Mobile', 'language': 'en', 'resource': 'file', 'reviewed': '100'})], 'travis': [ScreenshotConfig('build.json', payload_as_query_param=True)], 'trello': [ScreenshotConfig('adding_comment_to_card.json')], 'updown': [ScreenshotConfig('check_multiple_events.json')], 'uptimerobot': [ScreenshotConfig('uptimerobot_monitor_up.json')], 'wekan': [ScreenshotConfig('add_comment.json')], 'wordpress': [ScreenshotConfig('publish_post.txt', 'wordpress_post_created.png')], 'zabbix': [ScreenshotConfig('zabbix_alert.json')], 'zendesk': [ScreenshotConfig('', use_basic_auth=True, extra_params={'ticket_title': 'Hardware Ecosystem Compatibility Inquiry', 'ticket_id': '4837', 'message': 'Hi, I am planning to purchase the X5000 smartphone and want to ensure compatibility with my existing devices - WDX10 wireless earbuds and Z600 smartwatch. Are there any known issues?'})]}

def get_all_event_types_for_integration(integration: Integration) -> Optional[List[str]]:
    integration = INTEGRATIONS[integration.name]
    if isinstance(integration, WebhookIntegration):
        if integration.name == 'githubsponsors':
            return import_string('zerver.webhooks.github.view.SPONSORS_EVENT_TYPES')
        function = integration.get_function()
        if hasattr(function, '_all_event_types'):
            return function._all_event_types
    return None
