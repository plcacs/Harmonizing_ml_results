import logging
import os
from urllib.parse import quote
import requests
from alerta.plugins import PluginBase, app
LOG = logging.getLogger('alerta.plugins.gitlab')
GITLAB_URL = 'https://gitlab.com/api/v4'
GITLAB_PROJECT_ID = os.environ.get('GITLAB_PROJECT_ID', None) or app.config['GITLAB_PROJECT_ID']
GITLAB_ACCESS_TOKEN = os.environ.get('GITLAB_PERSONAL_ACCESS_TOKEN') or app.config['GITLAB_PERSONAL_ACCESS_TOKEN']

class GitlabIssue(PluginBase):

    def __init__(self, name: Union[None, str, typing.Iterable[str]]=None) -> None:
        self.headers = {'Private-Token': GITLAB_ACCESS_TOKEN}
        super().__init__()

    def pre_receive(self, alert: list[tuple[str]], **kwargs) -> list[tuple[str]]:
        return alert

    def post_receive(self, alert: list[tuple[str]], **kwargs) -> list[tuple[str]]:
        return alert

    def status_change(self, alert: str, status: str, text: str, **kwargs) -> tuple[str]:
        return (alert, status, text)

    def take_action(self, alert: Union[str, dict, dict[str, typing.Any]], action: str, text: Union[str, set, bytes], **kwargs) -> tuple[typing.Union[str,dict,dict[str, typing.Any],set,bytes]]:
        """should return internal id of external system"""
        BASE_URL = '{}/projects/{}'.format(GITLAB_URL, quote(GITLAB_PROJECT_ID, safe=''))
        if action == 'createIssue':
            if 'issue_iid' not in alert.attributes:
                url = BASE_URL + '/issues?title=' + alert.text
                r = requests.post(url, headers=self.headers)
                alert.attributes['issue_iid'] = r.json().get('iid', None)
                alert.attributes['gitlabUrl'] = '<a href="{}" target="_blank">Issue #{}</a>'.format(r.json().get('web_url', None), r.json().get('iid', None))
        elif action == 'updateIssue':
            if 'issue_iid' in alert.attributes:
                issue_iid = alert.attributes['issue_iid']
                body = 'Update: ' + alert.text
                url = BASE_URL + '/issues/{}/discussions?body={}'.format(issue_iid, body)
                r = requests.post(url, headers=self.headers)
        return (alert, action, text)