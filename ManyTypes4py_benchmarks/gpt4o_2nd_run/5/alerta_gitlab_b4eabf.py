import logging
import os
from urllib.parse import quote
import requests
from alerta.plugins import PluginBase, app
from typing import Optional, Tuple, Any

LOG = logging.getLogger('alerta.plugins.gitlab')
GITLAB_URL = 'https://gitlab.com/api/v4'
GITLAB_PROJECT_ID: str = os.environ.get('GITLAB_PROJECT_ID', None) or app.config['GITLAB_PROJECT_ID']
GITLAB_ACCESS_TOKEN: str = os.environ.get('GITLAB_PERSONAL_ACCESS_TOKEN') or app.config['GITLAB_PERSONAL_ACCESS_TOKEN']

class GitlabIssue(PluginBase):

    def __init__(self, name: Optional[str] = None) -> None:
        self.headers = {'Private-Token': GITLAB_ACCESS_TOKEN}
        super().__init__()

    def pre_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def post_receive(self, alert: Any, **kwargs: Any) -> Any:
        return alert

    def status_change(self, alert: Any, status: str, text: str, **kwargs: Any) -> Tuple[Any, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Any, action: str, text: str, **kwargs: Any) -> Tuple[Any, str, str]:
        """should return internal id of external system"""
        BASE_URL: str = '{}/projects/{}'.format(GITLAB_URL, quote(GITLAB_PROJECT_ID, safe=''))
        if action == 'createIssue':
            if 'issue_iid' not in alert.attributes:
                url: str = BASE_URL + '/issues?title=' + alert.text
                r = requests.post(url, headers=self.headers)
                alert.attributes['issue_iid'] = r.json().get('iid', None)
                alert.attributes['gitlabUrl'] = '<a href="{}" target="_blank">Issue #{}</a>'.format(r.json().get('web_url', None), r.json().get('iid', None))
        elif action == 'updateIssue':
            if 'issue_iid' in alert.attributes:
                issue_iid: str = alert.attributes['issue_iid']
                body: str = 'Update: ' + alert.text
                url: str = BASE_URL + '/issues/{}/discussions?body={}'.format(issue_iid, body)
                r = requests.post(url, headers=self.headers)
        return (alert, action, text)
