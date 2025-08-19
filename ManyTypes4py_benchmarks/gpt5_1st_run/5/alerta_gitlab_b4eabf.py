import logging
import os
from urllib.parse import quote
from typing import Any, Dict, MutableMapping, Tuple, Optional, Protocol, cast
import requests
from alerta.plugins import PluginBase, app

LOG: logging.Logger = logging.getLogger('alerta.plugins.gitlab')
GITLAB_URL: str = 'https://gitlab.com/api/v4'
GITLAB_PROJECT_ID: str = cast(str, os.environ.get('GITLAB_PROJECT_ID', None) or app.config['GITLAB_PROJECT_ID'])
GITLAB_ACCESS_TOKEN: str = cast(str, os.environ.get('GITLAB_PERSONAL_ACCESS_TOKEN') or app.config['GITLAB_PERSONAL_ACCESS_TOKEN'])

class AlertLike(Protocol):
    text: str
    attributes: MutableMapping[str, Any]

class GitlabIssue(PluginBase):

    def __init__(self, name: Optional[str] = None) -> None:
        self.headers: Dict[str, str] = {'Private-Token': GITLAB_ACCESS_TOKEN}
        super().__init__()

    def pre_receive(self, alert: AlertLike, **kwargs: Any) -> AlertLike:
        return alert

    def post_receive(self, alert: AlertLike, **kwargs: Any) -> AlertLike:
        return alert

    def status_change(self, alert: AlertLike, status: str, text: str, **kwargs: Any) -> Tuple[AlertLike, str, str]:
        return (alert, status, text)

    def take_action(self, alert: AlertLike, action: str, text: str, **kwargs: Any) -> Tuple[AlertLike, str, str]:
        """should return internal id of external system"""
        BASE_URL: str = '{}/projects/{}'.format(GITLAB_URL, quote(GITLAB_PROJECT_ID, safe=''))
        if action == 'createIssue':
            if 'issue_iid' not in alert.attributes:
                url: str = BASE_URL + '/issues?title=' + alert.text
                r: requests.Response = requests.post(url, headers=self.headers)
                alert.attributes['issue_iid'] = r.json().get('iid', None)
                alert.attributes['gitlabUrl'] = '<a href="{}" target="_blank">Issue #{}</a>'.format(r.json().get('web_url', None), r.json().get('iid', None))
        elif action == 'updateIssue':
            if 'issue_iid' in alert.attributes:
                issue_iid: Any = alert.attributes['issue_iid']
                body: str = 'Update: ' + alert.text
                url: str = BASE_URL + '/issues/{}/discussions?body={}'.format(issue_iid, body)
                r: requests.Response = requests.post(url, headers=self.headers)
        return (alert, action, text)