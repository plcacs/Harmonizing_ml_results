import logging
import os
from urllib.parse import quote
from typing import Optional, Tuple, Any, Dict
import requests
from alerta.plugins import PluginBase, app
from alerta.models.alert import Alert  # Assuming Alert is imported from alerta.models.alert

LOG: logging.Logger = logging.getLogger('alerta.plugins.gitlab')
GITLAB_URL: str = 'https://gitlab.com/api/v4'
GITLAB_PROJECT_ID: Optional[str] = os.environ.get('GITLAB_PROJECT_ID') or app.config['GITLAB_PROJECT_ID']
GITLAB_ACCESS_TOKEN: str = os.environ.get('GITLAB_PERSONAL_ACCESS_TOKEN') or app.config['GITLAB_PERSONAL_ACCESS_TOKEN']

class GitlabIssue(PluginBase):

    def __init__(self, name: Optional[str] = None) -> None:
        self.headers: Dict[str, str] = {'Private-Token': GITLAB_ACCESS_TOKEN}
        super().__init__()

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        return (alert, status, text)

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        """should return internal id of external system"""
        BASE_URL: str = f"{GITLAB_URL}/projects/{quote(GITLAB_PROJECT_ID, safe='')}"
        if action == 'createIssue':
            if 'issue_iid' not in alert.attributes:
                url: str = f"{BASE_URL}/issues?title={alert.text}"
                r: requests.Response = requests.post(url, headers=self.headers)
                alert.attributes['issue_iid'] = r.json().get('iid')
                alert.attributes['gitlabUrl'] = f'<a href="{r.json().get("web_url")}" target="_blank">Issue #{r.json().get("iid")}</a>'
        elif action == 'updateIssue':
            if 'issue_iid' in alert.attributes:
                issue_iid: Any = alert.attributes['issue_iid']
                body: str = f'Update: {alert.text}'
                url: str = f"{BASE_URL}/issues/{issue_iid}/discussions?body={body}"
                r: requests.Response = requests.post(url, headers=self.headers)
        return (alert, action, text)
