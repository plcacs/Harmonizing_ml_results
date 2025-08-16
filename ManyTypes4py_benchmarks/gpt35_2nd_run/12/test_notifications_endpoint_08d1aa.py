from raiden.api.objects import Notification
from raiden.tests.utils.detect_failure import raise_on_failure
from flask.testing import FlaskClient
from raiden.api.rest import RestAPI
from raiden.api.rest import APIServer

def test_get_empty_notifications(client: FlaskClient, api_server_test_instance: APIServer) -> None:
def test_get_notification(client: FlaskClient, api_server_test_instance: APIServer) -> None:
def test_get_many_notifications(client: FlaskClient, api_server_test_instance: APIServer) -> None:
def test_get_notifications_empty_queue(client: FlaskClient, api_server_test_instance: APIServer) -> None:
def test_notifications_with_same_id_are_overwritten(client: FlaskClient, api_server_test_instance: APIServer) -> None:

def create_notification(api_server: RestAPI, notification_id: str = None, summary: str = None, body: str = None, urgency: str = None) -> Notification:
