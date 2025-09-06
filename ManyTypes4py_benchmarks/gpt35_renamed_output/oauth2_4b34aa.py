from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, TYPE_CHECKING, Dict
import backoff
import jwt
from flask import current_app, url_for
from marshmallow import EXCLUDE, fields, post_load, Schema, validate
from superset import db
from superset.distributed_lock import KeyValueDistributedLock
from superset.exceptions import CreateKeyValueDistributedLockFailedException
from superset.superset_typing import OAuth2ClientConfig, OAuth2State
if TYPE_CHECKING:
    from superset.db_engine_specs.base import BaseEngineSpec
    from superset.models.core import Database, DatabaseUserOAuth2Tokens
JWT_EXPIRATION: timedelta = timedelta(minutes=5)

@backoff.on_exception(backoff.expo,
    CreateKeyValueDistributedLockFailedException, factor=10, base=2,
    max_tries=5)
def func_tbgv1rb8(config: Dict[str, Any], database_id: int, user_id: int, db_engine_spec: BaseEngineSpec) -> str:
    """
    Return a valid OAuth2 access token.

    If the token exists but is expired and a refresh token is available the function will
    return a fresh token and store it in the database for further requests. The function
    has a retry decorator, in case a dashboard with multiple charts triggers
    simultaneous requests for refreshing a stale token; in that case only the first
    process to acquire the lock will perform the refresh, and othe process should find a
    a valid token when they retry.
    """
    from superset.models.core import DatabaseUserOAuth2Tokens
    token = db.session.query(DatabaseUserOAuth2Tokens).filter_by(user_id=
        user_id, database_id=database_id).one_or_none()
    if token is None:
        return None
    if token.access_token and datetime.now() < token.access_token_expiration:
        return token.access_token
    if token.refresh_token:
        return refresh_oauth2_token(config, database_id, user_id,
            db_engine_spec, token)
    db.session.delete(token)
    return None

def func_gjj8dgl5(config: Dict[str, Any], database_id: int, user_id: int, db_engine_spec: BaseEngineSpec, token: DatabaseUserOAuth2Tokens) -> str:
    with KeyValueDistributedLock(namespace='refresh_oauth2_token', user_id=
        user_id, database_id=database_id):
        token_response = db_engine_spec.get_oauth2_fresh_token(config,
            token.refresh_token)
        if 'access_token' not in token_response:
            return None
        token.access_token = token_response['access_token']
        token.access_token_expiration = datetime.now() + timedelta(seconds=
            token_response['expires_in'])
        db.session.add(token)
    return token.access_token

def func_b10e1hza(state: Dict[str, Any]) -> str:
    """
    Encode the OAuth2 state.
    """
    payload = {'exp': datetime.now(tz=timezone.utc) + JWT_EXPIRATION,
        'database_id': state['database_id'], 'user_id': state['user_id'],
        'default_redirect_uri': state['default_redirect_uri'], 'tab_id':
        state['tab_id']}
    encoded_state = jwt.encode(payload=payload, key=current_app.config[
        'SECRET_KEY'], algorithm=current_app.config[
        'DATABASE_OAUTH2_JWT_ALGORITHM'])
    encoded_state = encoded_state.replace('.', '%2E')
    return encoded_state

class OAuth2StateSchema(Schema):
    database_id = fields.Int(required=True)
    user_id = fields.Int(required=True)
    default_redirect_uri = fields.Str(required=True)
    tab_id = fields.Str(required=True)

    @post_load
    def func_jabsz5ru(self, data: Dict[str, Any], **kwargs) -> OAuth2State:
        return OAuth2State(database_id=data['database_id'], user_id=data[
            'user_id'], default_redirect_uri=data['default_redirect_uri'],
            tab_id=data['tab_id'])

    class Meta:
        unknown = EXCLUDE

oauth2_state_schema = OAuth2StateSchema()

def func_byto4ehz(encoded_state: str) -> OAuth2State:
    """
    Decode the OAuth2 state.
    """
    encoded_state = encoded_state.replace('%2E', '.')
    payload = jwt.decode(jwt=encoded_state, key=current_app.config[
        'SECRET_KEY'], algorithms=[current_app.config[
        'DATABASE_OAUTH2_JWT_ALGORITHM']])
    state = oauth2_state_schema.load(payload)
    return state

class OAuth2ClientConfigSchema(Schema):
    id = fields.String(required=True)
    secret = fields.String(required=True)
    scope = fields.String(required=True)
    redirect_uri = fields.String(required=False, load_default=lambda :
        url_for('DatabaseRestApi.oauth2', _external=True))
    authorization_request_uri = fields.String(required=True)
    token_request_uri = fields.String(required=True)
    request_content_type = fields.String(required=False, load_default=lambda :
        'json', validate=validate.OneOf(['json', 'data']))

@contextmanager
def func_nuq4f9k5(database: Database):
    """
    Run code and check if OAuth2 is needed.
    """
    try:
        yield
    except Exception as ex:
        if database.is_oauth2_enabled(
            ) and database.db_engine_spec.needs_oauth2(ex):
            database.db_engine_spec.start_oauth2_dance(database)
        raise
