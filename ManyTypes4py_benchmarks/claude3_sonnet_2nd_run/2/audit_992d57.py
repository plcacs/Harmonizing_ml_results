import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import blinker
import requests
from flask import Flask, g, Request
from alerta.utils.format import CustomJSONEncoder

audit_signals = blinker.Namespace()
admin_audit_trail = audit_signals.signal('admin')
write_audit_trail = audit_signals.signal('write')
read_audit_trail = audit_signals.signal('read')
auth_audit_trail = audit_signals.signal('auth')

class AuditTrail:

    def __init__(self, app: Optional[Flask] = None) -> None:
        self.app: Optional[Flask] = app
        self.audit_url: Optional[str] = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        self.audit_url = app.config['AUDIT_URL']
        if 'admin' in app.config['AUDIT_TRAIL']:
            if app.config['AUDIT_LOG']:
                admin_audit_trail.connect(self.admin_log_response, app)
            if self.audit_url:
                admin_audit_trail.connect(self.admin_webhook_response, app)
        if 'write' in app.config['AUDIT_TRAIL']:
            if app.config['AUDIT_LOG']:
                write_audit_trail.connect(self.write_log_response, app)
            if self.audit_url:
                write_audit_trail.connect(self.write_webhook_response, app)
        if 'auth' in app.config['AUDIT_TRAIL']:
            if app.config['AUDIT_LOG']:
                auth_audit_trail.connect(self.auth_log_response, app)
            if self.audit_url:
                auth_audit_trail.connect(self.auth_webhook_response, app)

    def _log_response(self, app: Flask, category: str, event: str, message: str, user: str, 
                     customers: List[str], scopes: List[str], resource_id: str, type: str, 
                     request: Request, **extra: Any) -> None:
        app.logger.info(self._fmt(app, category, event, message, user, customers, scopes, resource_id, type, request, **extra))

    def _webhook_response(self, app: Flask, category: str, event: str, message: str, user: str, 
                         customers: List[str], scopes: List[str], resource_id: str, type: str, 
                         request: Request, **extra: Any) -> None:
        payload = self._fmt(app, category, event, message, user, customers, scopes, resource_id, type, request, **extra)
        try:
            requests.post(self.audit_url, data=payload, timeout=2)
        except Exception as e:
            app.logger.warning(f'Failed to send audit log entry to "{self.audit_url}" - {str(e)}')

    def admin_log_response(self, app: Flask, **kwargs: Any) -> None:
        self._log_response(app, 'admin', **kwargs)

    def admin_webhook_response(self, app: Flask, **kwargs: Any) -> None:
        self._webhook_response(app, 'admin', **kwargs)

    def write_log_response(self, app: Flask, **kwargs: Any) -> None:
        self._log_response(app, 'write', **kwargs)

    def write_webhook_response(self, app: Flask, **kwargs: Any) -> None:
        self._webhook_response(app, 'write', **kwargs)

    def auth_log_response(self, app: Flask, **kwargs: Any) -> None:
        self._log_response(app, 'auth', **kwargs)

    def auth_webhook_response(self, app: Flask, **kwargs: Any) -> None:
        self._webhook_response(app, 'auth', **kwargs)

    @staticmethod
    def _fmt(app: Flask, category: str, event: str, message: str, user: str, 
            customers: List[str], scopes: List[str], resource_id: str, type: str, 
            request: Request, **extra: Any) -> str:

        def get_redacted_data(r: Request) -> Union[Dict[str, Any], str, None]:
            data = r.get_json(silent=True)
            if data and app.config['AUDIT_LOG_REDACT']:
                if 'password' in data:
                    data['password'] = '[REDACTED]'
            if app.config['AUDIT_LOG_JSON']:
                return data
            return json.dumps(data)
        
        return json.dumps({
            'id': str(uuid.uuid4()),
            '@timestamp': datetime.utcnow(),
            'event': event,
            'category': category,
            'message': message,
            'user': {
                'id': user,
                'customers': customers,
                'scopes': scopes
            },
            'resource': {
                'id': resource_id,
                'type': type
            },
            'request': {
                'id': g.request_id if hasattr(g, 'request_id') else None,
                'endpoint': request.endpoint,
                'method': request.method,
                'url': request.url,
                'args': request.args.to_dict(),
                'data': get_redacted_data(request),
                'remoteIp': request.remote_addr,
                'userAgent': request.headers.get('User-Agent')
            },
            'extra': extra
        }, cls=CustomJSONEncoder)
