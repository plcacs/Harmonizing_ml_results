import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Callable
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
    app: Optional[Flask]

    def __init__(self, app: Optional[Flask] = None) -> None:
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        self.audit_url: Optional[str] = app.config.get('AUDIT_URL')
        if 'admin' in app.config.get('AUDIT_TRAIL', []):
            if app.config.get('AUDIT_LOG', False):
                admin_audit_trail.connect(self.admin_log_response, sender=app)
            if self.audit_url:
                admin_audit_trail.connect(self.admin_webhook_response, sender=app)
        if 'write' in app.config.get('AUDIT_TRAIL', []):
            if app.config.get('AUDIT_LOG', False):
                write_audit_trail.connect(self.write_log_response, sender=app)
            if self.audit_url:
                write_audit_trail.connect(self.write_webhook_response, sender=app)
        if 'auth' in app.config.get('AUDIT_TRAIL', []):
            if app.config.get('AUDIT_LOG', False):
                auth_audit_trail.connect(self.auth_log_response, sender=app)
            if self.audit_url:
                auth_audit_trail.connect(self.auth_webhook_response, sender=app)

    def _log_response(
        self,
        app: Flask,
        category: str,
        event: str,
        message: str,
        user: str,
        customers: Any,
        scopes: Any,
        resource_id: str,
        type: str,
        request: Request,
        **extra: Any
    ) -> None:
        log_message = self._fmt(app, category, event, message, user, customers, scopes, resource_id, type, request, **extra)
        app.logger.info(log_message)

    def _webhook_response(
        self,
        app: Flask,
        category: str,
        event: str,
        message: str,
        user: str,
        customers: Any,
        scopes: Any,
        resource_id: str,
        type: str,
        request: Request,
        **extra: Any
    ) -> None:
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
    def _fmt(
        app: Flask,
        category: str,
        event: str,
        message: str,
        user: str,
        customers: Any,
        scopes: Any,
        resource_id: str,
        type: str,
        request: Request,
        **extra: Any
    ) -> str:
        def get_redacted_data(r: Request) -> Any:
            data = r.get_json(silent=True)
            if data and app.config.get('AUDIT_LOG_REDACT', False):
                if 'password' in data:
                    data['password'] = '[REDACTED]'
            if app.config.get('AUDIT_LOG_JSON', False):
                return data
            return json.dumps(data)

        log_entry: Dict[str, Any] = {
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
                'id': getattr(g, 'request_id', None),
                'endpoint': request.endpoint,
                'method': request.method,
                'url': request.url,
                'args': request.args.to_dict(),
                'data': get_redacted_data(request),
                'remoteIp': request.remote_addr,
                'userAgent': request.headers.get('User-Agent')
            },
            'extra': extra
        }
        return json.dumps(log_entry, cls=CustomJSONEncoder)
