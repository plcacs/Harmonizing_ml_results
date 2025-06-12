from __future__ import annotations
import html
import inspect
import sys
import traceback
import typing
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
STYLES = '\np {\n    color: #211c1c;\n}\n.traceback-container {\n    border: 1px solid #038BB8;\n}\n.traceback-title {\n    background-color: #038BB8;\n    color: lemonchiffon;\n    padding: 12px;\n    font-size: 20px;\n    margin-top: 0px;\n}\n.frame-line {\n    padding-left: 10px;\n    font-family: monospace;\n}\n.frame-filename {\n    font-family: monospace;\n}\n.center-line {\n    background-color: #038BB8;\n    color: #f9f6e1;\n    padding: 5px 0px 5px 5px;\n}\n.lineno {\n    margin-right: 5px;\n}\n.frame-title {\n    font-weight: unset;\n    padding: 10px 10px 10px 10px;\n    background-color: #E4F4FD;\n    margin-right: 10px;\n    color: #191f21;\n    font-size: 17px;\n    border: 1px solid #c7dce8;\n}\n.collapse-btn {\n    float: right;\n    padding: 0px 5px 1px 5px;\n    border: solid 1px #96aebb;\n    cursor: pointer;\n}\n.collapsed {\n  display: none;\n}\n.source-code {\n  font-family: courier;\n  font-size: small;\n  padding-bottom: 10px;\n}\n'
JS = '\n<script type="text/javascript">\n    function collapse(element){\n        const frameId = element.getAttribute("data-frame-id");\n        const frame = document.getElementById(frameId);\n\n        if (frame.classList.contains("collapsed")){\n            element.innerHTML = "&#8210;";\n            frame.classList.remove("collapsed");\n        } else {\n            element.innerHTML = "+";\n            frame.classList.add("collapsed");\n        }\n    }\n</script>\n'
TEMPLATE = '\n<html>\n    <head>\n        <style type=\'text/css\'>\n            {styles}\n        </style>\n        <title>Starlette Debugger</title>\n    </head>\n    <body>\n        <h1>500 Server Error</h1>\n        <h2>{error}</h2>\n        <div class="traceback-container">\n            <p class="traceback-title">Traceback</p>\n            <div>{exc_html}</div>\n        </div>\n        {js}\n    </body>\n</html>\n'
FRAME_TEMPLATE = '\n<div>\n    <p class="frame-title">File <span class="frame-filename">{frame_filename}</span>,\n    line <i>{frame_lineno}</i>,\n    in <b>{frame_name}</b>\n    <span class="collapse-btn" data-frame-id="{frame_filename}-{frame_lineno}" onclick="collapse(this)">{collapse_button}</span>\n    </p>\n    <div id="{frame_filename}-{frame_lineno}" class="source-code {collapsed}">{code_context}</div>\n</div>\n'
LINE = '\n<p><span class="frame-line">\n<span class="lineno">{lineno}.</span> {line}</span></p>\n'
CENTER_LINE = '\n<p class="center-line"><span class="frame-line center-line">\n<span class="lineno">{lineno}.</span> {line}</span></p>\n'

class ServerErrorMiddleware:
    """
    Handles returning 500 responses when a server error occurs.

    If 'debug' is set, then traceback responses will be returned,
    otherwise the designated 'handler' will be called.

    This middleware class should generally be used to wrap *everything*
    else up, so that unhandled exceptions anywhere in the stack
    always result in an appropriate 500 response.
    """

    def __init__(self, app, handler=None, debug=False):
        self.app = app
        self.handler = handler
        self.debug = debug

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        response_started = False

        async def _send(message):
            nonlocal response_started, send
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)
        try:
            await self.app(scope, receive, _send)
        except Exception as exc:
            request = Request(scope)
            if self.debug:
                response = self.debug_response(request, exc)
            elif self.handler is None:
                response = self.error_response(request, exc)
            elif is_async_callable(self.handler):
                response = await self.handler(request, exc)
            else:
                response = await run_in_threadpool(self.handler, request, exc)
            if not response_started:
                await response(scope, receive, send)
            raise exc

    def format_line(self, index, line, frame_lineno, frame_index):
        values = {'line': html.escape(line).replace(' ', '&nbsp'), 'lineno': frame_lineno - frame_index + index}
        if index != frame_index:
            return LINE.format(**values)
        return CENTER_LINE.format(**values)

    def generate_frame_html(self, frame, is_collapsed):
        code_context = ''.join((self.format_line(index, line, frame.lineno, frame.index) for index, line in enumerate(frame.code_context or [])))
        values = {'frame_filename': html.escape(frame.filename), 'frame_lineno': frame.lineno, 'frame_name': html.escape(frame.function), 'code_context': code_context, 'collapsed': 'collapsed' if is_collapsed else '', 'collapse_button': '+' if is_collapsed else '&#8210;'}
        return FRAME_TEMPLATE.format(**values)

    def generate_html(self, exc, limit=7):
        traceback_obj = traceback.TracebackException.from_exception(exc, capture_locals=True)
        exc_html = ''
        is_collapsed = False
        exc_traceback = exc.__traceback__
        if exc_traceback is not None:
            frames = inspect.getinnerframes(exc_traceback, limit)
            for frame in reversed(frames):
                exc_html += self.generate_frame_html(frame, is_collapsed)
                is_collapsed = True
        if sys.version_info >= (3, 13):
            exc_type_str = traceback_obj.exc_type_str
        else:
            exc_type_str = traceback_obj.exc_type.__name__
        error = f'{html.escape(exc_type_str)}: {html.escape(str(traceback_obj))}'
        return TEMPLATE.format(styles=STYLES, js=JS, error=error, exc_html=exc_html)

    def generate_plain_text(self, exc):
        return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    def debug_response(self, request, exc):
        accept = request.headers.get('accept', '')
        if 'text/html' in accept:
            content = self.generate_html(exc)
            return HTMLResponse(content, status_code=500)
        content = self.generate_plain_text(exc)
        return PlainTextResponse(content, status_code=500)

    def error_response(self, request, exc):
        return PlainTextResponse('Internal Server Error', status_code=500)