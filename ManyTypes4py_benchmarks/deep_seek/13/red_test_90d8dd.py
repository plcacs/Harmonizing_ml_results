import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from redbot.resource import HttpResource
import redbot.speak as rs
import thor
import threading
from tornado import gen
from tornado.options import parse_command_line
from tornado.testing import AsyncHTTPTestCase
from tornado.web import RequestHandler, Application, asynchronous
import unittest

class HelloHandler(RequestHandler):

    def get(self) -> None:
        self.write('Hello world')

class RedirectHandler(RequestHandler):

    def get(self, path: str) -> None:
        self.redirect(path, status=int(self.get_argument('status', '302')))

class PostHandler(RequestHandler):

    def post(self) -> None:
        assert self.get_argument('foo') == 'bar'
        self.redirect('/hello', status=303)

class ChunkedHandler(RequestHandler):

    @asynchronous
    @gen.engine
    def get(self) -> None:
        self.write('hello ')
        yield gen.Task(self.flush)
        self.write('world')
        yield gen.Task(self.flush)
        self.finish()

class CacheHandler(RequestHandler):

    def get(self, computed_etag: str) -> None:
        self.write(computed_etag)

    def compute_etag(self) -> str:
        return self._write_buffer[0]

class TestMixin(object):

    def get_handlers(self) -> List[Tuple[str, Type[RequestHandler]]]:
        return [('/hello', HelloHandler), ('/redirect(/.*)', RedirectHandler), ('/post', PostHandler), ('/chunked', ChunkedHandler), ('/cache/(.*)', CacheHandler)]

    def get_app_kwargs(self) -> Dict[str, Any]:
        return dict(static_path='.')

    def get_allowed_warnings(self) -> List[Any]:
        return [rs.FRESHNESS_HEURISTIC, rs.CONNEG_GZIP_BAD]

    def get_allowed_errors(self) -> List[Any]:
        return []

    def check_url(self, path: str, method: str = 'GET', body: Optional[str] = None, headers: Optional[List[Tuple[str, str]]] = None, expected_status: int = 200, allowed_warnings: Optional[List[Any]] = None, allowed_errors: Optional[List[Any]] = None) -> None:
        url = self.get_url(path)
        red = self.run_redbot(url, method, body, headers)
        if not red.response.complete:
            if isinstance(red.response.http_error, Exception):
                logging.warning((red.response.http_error.desc, vars(red.response.http_error), url))
                raise red.response.http_error.res_error
            else:
                raise Exception('unknown error; incomplete response')
        self.assertEqual(int(red.response.status_code), expected_status)
        allowed_warnings = (allowed_warnings or []) + self.get_allowed_warnings()
        allowed_errors = (allowed_errors or []) + self.get_allowed_errors()
        errors = []
        warnings = []
        for msg in red.response.notes:
            if msg.level == 'bad':
                logger = logging.error
                if not isinstance(msg, tuple(allowed_errors)):
                    errors.append(msg)
            elif msg.level == 'warning':
                logger = logging.warning
                if not isinstance(msg, tuple(allowed_warnings)):
                    warnings.append(msg)
            elif msg.level in ('good', 'info', 'uri'):
                logger = logging.info
            else:
                raise Exception('unknown level' + msg.level)
            logger('%s: %s (%s)', msg.category, msg.show_summary('en'), msg.__class__.__name__)
            logger(msg.show_text('en'))
        self.assertEqual(len(warnings) + len(errors), 0, 'Had %d unexpected warnings and %d errors' % (len(warnings), len(errors)))

    def run_redbot(self, url: str, method: str, body: Optional[str], headers: Optional[List[Tuple[str, str]]]) -> HttpResource:
        red = HttpResource(url, method=method, req_body=body, req_hdrs=headers)

        def work() -> None:
            red.run(thor.stop)
            thor.run()
            self.io_loop.add_callback(self.stop)
        thread = threading.Thread(target=work)
        thread.start()
        self.wait()
        thread.join()
        return red

    def test_hello(self) -> None:
        self.check_url('/hello')

    def test_static(self) -> None:
        self.check_url('/static/red_test.py', allowed_warnings=[rs.MISSING_HDRS_304])

    def test_static_versioned_url(self) -> None:
        self.check_url('/static/red_test.py?v=1234', allowed_warnings=[rs.MISSING_HDRS_304])

    def test_redirect(self) -> None:
        self.check_url('/redirect/hello', expected_status=302)

    def test_permanent_redirect(self) -> None:
        self.check_url('/redirect/hello?status=301', expected_status=301)

    def test_404(self) -> None:
        self.check_url('/404', expected_status=404)

    def test_post(self) -> None:
        body = 'foo=bar'
        self.check_url('/post', method='POST', body=body, headers=[('Content-Length', str(len(body))), ('Content-Type', 'application/x-www-form-urlencoded')], expected_status=303)

    def test_chunked(self) -> None:
        self.check_url('/chunked')

    def test_strong_etag_match(self) -> None:
        computed_etag = '"xyzzy"'
        etags = '"xyzzy"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=304)

    def test_multiple_strong_etag_match(self) -> None:
        computed_etag = '"xyzzy1"'
        etags = '"xyzzy1", "xyzzy2"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=304)

    def test_strong_etag_not_match(self) -> None:
        computed_etag = '"xyzzy"'
        etags = '"xyzzy1"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=200)

    def test_multiple_strong_etag_not_match(self) -> None:
        computed_etag = '"xyzzy"'
        etags = '"xyzzy1", "xyzzy2"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=200)

    def test_wildcard_etag(self) -> None:
        computed_etag = '"xyzzy"'
        etags = '*'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=304, allowed_warnings=[rs.MISSING_HDRS_304])

    def test_weak_etag_match(self) -> None:
        computed_etag = '"xyzzy1"'
        etags = 'W/"xyzzy1"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=304)

    def test_multiple_weak_etag_match(self) -> None:
        computed_etag = '"xyzzy2"'
        etags = 'W/"xyzzy1", W/"xyzzy2"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=304)

    def test_weak_etag_not_match(self) -> None:
        computed_etag = '"xyzzy2"'
        etags = 'W/"xyzzy1"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=200)

    def test_multiple_weak_etag_not_match(self) -> None:
        computed_etag = '"xyzzy3"'
        etags = 'W/"xyzzy1", W/"xyzzy2"'
        self.check_url('/cache/' + computed_etag, method='GET', headers=[('If-None-Match', etags)], expected_status=200)

class DefaultHTTPTest(AsyncHTTPTestCase, TestMixin):

    def get_app(self) -> Application:
        return Application(self.get_handlers(), **self.get_app_kwargs())

class GzipHTTPTest(AsyncHTTPTestCase, TestMixin):

    def get_app(self) -> Application:
        return Application(self.get_handlers(), gzip=True, **self.get_app_kwargs())

    def get_allowed_errors(self) -> List[Any]:
        return super().get_allowed_errors() + [rs.VARY_ETAG_DOESNT_CHANGE]

if __name__ == '__main__':
    parse_command_line()
    unittest.main()
