import io
import re
import socketserver
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

BYTE_RANGE_RE = re.compile('bytes=(\\d+)-(\\d+)?$')


def copy_byte_range(infile: BinaryIO, outfile: BinaryIO, start: Optional[int] = None, stop: Optional[int] = None, bufsize: int = 16 * 1024) -> None:
    """Like shutil.copyfileobj, but only copy a range of the streams.

    Both start and stop are inclusive.
    """
    if start is not None:
        infile.seek(start)
    while True:
        to_read = min(bufsize, (stop + 1 - infile.tell()) if stop is not None else bufsize)
        buf = infile.read(to_read)
        if not buf:
            break
        outfile.write(buf)


def parse_byte_range(byte_range: str) -> Tuple[Optional[int], Optional[int]]:
    """Returns the two numbers in 'bytes=123-456' or throws ValueError.

    The last number or both numbers may be None.
    """
    if byte_range.strip() == '':
        return (None, None)
    match = BYTE_RANGE_RE.match(byte_range)
    if not match:
        raise ValueError('Invalid byte range {}'.format(byte_range))
    first, last = [int(x) if x else None for x in match.groups()]
    assert first is not None
    if last is not None and last < first:
        raise ValueError('Invalid byte range {}'.format(byte_range))
    return (first, last)


def serve_file(filename: str, address: str = '', port: int = 45114, content_type: Optional[str] = None, single_req: bool = False) -> None:
    if content_type is None:
        content_type = 'video/mp4'
    mediapath = Path(filename)
    stats = mediapath.stat()

    class FileHandler(BaseHTTPRequestHandler):
        def format_size(self, size: int) -> Tuple[float, str]:
            for size_unity in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024:
                    return (size, size_unity)
                size = size / 1024
            return (size * 1024, size_unity)

        def log_message(self, format: str, *args: object, **kwargs: object) -> None:
            size, size_unity = self.format_size(stats.st_size)
            format += ' {} - {:0.2f} {}'.format(content_type, size, size_unity)
            super().log_message(format, *args, **kwargs)

        def do_GET(self) -> None:
            if 'Range' not in self.headers:
                first, last = (0, stats.st_size)
            else:
                try:
                    first, last = parse_byte_range(self.headers['Range'])
                except ValueError:
                    self.send_error(400, 'Invalid byte range')
                    return
            if last is None or last >= stats.st_size:
                last = stats.st_size - 1
            response_length: int = last - first + 1
            try:
                if 'Range' not in self.headers:
                    self.send_response(200)
                else:
                    self.send_response(206)
                    self.send_header('Content-Range', 'bytes {}-{}/{}'.format(first, last, stats.st_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', str(response_length))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Last-Modified', time.strftime('%a %d %b %Y %H:%M:%S GMT', time.localtime(stats.st_mtime)))
                self.end_headers()
                mediafile: BinaryIO = open(str(mediapath), 'rb')
                copy_byte_range(mediafile, self.wfile, first, last)
            except ConnectionResetError:
                pass
            except BrokenPipeError:
                print('Device disconnected while playing. Please check that the video file is compatible with the device.', file=sys.stderr)
            except Exception:
                traceback.print_exc()
            finally:
                mediafile.close()

    httpd = socketserver.TCPServer((address, port), FileHandler)
    if single_req:
        httpd.handle_request()
    else:
        httpd.serve_forever()
    httpd.server_close()