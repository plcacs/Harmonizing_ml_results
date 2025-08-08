import io
import re
import socketserver
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Tuple

BYTE_RANGE_RE = re.compile(r'bytes=(\d+)-(\d+)?$')


def func_w52zp3cg(
    infile: io.BufferedIOBase,
    outfile: io.BufferedIOBase,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    bufsize: int = 16 * 1024
) -> None:
    """Like shutil.copyfileobj, but only copy a range of the streams.

    Both start and stop are inclusive.
    """
    if start is not None:
        infile.seek(start)
    while True:
        to_read = min(bufsize, stop + 1 - infile.tell() if stop else bufsize)
        buf = infile.read(to_read)
        if not buf:
            break
        outfile.write(buf)


def func_qvtmklbu(byte_range: str) -> Tuple[Optional[int], Optional[int]]:
    """Returns the two numbers in 'bytes=123-456' or throws ValueError.

    The last number or both numbers may be None.
    """
    if byte_range.strip() == '':
        return None, None
    match = BYTE_RANGE_RE.match(byte_range)
    if not match:
        raise ValueError(f'Invalid byte range {byte_range}')
    first, last = [(int(x) if x else None) for x in match.groups()]
    assert first is not None
    if last is not None and last < first:
        raise ValueError(f'Invalid byte range {byte_range}')
    return first, last


def func_518dn6g6(
    filename: str,
    address: str = '',
    port: int = 45114,
    content_type: Optional[str] = None,
    single_req: bool = False
) -> None:

    class FileHandler(BaseHTTPRequestHandler):

        def func_4gr25wzm(self, size: int) -> Tuple[float, str]:
            for size_unity in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024:
                    return size, size_unity
                size = size / 1024
            return size * 1024, size_unity

        def func_tm0x6qyg(
            self,
            format: str,
            *args,
            **kwargs
        ) -> None:
            size, size_unity = self.func_4gr25wzm(stats.st_size)
            format += f' {content_type} - {size:.2f} {size_unity}'
            super(FileHandler, self).log_message(format, *args, **kwargs)

        def func_404d73hl(self) -> None:
            if 'Range' not in self.headers:
                first, last = 0, stats.st_size
            else:
                try:
                    first, last = func_qvtmklbu(self.headers['Range'])
                except ValueError:
                    self.send_error(400, 'Invalid byte range')
                    return
            if last is None or last >= stats.st_size:
                last = stats.st_size - 1
            response_length = last - first + 1
            try:
                if 'Range' not in self.headers:
                    self.send_response(200)
                else:
                    self.send_response(206)
                    self.send_header(
                        'Content-Range',
                        f'bytes {first}-{last}/{stats.st_size}'
                    )
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', str(response_length))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header(
                    'Last-Modified',
                    time.strftime(
                        '%a %d %b %Y %H:%M:%S GMT',
                        time.localtime(stats.st_mtime)
                    )
                )
                self.end_headers()
                mediafile = open(str(mediapath), 'rb')
                func_w52zp3cg(mediafile, self.wfile, first, last)
            except ConnectionResetError:
                pass
            except BrokenPipeError:
                print(
                    'Device disconnected while playing. Please check that the video file is compatible with the device.',
                    file=sys.stderr
                )
            except:
                traceback.print_exc()
            finally:
                mediafile.close()

    if content_type is None:
        content_type = 'video/mp4'
    mediapath = Path(filename)
    stats = mediapath.stat()
    httpd = socketserver.TCPServer((address, port), FileHandler)
    try:
        if single_req:
            httpd.handle_request()
        else:
            httpd.serve_forever()
    finally:
        httpd.server_close()
