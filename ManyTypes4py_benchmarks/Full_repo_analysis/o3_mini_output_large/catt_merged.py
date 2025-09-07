import unittest
from typing import Any, Callable
from yt_dlp.utils import DownloadError
from catt.stream_info import StreamInfo

def ignore_tmr_failure(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Ignore "Too many requests" failures in a test.

    YouTube will sometimes throttle us and cause the tests to flap. This decorator
    catches the "Too many requests" exceptions in tests and ignores them.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except DownloadError as err:
            if 'HTTP Error 429:' in str(err):
                pass
            else:
                raise
    return wrapper

class TestThings(unittest.TestCase):

    @ignore_tmr_failure
    def test_stream_info_youtube_video(self) -> None:
        stream = StreamInfo('https://www.youtube.com/watch?v=VZMfhtKa-wo', throw_ytdl_dl_errs=True)
        self.assertIn('https://', stream.video_url)
        self.assertEqual(stream.video_id, 'VZMfhtKa-wo')
        self.assertTrue(stream.is_remote_file)
        self.assertEqual(stream.extractor, 'youtube')

    @ignore_tmr_failure
    def test_stream_info_youtube_playlist(self) -> None:
        stream = StreamInfo('https://www.youtube.com/playlist?list=PL9Z0stL3aRykWNoVQW96JFIkelka_93Sc', throw_ytdl_dl_errs=True)
        self.assertIsNone(stream.video_url)
        self.assertEqual(stream.playlist_id, 'PL9Z0stL3aRykWNoVQW96JFIkelka_93Sc')
        self.assertTrue(stream.is_playlist)
        self.assertEqual(stream.extractor, 'youtube')

    def test_stream_info_other_video(self) -> None:
        stream = StreamInfo('https://www.twitch.tv/twitch/clip/MistySoftPenguinKappaPride')
        self.assertIn('https://', stream.video_url)
        self.assertEqual(stream.video_id, '492743767')
        self.assertTrue(stream.is_remote_file)
        self.assertEqual(stream.extractor, 'twitch')

if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())