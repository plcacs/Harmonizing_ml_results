"""Download pictures (or videos) along with their captions and other metadata from Instagram."""
import ast
import datetime
import os
import re
import sys
from argparse import ArgumentParser, ArgumentTypeError, SUPPRESS
from enum import IntEnum
from typing import List, Optional, Dict, Set, Any, Callable, Union, Tuple, TypeVar, cast
from . import AbortDownloadException, BadCredentialsException, Instaloader, InstaloaderException, InvalidArgumentException, LoginException, Post, Profile, ProfileNotExistsException, StoryItem, TwoFactorAuthRequiredException, __version__, load_structure_from_file
from .instaloader import get_default_session_filename, get_default_stamps_filename
from .instaloadercontext import default_user_agent
from .lateststamps import LatestStamps

try:
    import browser_cookie3
    bc3_library = True
except ImportError:
    bc3_library = False

class ExitCode(IntEnum):
    SUCCESS = 0
    NON_FATAL_ERROR = 1
    INIT_FAILURE = 2
    LOGIN_FAILURE = 3
    DOWNLOAD_ABORTED = 4
    USER_ABORTED = 5
    UNEXPECTED_ERROR = 99

def usage_string() -> str:
    argv0 = os.path.basename(sys.argv[0])
    argv0 = 'instaloader' if argv0 == '__main__.py' else argv0
    return '\n{0} [--comments] [--geotags]\n{2:{1}} [--stories] [--highlights] [--tagged] [--reels] [--igtv]\n{2:{1}} [--login YOUR-USERNAME] [--fast-update]\n{2:{1}} profile | "#hashtag" | %%location_id | :stories | :feed | :saved\n{0} --help'.format(argv0, len(argv0), '')

def http_status_code_list(code_list_str: str) -> List[int]:
    codes = [int(s) for s in code_list_str.split(',')]
    for code in codes:
        if not 100 <= code <= 599:
            raise ArgumentTypeError('Invalid HTTP status code: {}'.format(code))
    return codes

T = TypeVar('T', Post, StoryItem)
def filterstr_to_filterfunc(filter_str: str, item_type: T) -> Callable[[T], bool]:
    """Takes an --post-filter=... or --storyitem-filter=... filter
     specification and makes a filter_func Callable out of it."""

    class TransformFilterAst(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.Attribute:
            if not isinstance(node.ctx, ast.Load):
                raise InvalidArgumentException('Invalid filter: Modifying variables ({}) not allowed.'.format(node.id))
            if node.id == 'datetime':
                return node
            if not hasattr(item_type, node.id):
                raise InvalidArgumentException('Invalid filter: {} not a {} attribute.'.format(node.id, item_type.__name__))
            new_node = ast.Attribute(ast.copy_location(ast.Name('item', ast.Load()), node), node.id, ast.copy_location(ast.Load(), node))
            return ast.copy_location(new_node, node)
    input_filename = '<command line filter parameter>'
    compiled_filter = compile(TransformFilterAst().visit(ast.parse(filter_str, filename=input_filename, mode='eval')), filename=input_filename, mode='eval')

    def filterfunc(item: T) -> bool:
        return bool(eval(compiled_filter, {'item': item, 'datetime': datetime.datetime}))
    return filterfunc

def get_cookies_from_instagram(domain: str, browser: str, cookie_file: str = '', cookie_name: str = '') -> Union[Dict[str, str], str]:
    supported_browsers = {'brave': browser_cookie3.brave, 'chrome': browser_cookie3.chrome, 'chromium': browser_cookie3.chromium, 'edge': browser_cookie3.edge, 'firefox': browser_cookie3.firefox, 'librewolf': browser_cookie3.librewolf, 'opera': browser_cookie3.opera, 'opera_gx': browser_cookie3.opera_gx, 'safari': browser_cookie3.safari, 'vivaldi': browser_cookie3.vivaldi}
    if browser not in supported_browsers:
        raise InvalidArgumentException('Loading cookies from the specified browser failed\nSupported browsers are Brave, Chrome, Chromium, Edge, Firefox, LibreWolf, Opera, Opera_GX, Safari and Vivaldi')
    cookies: Dict[str, str] = {}
    browser_cookies = list(supported_browsers[browser](cookie_file=cookie_file))
    for cookie in browser_cookies:
        if domain in cookie.domain:
            cookies[cookie.name] = cookie.value
    if cookies:
        print(f'Cookies loaded successfully from {browser}')
    else:
        raise LoginException(f'No cookies found for Instagram in {browser}, Are you logged in successfully in {browser}?')
    if cookie_name:
        return cookies.get(cookie_name, '')
    else:
        return cookies

def import_session(browser: str, instaloader: Instaloader, cookiefile: str) -> None:
    cookie = get_cookies_from_instagram('instagram', browser, cookiefile)
    if cookie is not None:
        instaloader.context.update_cookies(cookie)
        username = instaloader.test_login()
        if not username:
            raise LoginException(f'Not logged in. Are you logged in successfully in {browser}?')
        instaloader.context.username = username
        print(f'{username} has been successfully logged in.')
        print(f'Next time use --login={username} to reuse the same session.')

def _main(instaloader: Instaloader, targetlist: List[str], username: Optional[str] = None, password: Optional[str] = None, sessionfile: Optional[str] = None, download_profile_pic: bool = True, download_posts: bool = True, download_stories: bool = False, download_highlights: bool = False, download_tagged: bool = False, download_reels: bool = False, download_igtv: bool = False, fast_update: bool = False, latest_stamps_file: Optional[str] = None, max_count: Optional[int] = None, post_filter_str: Optional[str] = None, storyitem_filter_str: Optional[str] = None, browser: Optional[str] = None, cookiefile: Optional[str] = None) -> ExitCode:
    """Download set of profiles, hashtags etc. and handle logging in and session files if desired."""
    post_filter: Optional[Callable[[Post], bool]] = None
    if post_filter_str is not None:
        post_filter = filterstr_to_filterfunc(post_filter_str, Post)
        instaloader.context.log('Only download posts with property "{}".'.format(post_filter_str))
    storyitem_filter: Optional[Callable[[StoryItem], bool]] = None
    if storyitem_filter_str is not None:
        storyitem_filter = filterstr_to_filterfunc(storyitem_filter_str, StoryItem)
        instaloader.context.log('Only download storyitems with property "{}".'.format(storyitem_filter_str))
    latest_stamps: Optional[LatestStamps] = None
    if latest_stamps_file is not None:
        latest_stamps = LatestStamps(latest_stamps_file)
        instaloader.context.log(f'Using latest stamps from {latest_stamps_file}.')
    if browser and bc3_library:
        import_session(browser.lower(), instaloader, cookiefile)
    elif browser and (not bc3_library):
        raise InvalidArgumentException('browser_cookie3 library is needed to load cookies from browsers')
    if username is not None:
        if not re.match('^[A-Za-z0-9._]+$', username):
            instaloader.context.error('Warning: Parameter "{}" for --login is not a valid username.'.format(username))
        try:
            instaloader.load_session_from_file(username, sessionfile)
        except FileNotFoundError as err:
            if sessionfile is not None:
                print(err, file=sys.stderr)
            instaloader.context.log('Session file does not exist yet - Logging in.')
        if not instaloader.context.is_logged_in or username != instaloader.test_login():
            if password is not None:
                try:
                    instaloader.login(username, password)
                except TwoFactorAuthRequiredException:
                    instaloader.context.error('Warning: There have been reports of 2FA currently not working. Consider importing session cookies from your browser with --load-cookies.')
                    while True:
                        try:
                            code = input('Enter 2FA verification code: ')
                            instaloader.two_factor_login(code)
                            break
                        except BadCredentialsException as err:
                            print(err, file=sys.stderr)
                            pass
            else:
                try:
                    instaloader.interactive_login(username)
                except KeyboardInterrupt:
                    print('\nInterrupted by user.', file=sys.stderr)
                    return ExitCode.USER_ABORTED
        instaloader.context.log('Logged in as %s.' % username)
    if instaloader.download_geotags and (not instaloader.context.is_logged_in):
        instaloader.context.error('Warning: Login is required to download geotags of posts.')
    profiles: Set[Profile] = set()
    anonymous_retry_profiles: Set[Profile] = set()
    exit_code = ExitCode.SUCCESS
    try:
        for target in targetlist:
            if (target.endswith('.json') or target.endswith('.json.xz')) and os.path.isfile(target):
                with instaloader.context.error_catcher(target):
                    structure = load_structure_from_file(instaloader.context, target)
                    if isinstance(structure, Post):
                        if post_filter is not None and (not post_filter(structure)):
                            instaloader.context.log('<{} ({}) skipped>'.format(structure, target), flush=True)
                            continue
                        instaloader.context.log('Downloading {} ({})'.format(structure, target))
                        instaloader.download_post(structure, os.path.dirname(target))
                    elif isinstance(structure, StoryItem):
                        if storyitem_filter is not None and (not storyitem_filter(structure)):
                            instaloader.context.log('<{} ({}) skipped>'.format(structure, target), flush=True)
                            continue
                        instaloader.context.log('Attempting to download {} ({})'.format(structure, target))
                        instaloader.download_storyitem(structure, os.path.dirname(target))
                    elif isinstance(structure, Profile):
                        raise InvalidArgumentException('Profile JSON are ignored. Pass "{}" to download that profile'.format(structure.username))
                    else:
                        raise InvalidArgumentException('{} JSON file not supported as target'.format(structure.__class__.__name__))
                continue
            target = target.rstrip('/')
            with instaloader.context.error_catcher(target):
                if re.match('^@[A-Za-z0-9._]+$', target):
                    instaloader.context.log('Retrieving followees of %s...' % target[1:])
                    profile = Profile.from_username(instaloader.context, target[1:])
                    for followee in profile.get_followees():
                        instaloader.save_profile_id(followee)
                        profiles.add(followee)
                elif re.match('^#\\w+$', target):
                    instaloader.download_hashtag(hashtag=target[1:], max_count=max_count, fast_update=fast_update, post_filter=post_filter, profile_pic=download_profile_pic, posts=download_posts)
                elif re.match('^-[A-Za-z0-9-_]+$', target):
                    instaloader.download_post(Post.from_shortcode(instaloader.context, target[1:]), target)
                elif re.match('^%[0-9]+$', target):
                    instaloader.download_location(location=target[1:], max_count=max_count, fast_update=fast_update, post_filter=post_filter)
                elif target == ':feed':
                    instaloader.download_feed_posts(fast_update=fast_update, max_count=max_count, post_filter=post_filter)
                elif target == ':stories':
                    instaloader.download_stories(fast_update=fast_update, storyitem_filter=storyitem_filter)
                elif target == ':saved':
                    instaloader.download_saved_posts(fast_update=fast_update, max_count=max_count, post_filter=post_filter)
                elif re.match('^[A-Za-z0-9._]+$', target):
                    download_profile_content = download_posts or download_tagged or download_reels or download_igtv
                    try:
                        profile = instaloader.check_profile_id(target, latest_stamps)
                        if instaloader.context.is_logged_in and profile.has_blocked_viewer:
                            if download_profile_pic or (download_profile_content and (not profile.is_private)):
                                raise ProfileNotExistsException('{} blocked you; But we download her anonymously.'.format(target))
                            else:
                                instaloader.context.error('{} blocked you.'.format(target))
                        else:
                            profiles.add(profile)
                    except ProfileNotExistsException as err:
                        if instaloader.context.is_logged_in and (download_profile_pic or download_profile_content):
                            instaloader.context.log(err)
                            instaloader.context.log('Trying again anonymously, helps in case you are just blocked.')
                            with instaloader.anonymous_copy() as anonymous_loader:
                                with instaloader.context.error_catcher():
                                    anonymous_retry_profiles.add(anonymous_loader.check_profile_id(target, latest_stamps))
                                    instaloader.context.error('Warning: {} will be downloaded anonymously ("{}").'.format(target, err))
                        else:
                            raise
                else:
                    target_type = {'#': 'hashtag', '%': 'location', '-': 'shortcode'}.get(target[0], 'username')
                    raise ProfileNotExistsException('Invalid {} {}'.format(target_type, target))
        if len(profiles) > 1:
            instaloader.context.log('Downloading {} profiles: {}'.format(len(profiles), ' '.join([p.username for p in profiles])))
        if instaloader.context.iphone_support and profiles and (download_profile_pic or download_posts) and (not instaloader.context.is_logged_in):
            instaloader.context.log('Hint: Login to download higher-quality versions of pictures.')
        instaloader.download_profiles(profiles, download_profile_pic, download_posts, download_tagged, download_igtv, download_highlights, download_stories, fast_update, post_filter, storyitem_filter, latest_stamps=latest_stamps, reels=download_reels)
        if anonymous_retry_profiles:
            instaloader.context.log('Downloading anonymously: {}'.format(' '.join([p.username for p in anonymous_retry_profiles])))
            with instaloader.anonymous_copy() as anonymous_loader:
                anonymous_loader.download_profiles(anonymous_retry_profiles, download_profile_pic, download_posts, download_tagged, download_igtv, fast_update=fast_update, post_filter=post_filter, latest_stamps=latest_stamps, reels=download_reels)
    except KeyboardInterrupt:
        print('\nInterrupted by user.', file=sys.stderr)
        exit_code = ExitCode.USER_ABORTED
    except AbortDownloadException as exc:
        print('\nDownload aborted: {}.'.format(exc), file=sys.stderr)
        exit_code = ExitCode.DOWNLOAD_ABORTED
    if instaloader.context.is_logged_in:
        instaloader.save_session_to_file(sessionfile)
    if not targetlist:
        if instaloader.context.is_logged_in:
            instaloader.context.log('No targets were specified, thus nothing has been downloaded.')
        else:
            instaloader.context.log('usage:' + usage_string())
            exit_code = ExitCode.INIT_FAILURE
    return exit_code

def main() -> None:
    parser = ArgumentParser(description=__doc__, add_help=False, usage=usage_string(), epilog='The complete documentation can be found at https://instaloader.github.io/.', fromfile_prefix_chars='+')
    g_targets = parser.add_argument_group('What to Download', 'Specify a list of targets. For each of these, Instaloader creates a folder and downloads all posts. The following targets are supported:')
    g_targets.add_argument('profile', nargs='*', help='Download profile. If an already-downloaded profile has been renamed, Instaloader automatically finds it by its unique ID and renames the folder likewise.')
    g_targets.add_argument('_at_profile', nargs='*', metavar='@profile', help='Download all followees of profile. Requires login. Consider using :feed rather than @yourself.')
    g_targets.add_argument('_hashtag', nargs='*', metavar='"#hashtag"', help='Download #hashtag.')
    g_targets.add_argument('_location', nargs='*', metavar='%location_id', help='Download %%location_id. Requires login.')
    g_targets.add_argument('_feed', nargs='*', metavar=':feed', help='Download pictures from your feed. Requires login.')
    g_targets.add_argument('_stories', nargs='*', metavar=':stories', help='Download the stories of your followees. Requires login.')
    g_targets.add_argument('_saved', nargs='*', metavar=':saved', help='Download the posts that you marked as saved. Requires login.')
    g_targets.add_argument('_singlepost', nargs='*', metavar='-- -shortcode', help='Download the post with the given shortcode')
    g_targets.add_argument('_json', nargs='*', metavar='filename.json[.xz]', help='Re-Download the given object.')
    g_targets.add_argument('_fromfile', nargs='*', metavar='+args.txt', help='Read targets (and options) from given textfile.')
    g_post = parser.add_argument_group('What to Download of each Post')
    g_prof = parser.add_argument_group('What to Download of each Profile')
    g_prof.add_argument('-P', '--profile-pic-only', action='store