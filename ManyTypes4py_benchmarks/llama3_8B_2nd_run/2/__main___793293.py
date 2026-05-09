def _main(
    instaloader: Instaloader,
    targetlist: List[str],
    username: Optional[str] = None,
    password: Optional[str] = None,
    sessionfile: Optional[str] = None,
    download_profile_pic: bool = True,
    download_posts: bool = True,
    download_stories: bool = False,
    download_highlights: bool = False,
    download_tagged: bool = False,
    download_reels: bool = False,
    download_igtv: bool = False,
    fast_update: bool = False,
    latest_stamps_file: Optional[str] = None,
    max_count: Optional[int] = None,
    post_filter_str: Optional[str] = None,
    storyitem_filter_str: Optional[str] = None,
    browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
) -> ExitCode:
    ...

def main() -> None:
    parser = ArgumentParser(description=__doc__, add_help=False, usage=usage_string(), epilog='The complete documentation can be found at https://instaloader.github.io/.', fromfile_prefix_chars='+')
    ...
    args = parser.parse_args()
    try:
        ...
        return _main(
            Instaloader(...),
            args.profile,
            username=args.login.lower() if args.login is not None else None,
            password=args.password,
            sessionfile=args.sessionfile,
            download_profile_pic=download_profile_pic,
            download_posts=download_posts,
            download_stories=download_stories,
            download_highlights=download_highlights,
            download_tagged=download_tagged,
            download_reels=download_reels,
            download_igtv=download_igtv,
            fast_update=args.fast_update,
            latest_stamps_file=args.latest_stamps,
            max_count=int(args.count) if args.count is not None else None,
            post_filter_str=args.post_filter,
            storyitem_filter_str=args.storyitem_filter,
            browser=args.load_cookies,
            cookiefile=args.cookiefile,
        )
    except InvalidArgumentException as err:
        print(err, file=sys.stderr)
        exit_code = ExitCode.INIT_FAILURE
    ...
    sys.exit(exit_code)
