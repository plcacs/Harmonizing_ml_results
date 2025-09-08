from typing import List, Dict, Any, Optional, Union

from NEMbox.api import NetEase
from NEMbox.api import Parse

def test_api() -> None:
    api: NetEase = NetEase()
    ids: List[int] = [347230, 496619464, 405998841, 28012031]
    print(api.songs_url(ids))
    print(api.songs_detail(ids))
    print(Parse.song_url(api.songs_detail(ids)[0]))
    print(api.songs_url([561307346]))
